// effect_nam_a2lite.h
//
// A NAM "A2-Lite" (3-channel Architecture-2 WaveNet) inference core wrapped as a
// Teensy Audio Library AudioStream effect. One mono in, one mono out.
//
// The DSP math is a faithful, dependency-free (no Eigen) float32 port of the
// Channels==3 path in sdatkinson/NeuralAmpModelerCore -> NAM/wavenet/a2_fast.cpp.
// That path is already plain unrolled scalar C++ upstream; only the 8-channel
// path uses Eigen, so the 3-channel model ports cleanly to a Cortex-M7.
//
// Weights: generate nam_weights.h 
//
// IMPORTANT — sample rate: A2 models are trained at 48 kHz. The Audio Library
// defaults to 44.1 kHz. For best fidelity run the whole graph at 48 kHz (set
// the codec to 48k and override AUDIO_SAMPLE_RATE_EXACT). Running at 44.1k
// "works" but slightly detunes the model's time constants. 
//


#pragma once
#include <Arduino.h>
#include <AudioStream.h>
#include <string.h>

class AudioEffectNamA2Lite : public AudioStream {
public:
  AudioEffectNamA2Lite() : AudioStream(1, inputQueueArray) {}

  // weights: 1871 floats in A2FastModel<3>::_load_weights order.
  // (kNamWeightCount / g_namWeights come from nam_weights.h.)
  // doPrewarm: run ~6346 silent samples to fill the dilation lines. Leave true
  // for live use; set false for bit-compare tests where you feed your own
  // leading silence and want the port to start from all-zero state.
  bool begin(const float* weights, unsigned n, bool doPrewarm = true) {
    if (n != kWeightCount) return false;
    loadWeights(weights);
    reset();
    if (doPrewarm) prewarm();
    ready = true;
    return true;
  }

  // 0..1 output trim (post head_scale). Handy as a live "amount" knob.
  void gain(float g) { outGain = g; }

  // Float in/out path (no int16 quantization). Process n <= AUDIO_BLOCK_SAMPLES
  // frames. Used by update() and exposed for testing / float audio graphs.
  void processFloat(const float* in, float* out, int n) {
    for (int f = 0; f < n; f++) {
      const float x = in[f];
      cond[f] = x;
      float* lin = &layer_in[(size_t)f * C];
      lin[0] = rechannel_w[0] * x;
      lin[1] = rechannel_w[1] * x;
      lin[2] = rechannel_w[2] * x;
    }
    memset(head_sum, 0, (size_t)n * C * sizeof(float));
    for (int li = 0; li < L; li++) layerForward(li, n);
    headForward(n);
    for (int f = 0; f < n; f++) out[f] = head_out[f] * outGain;
  }

  virtual void update(void) override {
    audio_block_t* in = receiveReadOnly(0);
    if (!in) return;
    if (!ready) { transmit(in); release(in); return; }

    audio_block_t* out = allocate();
    if (!out) { release(in); return; }

    const int n = AUDIO_BLOCK_SAMPLES;

    // int16 -> float
    float scratch_in[AUDIO_BLOCK_SAMPLES];
    float scratch_out[AUDIO_BLOCK_SAMPLES];
    for (int f = 0; f < n; f++)
      scratch_in[f] = (float)in->data[f] * (1.0f / 32768.0f);

    processFloat(scratch_in, scratch_out, n);

    // float -> int16 with clip.
    for (int f = 0; f < n; f++) {
      float y = scratch_out[f];
      y = y < -1.0f ? -1.0f : (y > 1.0f ? 1.0f : y);
      out->data[f] = (int16_t)(y * 32767.0f);
    }

    transmit(out);
    release(out);
    release(in);
  }

private:
  // ---- A2 fixed shape (must match a2_fast.h) ----
  static constexpr int C = 3;           // A2-Lite channels == bottleneck
  static constexpr int L = 23;          // layers
  static constexpr int HEAD_K = 16;
  static constexpr float SLOPE = 0.01f; // LeakyReLU

  static constexpr int kKernel[L] =
    {6,6,6,6,6,6,6,6,6,6,6,6,6,6,15,15,6,6,6,6,6,6,6};
  static constexpr int kDil[L] =
    {1,3,7,17,41,101,239,1,3,7,17,41,101,239,1,13,1,3,7,17,41,101,239};

  static constexpr int maxLookback(int li) { return (kKernel[li] - 1) * kDil[li]; }
  static constexpr int histCols(int li)    { return 2 * maxLookback(li) + AUDIO_BLOCK_SAMPLES; }

  // Total history storage for all layers + head, packed into one pool.
  static constexpr int sumHistFloats() {
    int s = 0;
    for (int li = 0; li < L; li++) s += C * histCols(li);
    s += C * (2 * (HEAD_K - 1) + AUDIO_BLOCK_SAMPLES); // head ring
    return s;
  }

  // ---- weights (internal layout) ----
  float rechannel_w[C];
  struct Layer {
    int K, D, lookback, cols;
    float conv_w[15 * C * C];   // sized for max K=15; col-major per tap: w[k*C*C + j*C + i]
    float conv_b[C];
    float mixin_w[C];
    float l1x1_w[C * C];        // col-major: w[j*C + i]
    float l1x1_b[C];
    float* hist;                // -> pool, C rows, cols columns, col-major
    int write_pos;
  } layer[L];

  float head_w[HEAD_K][C];      // head_w[k][j]
  float head_b, head_scale;
  float* head_hist;             // -> pool
  int head_write_pos;

  // ---- working buffers ----
  float cond[AUDIO_BLOCK_SAMPLES];
  float layer_in[AUDIO_BLOCK_SAMPLES * C];
  float head_sum[AUDIO_BLOCK_SAMPLES * C];
  float z[AUDIO_BLOCK_SAMPLES * C];
  float head_out[AUDIO_BLOCK_SAMPLES];

  audio_block_t* inputQueueArray[1];
  bool  ready  = false;
  float outGain = 1.0f;

  static constexpr unsigned kWeightCount = 1871;

  // History pool (~185 KB) for the dilation lines, allocated in reset().
  float* pool;

  void loadWeights(const float* w) {
    const float* p = w;
    // rechannel 1->C, no bias
    for (int i = 0; i < C; i++) rechannel_w[i] = *p++;

    for (int li = 0; li < L; li++) {
      Layer& La = layer[li];
      La.K = kKernel[li]; La.D = kDil[li];
      La.lookback = maxLookback(li); La.cols = histCols(li);
      const int K = La.K;
      // conv1d C->C, kernel K, bias.  read order i(out),j(in),k(tap)
      for (int i = 0; i < C; i++)
        for (int j = 0; j < C; j++)
          for (int k = 0; k < K; k++)
            La.conv_w[k * C * C + j * C + i] = *p++;
      for (int i = 0; i < C; i++) La.conv_b[i] = *p++;
      // mixin 1->C, no bias
      for (int i = 0; i < C; i++) La.mixin_w[i] = *p++;
      // layer1x1 C->C, bias.  read order i(out),j(in)
      for (int i = 0; i < C; i++)
        for (int j = 0; j < C; j++)
          La.l1x1_w[j * C + i] = *p++;
      for (int i = 0; i < C; i++) La.l1x1_b[i] = *p++;
    }
    // head C->1, k=16, bias.  read order j(in),k(tap)
    for (int j = 0; j < C; j++)
      for (int k = 0; k < HEAD_K; k++)
        head_w[k][j] = *p++;
    head_b = *p++;
    head_scale = *p++;
  }

  void reset() {

    static float backing[sumHistFloats()];
    pool = backing;
    memset(pool, 0, sizeof(backing));
    float* cur = pool;
    for (int li = 0; li < L; li++) {
      layer[li].hist = cur;
      cur += C * layer[li].cols;
      layer[li].write_pos = layer[li].lookback;
    }
    head_hist = cur;
    head_write_pos = HEAD_K - 1;
  }

  // Run enough silence through to fill the deepest dilation line + head.
  void prewarm() {
    int pw = HEAD_K - 1;
    for (int li = 0; li < L; li++) pw += layer[li].lookback;  // == 6346 samples
    const int n = AUDIO_BLOCK_SAMPLES;
    int done = 0;
    while (done < pw) {
      for (int f = 0; f < n; f++) {
        cond[f] = 0.0f;
        float* lin = &layer_in[(size_t)f * C];
        lin[0] = lin[1] = lin[2] = 0.0f;
      }
      memset(head_sum, 0, (size_t)n * C * sizeof(float));
      for (int li = 0; li < L; li++) layerForward(li, n);
      headForward(n);
      done += n;
    }
  }

  // Linear ring with periodic memmove-rewind (a2_fast NAM_A2_RING_MODE==0).
  void ringWrite(Layer& La, const float* src, int n) {
    if (La.write_pos + n > La.cols) {
      const int keep = La.lookback;
      memmove(La.hist, La.hist + (size_t)(La.write_pos - keep) * C,
              (size_t)keep * C * sizeof(float));
      La.write_pos = keep;
    }
    memcpy(La.hist + (size_t)La.write_pos * C, src, (size_t)n * C * sizeof(float));
    La.write_pos += n;
  }

  void layerForward(int li, int n) {
    Layer& La = layer[li];
    ringWrite(La, layer_in, n);          // write current block (becomes tap offset 0)

    const int K = La.K, D = La.D;
    const int base = La.write_pos - n;

    // conv: seed z with bias, accumulate K taps (col-major per tap).
    for (int f = 0; f < n; f++) {
      float* zf = &z[(size_t)f * C];
      zf[0] = La.conv_b[0]; zf[1] = La.conv_b[1]; zf[2] = La.conv_b[2];
    }
    for (int k = 0; k < K; k++) {
      const float* wk = &La.conv_w[(size_t)k * C * C];
      const int tap_base = base - (K - 1 - k) * D;
      for (int f = 0; f < n; f++) {
        const float* s = &La.hist[(size_t)(tap_base + f) * C];
        float* zf = &z[(size_t)f * C];
        const float s0 = s[0], s1 = s[1], s2 = s[2];
        zf[0] += wk[0]*s0 + wk[3]*s1 + wk[6]*s2;
        zf[1] += wk[1]*s0 + wk[4]*s1 + wk[7]*s2;
        zf[2] += wk[2]*s0 + wk[5]*s1 + wk[8]*s2;
      }
    }
    // mixin + LeakyReLU + head accumulate + layer1x1 residual.
    const float* mw = La.mixin_w;
    const float* lw = La.l1x1_w;  // col-major: lw[j*C + i]
    const float* lb = La.l1x1_b;
    for (int f = 0; f < n; f++) {
      float* zf = &z[(size_t)f * C];
      const float cf = cond[f];
      float a0 = zf[0] + mw[0]*cf;
      float a1 = zf[1] + mw[1]*cf;
      float a2 = zf[2] + mw[2]*cf;
      a0 = a0 >= 0.0f ? a0 : a0*SLOPE;
      a1 = a1 >= 0.0f ? a1 : a1*SLOPE;
      a2 = a2 >= 0.0f ? a2 : a2*SLOPE;
      float* hs = &head_sum[(size_t)f * C];
      hs[0] += a0; hs[1] += a1; hs[2] += a2;
      float* lin = &layer_in[(size_t)f * C];
      lin[0] += lb[0] + lw[0]*a0 + lw[3]*a1 + lw[6]*a2;
      lin[1] += lb[1] + lw[1]*a0 + lw[4]*a1 + lw[7]*a2;
      lin[2] += lb[2] + lw[2]*a0 + lw[5]*a1 + lw[8]*a2;
    }
  }

  void headForward(int n) {
    // write head_sum into head ring
    if (head_write_pos + n > 2 * (HEAD_K - 1) + AUDIO_BLOCK_SAMPLES) {
      const int keep = HEAD_K - 1;
      memmove(head_hist, head_hist + (size_t)(head_write_pos - keep) * C,
              (size_t)keep * C * sizeof(float));
      head_write_pos = keep;
    }
    memcpy(head_hist + (size_t)head_write_pos * C, head_sum,
           (size_t)n * C * sizeof(float));
    head_write_pos += n;

    const int base = head_write_pos - n;
    for (int f = 0; f < n; f++) {
      float y = head_b;
      for (int k = 0; k < HEAD_K; k++) {
        const int col = base + f - (HEAD_K - 1 - k);
        const float* s = &head_hist[(size_t)col * C];
        const float* wk = head_w[k];
        y += wk[0]*s[0] + wk[1]*s[1] + wk[2]*s[2];
      }
      head_out[f] = y * head_scale;
    }
  }
};


constexpr int AudioEffectNamA2Lite::kKernel[];
constexpr int AudioEffectNamA2Lite::kDil[];

