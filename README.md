# teensy-audio-lib-nam
NAM A2-lite inference for Teensy Audio Library

IMPORTANT — sample rate: A2 models are trained at 48 kHz. The Audio Library defaults to 44.1 kHz. For best fidelity run the whole graph at 48 kHz (set the codec to 48k and override AUDIO_SAMPLE_RATE_EXACT). Running at 44.1k "works" but slightly detunes the model's time constants. 
