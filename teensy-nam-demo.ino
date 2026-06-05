// NamA2Lite_USB.ino — USB audio in -> NAM A2-Lite -> USB audio out L (processed),
//                      USB audio in (dry) -> USB audio out R, for an A/B over USB.
//
// REQUIRED: Tools -> USB Type -> "Audio" (or any option that includes Audio,
// e.g. "Serial + MIDI + Audio"). Without it AudioInputUSB/AudioOutputUSB won't
// enumerate.
//
// The Teensy appears to your computer as a stereo USB audio device ("Teensy
// Audio") at 44.1 kHz. Send a DI to the Teensy's output (left channel is used);
// record/monitor its input: left is the model output, right is the dry DI.
//
// Put these three files in the same sketch folder:
//   NamA2Lite_USB.ino
//   effect_nam_a2lite.h
//   nam_weights.h   (defines g_namDeluxeReverb, kNamWeightCount)

#include <Audio.h>
#include "effect_nam_a2lite.h"
#include "nam_weights.h"

AudioInputUSB         usbIn;    // stereo audio from host
AudioEffectNamA2Lite  nam;      // mono amp model
AudioOutputUSB        usbOut;   // stereo audio back to host
AudioOutputMQS        mqs1; 

AudioConnection  p1(usbIn, 0, nam, 0);     // host left -> nam
AudioConnection  p2(nam,   0, usbOut, 0);  // nam   -> USB out L (processed)
AudioConnection  p3(usbIn, 0, usbOut, 1);  // host left -> USB out R (dry)

void setup() {
  Serial.begin(115200);
  AudioMemory(24);

  nam.gain(1.0f);
  if (!nam.begin(g_namDeluxeReverb, kNamWeightCount)) {
    Serial.println("NAM begin() failed - weight count mismatch?");
    while (1) {}
  }
  Serial.println("USB audio -> NAM A2-Lite -> USB audio, running.");
}

void loop() {
  Serial.printf("CPU %.1f%% (max %.1f%%)   blocks %d (max %d)\n",
                AudioProcessorUsage(), AudioProcessorUsageMax(),
                AudioMemoryUsage(), AudioMemoryUsageMax());
  AudioProcessorUsageMaxReset();
  delay(1000);
}
