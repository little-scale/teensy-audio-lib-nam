# teensy-audio-lib-nam
NAM A2-lite inference for Teensy Audio Library

Use nam_a2lite_header.html to convert an existing A2 model .nam file (full + lite) into the extracted 3ch weights arrayn of the lite model as a .h file that can be used directly by the Teensy. 

A2 models are trained at 48 kHz. The Audio Library defaults to 44.1 kHz. For best fidelity run the whole graph at 48 kHz (set the codec to 48k and override AUDIO_SAMPLE_RATE_EXACT). Running at 44.1k "works" but slightly detunes the model's time constants. 
