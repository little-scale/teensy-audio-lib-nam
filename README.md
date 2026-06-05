# teensy-audio-lib-nam
NAM A2-lite inference for Teensy Audio Library

Use nam_a2lite_header.html to convert an existing A2 model .nam file (full + lite) into the extracted 3ch weights arrayn of the lite model as a .h file that can be used directly by the Teensy. 

A2 models are trained at 48 kHz. The Audio Library defaults to 44.1 kHz. Running at 44.1k "works" but slightly detunes the model's time constants. 
