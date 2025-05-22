# TCN Gradient Trajectory Error Prediction
This is a code repository accompanying the MRM paper "Improved Image Reconstruction and Diffusion Parameter Estimation Using a Temporal Convolutional Network Model of Gradient Trajectory Errors"

| Dataset               | Waveform file             | Waveform Description            |Data Samples     |Freq. Domain FWHM (kHz)  |
|-----------------------|---------------------------|-----------------------------------------------------------------------------|
| **Training Dataset**  | train1.txt                | Chirp (up-down)                 | 1260            | 85                      |
|                       | train2.txt                | Chirp (up-down)                 | 1260            | 80                      |
|                       | train3.txt                | Chirp (up-down)                 | 1260            | 18                      |
|                       | train4.txt                | Chirp (up-down)                 | 1260            | 12                      |
|                       | train5.txt                | Triangles                       | 1250            | 6                       |
|                       | train6.txt                | Multisine                       | 1250            | 24                      |
|                       | train7.txt                | Trapezoids                      | 1250            | 13                      |
|                       | train8.txt                | Trapezoids                      | 1250            | 23                      |
|                       | train9.txt                | Slew-constrained Chirp (down)   | 1260            | 33                      |
|                       | train10.txt               | Archimedean Spiral (out)        | 1260            | (20 +/- 0.5 kHz tone)   |    
|                       | train11.txt               | Variable Density Spiral         | 1450            | (9 +/- 2 kHz tone)      |
| **Validation Dataset**| val1.txt                  | Trapezoids                      | 1260            | 13                      |
|                       | val2.txt                  | Trapezoids                      | 1260            | 29                      |
|                       | val3.txt                  | Slew-constrained Chirp (down)   | 1260            | 31                      |
|                       | val4.txt                  | Archimedean Spiral (in)         | 1260            | (20 +/- 0.5 kHz tone)   | 
|                       | val5.txt                  | Archimedean Spiral (out)        | 1260            | (3 +/- 1 kHz tone)      |
|                       | val6.txt                  | Variable Density Spiral         | 1260            | (24 +/- 4 kHz tone)     |
|                       | val7.txt                  | Variable Density Spiral         | 1260            | (23 +/- 3 kHz tone)     |
| **Test Dataset**      | rosette_mshot.txt         |  40-shot 5-petal rosette        | 527             | 6                       |
|                       | spiral_mshot.txt          |  40-shot spiral in-out 671      | 527             | 13                      |
