# Interpolating high granularity solar generation and load consumption power data using SRGAN
The code implementation for interpolating high granularity solar generation and load consumption power data using super resolution generative adversarial network (SRGAN).
## Introduction
To date, most open access public smart meter datasets are still at 30-minute or hourly temporal resolution. While this level of granularity could be sufficient for billing or deriving aggregated generation or consumption patterns, it may not fully capture the weather transients or consumption spikes. One potential solution is to synthetically interpolate high resolution data from commonly accessible lower resolution data, for this work, the SRGAN model is used for this purpose.
## Requirements
* Python 2.7.13 
* tensorflow==1.9.0
* Keras==2.2.4
* numpy==1.15.2
* pandas==0.23.4
## Datasets
## Files
## Code References
The codes are built upon the SRGAN implementation from https://github.com/deepak112/Keras-SRGAN and the DCGAN implementation from https://github.com/eriklindernoren/Keras-GAN.
