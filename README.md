# [__H__] Mini Project — Speech Denoising using Deep Complex Unet (DCUNet))


This README focuses on how to test our model. In order to test the dataset, follow the below steps:

- Add our model inside `dcunet_epoch_120_DNS.pth` inside `complex_checkpoints` directory
- Run this to install libraries`pip install -r requirements.txt`
- Add the test dataset inside `test_audio` directory
- Run  `python complex_local_inference.py` 
- The output file will be generated inside the `test_audio_DNS_120th_result/input-clean.wav`


