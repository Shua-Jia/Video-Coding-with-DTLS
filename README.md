# Video-Coding (For Demo: see below)
This is the program page of "A New Hybrid Approach Combining Traditional Method and Deep Learning for Domain Transfer in Video Compression"
we also offer a demo, you can download the demo from https://github.com/jiashaohua1993/Video-Coding/blob/main/videoSRC22_1920x1080_24.mp4
## System requirments

### Dataset
You can download training data vimeo 90k from http://toflow.csail.mit.edu/
We use UVG, MCL-JCV, and HEVC Classes B, C, D, and E as our test data.

### Training 
Train the model by following the command lines below 
```
python train.py
```

Details: 
```
--data_path: [should be filled in the directory to the training dataset] 
--ckpt_path: [the location of your training model]
--is_train: [when train, set true]
```
### Inference
After the training you can run the following command for evaluation. 
```
python eval.py
```
follow is our experiment results.

<img width="509" height="264" alt="image" src="https://github.com/user-attachments/assets/9b38fb9e-c232-45da-8f5a-170e4723cd92" />

Related Code
https://github.com/GuoLusjtu/DVC
https://github.com/microsoft/DCVC

