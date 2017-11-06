# FBPConvNet - Matlab

Deep Convolutional Neural Network for Inverse Problems in Imaging <br />
http://ieeexplore.ieee.org/document/7949028/

Readme

1. Before launching FBPConvNet, the MatConvNet (http://www.vlfeat.org/matconvnet/) have to be properly installed. (For the GPU, it needs a different compilation.)
2. Properly modify matconvnet path in main.m and evaluation.m files.
3. To start, download 2 links;  
(1) pretrained network : https://drive.google.com/open?id=0B9fSVcoxJuVwMVJ1eWFPdEEwbWs , then put this network into 'pretrain' folder<br />
(2) dataset : https://drive.google.com/open?id=0B9fSVcoxJuVwMDlxbXdvcTRaM2M just place this data in the same folder with main.m
4. Use main.m for training. After training, run evaluation.m for deploy of test data set.

*note : phantom data set (x20) is only provided. SNR value may be slightly different with our paper. <br />
*note : these codes mainly ran on Matlab 2016a with GPU TITAN X (architecture : Maxwell)<br />
contact : Kyong Jin (kyonghwan.jin@gmail.com), 

special thanks to Junhong Min (Senior Researcher at Samsung Electronics) for providing initial codes.
