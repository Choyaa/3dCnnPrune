# 3dCnnPrune
A lightweight deep gesture recognition model on embedded computing platform


in order to mitigate the mismatch between the powerful computing power required by the model and  the embedded hardware computing resources,
we use the channel pruning method to compact the model structure and reduce the parameters of model on the basis of 3D-SqueezeNet and 3D-MobileNetV2. Without losing accuracy, we reduce the parameters by 75%. 
MobileNetV2:2.37M ——> 47K
SqueezeNet :1.84M ——> 49K
We deploy the pruned gesture recognition model on NVIDIA® Jetson Nano™. In our experimental analysis, we achieve to recognize hand gestures containing three gesture-phonemes with an accuracy of 96\% (in 100 classes). The speed of model inferring maintains at up to 33fps.
##Jester
You can download [Jester](https://20bn.com/datasets/jester/v1)here.

## SHGD (Scaling Hand Gesture Dataset)
You can download [SHGD](https://www.mmk.ei.tum.de/shgd/) here. The dataset includes two parts:
Single gestures and 3-tuple gestures. Every record includes infrared images and depth images. 

