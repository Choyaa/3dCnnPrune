# 3dCnnPrune

### Requirements

- Python 3.6
- Pytorch 0.4 or later versions.
- JetPack 4.5.1
- CUDA 10.2
- Other packages e.g. numpy, etc.

### Overview 

This project includes training using two 3D models SqueezeNet version 1.1 and MobileNetV2, which are adapted from their 2D models.

You can pretrain SqueezeNet 1.1 and MobileNetV2 with Jester, fine tune using SHGD15/13 and test models on SHGD Tuples.

### Dataset 

- [Jester v1](https://20bn.com/datasets/jester) for pretraining. Please follow the download instructions. 
- SHGD15/SHGD13 for training and validation. 
- SHGD Tuples for testing. 

### Training

**Before running the code, please give the paths to the datasets and the name of their annotations files!**

Here are some examples for (pre)training with Jester and fine-tuning with SHGD15/SHGD13 using SqueezeNet 1.1. 

- Pretraining with Jester: using SqueezeNet 1.1

  ```
  python main.py --root_path ~/  \
                 --video_path ~/Jester/20bn-jester-v1 \
                 --annotation_path GeScale_3D/Jester \
                 --result_path GeScale_3D/results/Jester \
                 --train_list 'train_videofolder.txt' \
                 --val_list 'val_videofolder.txt'  \
                 --dataset jester \
                 --modality 'RGB' \
                 --n_classes 27 \
                 --model squeezenet \
                 --train_crop random \
                 --learning_rate 0.1 \
                 --sample_duration 8 \
                 --batch_size 64 \
                 --n_epochs 60 \
  ```

- Training with SHGD 15: using pretrained (with jester) models; inputs are IR and Depth images.
**Please give the paths to the datasets and the name of their annotations files and also the pre**

  ```
  python main.py --root_path ~/  \
                 --video_path ~/SHGD/SHGD_Single \
                 --annotation_path GeScale_3D/SHGD \
                 --result_path GeScale_3D/results/SHGD_15 \
                 --train_list 'SHGD15_trainlist.txt' \
                 --val_list 'SHGD15_vallist.txt'  \
                 --dataset SHGD \
                 --modality 'IRD' \
                 --n_classes 27 \
                 --n_finetune_classes 15 \
                 --model squeezenet \
                 --train_crop random \
                 --learning_rate 0.1 \
                 --sample_duration 8 \
                 --batch_size 64 \
                 --n_epochs 60 \
                 --pretrain_path 'GeScale_3D/results/Jester/jester_squeezenet_RGB_8_best.pth' 
  ```

- Sparsity  with SHGD13: using pretrained (with SHGD15) models; inputs are IR and Depth images. 

  ```
  python main.py --root_path ~/  
                 --video_path ~/SHGD/SHGD_Single \
                 --annotation_path GeScale_3D/SHGD \
                 --result_path GeScale_3D/results/SHGD13_sparsity \
                 --train_list 'SHGD13_trainlist.txt' \
                 --val_list 'SHGD13_vallist.txt'  \
                 --dataset SHGD \
                 --modality 'IRD' \
                 --n_classes 15 \
                 --n_finetune_classes 13 \
                 --same_modality_finetune \
                 --model squeezenet \
                 --train_crop random \
                 --learning_rate 0.1 \
                 --sample_duration 8 \
                 --batch_size 64 \
                 --n_epochs 60 \
                 --pretrain_path 'GeScale_3D/results/SHGD_15/SHGD_squeezenet_IRD_8_best.pth'
  ```
- finetune with SHGD13:using pretrained (with SHGD13_sparsity) models; inputs are IR and Depth images.
  ```
  python main.py --root_path ~/  
                 --video_path ~/SHGD/SHGD_Single \
                 --annotation_path GeScale_3D/SHGD \
                 --result_path GeScale_3D/results/SHGD13_prune \
                 --train_list 'SHGD13_trainlist.txt' \
                 --val_list 'SHGD13_vallist.txt'  \
                 --dataset SHGD \
                 --modality 'IRD' \
                 --n_classes 13 \
                 --n_finetune_classes 13 \
                 --same_modality_finetune \
                 --model squeezenet \
                 --train_crop random \
                 --learning_rate 0.1 \
                 --sample_duration 8 \
                 --batch_size 64 \
                 --n_epochs 60 \
                 --pretrain_path 'GeScale_3D/results/SHGD13_sparsity/SHGD_squeezenet_IRD_8_best.pth'
  ```
### Test

For testing to recognize gesture tuples in SHGD Tuples, Please use the models trained with SHGD13. 

- Test with SHGD Tuples: using MobileNetV2; inputs are IR images only.

  ```
  python main.py --root_path ~/  \
                 --video_path ~/SHGD/SHGD_Tuples \
                 --annotation_path GeScale_3D/SHGD \
                 --result_path GeScale_3D/results \
                 --dataset SHGD \
                 --modality 'IRD' \
                 --test \
                 --test_list 'test_list.txt' \
                 --n_classes 13 \
                 --model squeezenet \
                 --sample_duration 8 \
                 --resume_path 'GeScale_3D/results/SHGD13_prune/SHGD_squeezenet_IRD_8_best.pth'
  ```

### Acknowledgment

Thank project [GeScale](https://github.com/yaorong0921/GeScale), this work is based on the framework of it.
