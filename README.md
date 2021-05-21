# AI6126 Project 1: CelebA Facial Attribute Recognition Challenge
40 face attributes prediction on [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) benchmark with PyTorch Implementation.   
The challange is to deal with domain gap and imbalanced data of the dataset.    
Three effective ways are summarized in my [blog article](https://bozliu.medium.com/three-effective-ways-to-deal-with-domain-gap-and-imbalanced-data-in-multi-class-classification-1949067ac374).    

## Baseline 

[Face attribute prediction](https://github.com/d-li14/face-attribute-prediction) 

## Dependencies

* Anaconda3 (Python 3.7.6, with Numpy etc.)
* PyTorch 1.6
* tensorboard, tensorboardX

## Dataset

* [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset is a large-scale face dataset with attribute-based annotations. Cropped and aligned face regions are utilized as the training source. 
* Pre-processed data and specific split list has been uploaded to [list](list) directory.
* [lfwA+](http://vis-www.cs.umass.edu/lfw/) dataset is the private test dataset. 

## Data Pre-processing 

After downloading the dataset, please unzip img_align_celeba. Put the folder of Anno with two files, list_eval_partition.txt and list_attr_celeba.txt inside. The folder of img_align_celeba and Anno are in the same directory with `Data Pre-processing.py`

## Dataset Structure

├── Anno  
    ├── identity_CelebA.txt  
    ├── list_attr_celeba.txt  
    ├── list_bbox_celeba.txt  
    ├── list_eval_partition.txt  
    ├── list_landmarks_align_celeba.txt  
    └── list_landmarks_celeba.txt  
    
├── Eval  
    └── list_eval_partition.txt

├── img_align_celeba  [Face Pictures]    
├── test_attr_list.txt  
├── train_attr_list.txt  
└── val_attr_list.txt  

## Checkpoints
1. **checkpoints_CrossEntropyLoss20** : Training with Cross Entropy loss for 20 epochs 
2. **checkpoints_FocalLoss20**: Training with focal loss for 20 epochs
3. **checkpoints+CrossEntropy20+FocalLoss7**:  Training with Cross Entropy loss for 20 epochs and continue to train with focal loss for another 7 epochs 


## Run Scripts 

### Train
`! python main.py -d './img_align_celeba' --resume './face-attribute-prediction/checkpoints/checkpoint.pth.tar' --gpu-id '0,1'`

`-d` refers to the location of image dataset, img_align_celeba  
`--resume` refers to the location of checkpoints   
`--gpu-id` refers to the GPU number on the cluster    

### Validation 
`! python main.py -d './img_align_celeba' -e --gpu-id '0,1' `   
-e refers to the validation mode

### Test 
`! python main.py -d './img_align_celeba' -t --gpu-id '0,1' `   
-t refers to the test mode


## Results

* Table1. Average accuracy with the combined loss function, cross-entropy loss for 20 epochs and focal loss for 7 epochs 
<table class="table table-bordered table-striped table-condensed">
   <tr>
      <td></td>
      <td>Validation Accuracy </td>
      <td>Test Accuracy </td>
   </tr>
   <tr>
      <td>Average Accuracy </td>
      <td>91.3018 </td>
      <td>90.9309 </td>
   </tr>
</table>


* Table 2 Average accuracy for single loss function 
<table class="table table-bordered table-striped table-condensed">
   <tr>
      <td>Single Loss Function </td>
      <td>Maximum Training epochs </td>
      <td>Validation Accuracy </td>
      <td>Test Accuracy </td>
   </tr>
   <tr>
      <td>Cross-Entropy</td>
      <td>20</td>
      <td>90.7780 </td>
      <td>90.5587 </td>
   </tr>
   <tr>
      <td>Focal Loss</td>
      <td>20</td>
      <td>91.1635 </td>
      <td>90.7420 </td>
   </tr>
</table>


* Table 3. Accuracy for each attribute with the combined two loss function (cross-entropy loss for 20 epochs and focal loss for 7 epochs)
<table class="table table-bordered table-striped table-condensed">
   <tr>
      <td>Attributes </td>
      <td>Validation Accuracy </td>
      <td>Test Accuracy </td>
   </tr>
   <tr>
      <td>5 o’clock Shadow</td>
      <td>93.8139 </td>
      <td>94.2240 </td>
   </tr>
   <tr>
      <td>Arched Eyebrows</td>
      <td>83.9936 </td>
      <td>82.7322 </td>
   </tr>
   <tr>
      <td>Attractive</td>
      <td>80.5356 </td>
      <td>82.1210 </td>
   </tr>
   <tr>
      <td>Bags Under Eyes</td>
      <td>83.9332 </td>
      <td>84.3753 </td>
   </tr>
   <tr>
      <td>Bald</td>
      <td>98.7920 </td>
      <td>98.8077 </td>
   </tr>
   <tr>
      <td>Bangs</td>
      <td>95.6460 </td>
      <td>95.5265 </td>
   </tr>
   <tr>
      <td>Big Lips</td>
      <td>82.4986 </td>
      <td>71.5109 </td>
   </tr>
   <tr>
      <td>Big Nose</td>
      <td>82.4936 </td>
      <td>83.6389 </td>
   </tr>
   <tr>
      <td>Black Hair</td>
      <td>91.3928 </td>
      <td>89.7305 </td>
   </tr>
   <tr>
      <td>Blond Hair</td>
      <td>95.3591 </td>
      <td>95.8271 </td>
   </tr>
   <tr>
      <td>Blurry</td>
      <td>96.3105 </td>
      <td>96.0876 </td>
   </tr>
   <tr>
      <td>Brown Hair</td>
      <td>85.7603 </td>
      <td>89.2997 </td>
   </tr>
   <tr>
      <td>Bushy Eyebrows</td>
      <td>92.1679 </td>
      <td>92.3404 </td>
   </tr>
   <tr>
      <td>Chubby</td>
      <td>95.1981 </td>
      <td>95.3411 </td>
   </tr>
   <tr>
      <td>Double Chin</td>
      <td>96.2652 </td>
      <td>96.1427 </td>
   </tr>
   <tr>
      <td>Eyeglasses</td>
      <td>99.3507 </td>
      <td>99.4590 </td>
   </tr>
   <tr>
      <td>Goatee</td>
      <td>96.2853 </td>
      <td>97.3299 </td>
   </tr>
   <tr>
      <td>Gray Hair</td>
      <td>97.7551 </td>
      <td>98.0212 </td>
   </tr>
   <tr>
      <td>Heavy Makeup</td>
      <td>91.8206 </td>
      <td>91.1332 </td>
   </tr>
   <tr>
      <td>High cheekbones</td>
      <td>87.7133 </td>
      <td>86.8751 </td>
   </tr>
   <tr>
      <td>Male</td>
      <td>98.1376 </td>
      <td>97.8609 </td>
   </tr>
   <tr>
      <td>Mouth Slightly Open</td>
      <td>93.4867 </td>
      <td>93.1770 </td>
   </tr>
   <tr>
      <td>Mustache</td>
      <td>96.1595 </td>
      <td>96.7588 </td>
   </tr>
   <tr>
      <td>Narrow Eyes</td>
      <td>92.6612 </td>
      <td>87.0905 </td>
   </tr>
   <tr>
      <td>No Beard</td>
      <td>95.6863 </td>
      <td>95.6217 </td>
   </tr>
   <tr>
      <td>Oval Face</td>
      <td>75.3914 </td>
      <td>74.9775 </td>
   </tr>
   <tr>
      <td>Pale Skin</td>
      <td>96.8943 </td>
      <td>97.1245 </td>
   </tr>
   <tr>
      <td>Pointy Nose</td>
      <td>76.4031 </td>
      <td>76.4703 </td>
   </tr>
   <tr>
      <td>Receding Hairline</td>
      <td>94.3474 </td>
      <td>93.3774 </td>
   </tr>
   <tr>
      <td>Rosy Cheeks</td>
      <td>95.0018 </td>
      <td>95.1007 </td>
   </tr>
   <tr>
      <td>Sideburns</td>
      <td>96.9094 </td>
      <td>97.6155 </td>
   </tr>
   <tr>
      <td>Smiling</td>
      <td>92.4800 </td>
      <td>92.2553 </td>
   </tr>
   <tr>
      <td>Straight Hair</td>
      <td>82.8963 </td>
      <td>82.7071 </td>
   </tr>
   <tr>
      <td>Wavy Hair</td>
      <td>84.5321 </td>
      <td>83.0578 </td>
   </tr>
   <tr>
      <td>Wearing Earrings</td>
      <td>90.4817 </td>
      <td>89.5301 </td>
   </tr>
   <tr>
      <td>Wearing Hat</td>
      <td>98.7869 </td>
      <td>98.9831 </td>
   </tr>
   <tr>
      <td>Wearing Lipstick</td>
      <td>92.2384 </td>
      <td>93.6279 </td>
   </tr>
   <tr>
      <td>Wearing Necklace</td>
      <td>88.6646 </td>
      <td>86.7398 </td>
   </tr>
   <tr>
      <td>Wearing Necktie</td>
      <td>96.2249 </td>
      <td>96.7188 </td>
   </tr>
   <tr>
      <td>Young</td>
      <td>87.6026 </td>
      <td>87.9170 </td>
   </tr>
</table>


## Detail File Structure

├── celeba.py    
├── checkpoints  
    ├── checkpoint.pth.tar  
    ├── log.eps  
    ├── logs  
    ├── log.txt  
    ├── model_best.pth.tar  
├── checkpoints_CrossEntropyLoss20  
├── checkpoints_FocalLoss20  
├── focal_loss.py  
├── LICENSE  
├── main.py  
├── models  
    ├── __init__.py  
    ├── mobilenetv1.py  
    ├── mobilenetv2.py  
    └── resnet.py  
├── Model Structure  
├── Model Summary  
├── **prediction.txt**  
├── README.md  
└── utils  

* `celeba.py` performs a map-style dataset representing a map from indices/keys to data samples used as one of inputs for DataLoader.
* *checkpoints* is to save the checkpoint of model in each epoch and also to save the model with highest validation accuracy so far. The loss function of the saved model checkpoints folder are two combined loss function with cross-entropy loss for the first 20 epochs and focal loss for the next 7 epochs
* *checkpoints_CrossEntropyLoss20* is to save the checkpoint of model with highest validation accuracy during the training with single loss function, cross-entropy loss for 20 epochs.
* *checkpoints_FocalLoss20* is to save the checkpoint of model with highest validation accuracy during the training with single loss function, focal loss for 20 epochs.
* `focal_loss.py` implements the focal loss function.
* `main.py` is the main function of the project
* models is the folder with different model structure. `resnet.py` is the one used in this project.
* Model Structure print the model, ResNet-50 used in this project
* Model Summary prints all layers in ResNet-50, the output shape and parameters of each layers, and the total parameters used for an single image input.
* **prediction.txt**  is predictions of the given new test set
* utils are the utility function required by the project
