# AIWSEN
AIWSEN: Adaptive Information Weighting and Synchronized Enhancement Network for Hyperspectral Change Detection

Welcome to the repository for our paper: "AIWSEN: Adaptive Information Weighting and Synchronized Enhancement Network for Hyperspectral Change Detection."

### Requirements
torch 1.7.1 + cu101

python 3.7

torchvision 0.8.2 + cu101

imageio  2.31.2

The main function for training is AWISEN/train_HSI.py. 

The specific hyperparameter settings can be found in AWISEN/configs/configs.py.

If you need to replace the dataset, please update the dataset files in the AWISEN/datasets/ directory and modify the corresponding settings in AWISEN/data/get_dataset.py.

### Citation
Please cite our paper if you use this code in your research.

> L. Wu, J. Peng, B. Yang, W. Sun and Z. Ye, "AIWSEN: Adaptive Information Weighting and Synchronized Enhancement Network for Hyperspectral Change Detection," in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2025.3531478. 
