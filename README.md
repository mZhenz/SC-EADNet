# SC-EADNet
The Official PyTorch implementation of SC-EADNet on TGRS 2022 paper [SC-EADNet: A Self-Supervised Contrastive Efficient Asymmetric Dilated Network for Hyperspectral Image Classification](https://ieeexplore.ieee.org/document/9627700).

## Requirements
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
pip install tqdm pandas scipy numpy
```

## Dataset
`Indian Pines`, `Salinas`, `PaviaU`, `Houston 2013` datasets are used in this repo, the dataset should be downloaded into `dataset/HSIdata` directory.
### Prepare Dataset
```bash
cd dataset
python makeTrainingSet.py
```

## Usage
### Train SC-EADNet
```
python main.py --batch_size 128 --epochs 100 
```

### Linear Evaluation
```
python linear.py --model_path /path/to/pretrained/checkpoint/
```

## Citation
```
@ARTICLE{9627700,  
author={Zhu, Mingzhen and Fan, Jiayuan and Yang, Qihang and Chen, Tao},  
journal={IEEE Transactions on Geoscience and Remote Sensing},   
title={SC-EADNet: A Self-Supervised Contrastive Efficient Asymmetric Dilated Network for Hyperspectral Image Classification},   
year={2022},  
volume={60},  
number={},  
pages={1-17},  
doi={10.1109/TGRS.2021.3131152}}
```


