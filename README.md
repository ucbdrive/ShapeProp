## Learning Saliency Propagation for Semi-supervised Instance Segmentation

![illustration](illustration.png)

## PyTorch Implementation
This repository contains:
* the **PyTorch** implementation of ShapeProp.
* the **Classwise semi-supervision (COCO's VOC->Non-VOC)** demo.

Please follow the instruction below to install it and run the experiment demo.

### Prerequisites
* Linux (tested on ubuntu 16.04LTS)
* NVIDIA GPU + CUDA CuDNN (tested on 8x GTX 2080 Ti)
* [COCO 2017 Dataset](http://cocodataset.org) (download and unzip)
* Please use PyTorch1.1 + Apex(#1564802) to avoid compilation errors

### Getting started

0. Create a conda environment:
    ```bash
    conda create --name ShapeProp -y
    conda activate ShapeProp
    ```

1. Clone this repo: 

    ```bash
    # git version must be greater than 1.9.10
    git clone https://github.com/ucbdrive/ShapeProp.git
    cd ShapeProp
    export DIR=$(pwd)
    ```

2. Install dependencies via a single command `bash $DIR/scripts/install.sh` or do it manually as follows:
    ```bash
    # Python
    conda install -y ipython pip
    # PyTorch
    conda install -y pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
    # Install deps
    pip install ninja yacs cython matplotlib tqdm opencv-python
    rm -r libs
    mkdir libs
    # COCOAPI
    cd $DIR/libs
    git clone https://github.com/cocodataset/cocoapi.git
    cd cocoapi/PythonAPI
    python setup.py build_ext install
    # APEX
    cd $DIR/libs
    git clone https://github.com/NVIDIA/apex.git
    cd apex
    python setup.py install --cuda_ext --cpp_ext
    # ShapeProp
    cd $DIR
    python setup.py build develop

    ```

3. Prepare dataset: 

    ```bash
    cd $DIR
    mkdir datasets
    ln -s PATH_TO_YOUR_COCO_DATASET datasets/coco
    bash scripts/prepare_data.sh
    ```

4. Run the classwise semi-supervision demo:

    ```bash
    cd $DIR
    # Mask R-CNN w/ ShapeProp
    bash scripts/train_shapeprop.sh
    # Mask R-CNN
    bash scripts/train_baseline.sh
    ```

## Citation 
If you use the code in your research, please cite:
```bibtex
@INPROCEEDINGS{Zhou2020ShapeProp,
    author = {Zhou, Yanzhao and Wang, Xin and and Jiao, Jianbin and Darrell, Trevor and Yu, Fisher},
    title = {Learning Saliency Propagation for Semi-supervised Instance Segmentation},
    booktitle = {CVPR},
    year = {2020}
}
```
