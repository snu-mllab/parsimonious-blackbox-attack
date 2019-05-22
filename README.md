# Code for Parsimonious Black-Box Adversarial Attacks via Efficient Combinatorial Optimization

This code is for reproducing the results in the paper "Parsimonious Black-Box Adversarial Attacks via Efficient Combinatorial Optimization" accepted at ICML 2019.

## Citing this work
```
@inproceedings{moonICML19,
    title= {Parsimonious Black-Box Adversarial Attacks via Efficient Combinatorial Optimization},
    author={Moon, Seungyong and An, Gaon and Song, Hyun Oh},
    booktitle = {Proceedings of the 36th International Conference on Machine Learning, {ICML} 2019},
    year={2019}
}
```

## Installation
* Python 3.5
* TensorFlow 1.4.0 (with GPU support)
* opencv-python
* Pillow

## Prerequisites
### Cifar-10
1. Download Cifar-10 dataset from and decompress it.
```
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvzf cifar-10-python.tar.gz
```

2. Download an adversarially trained model from [MadryLab](https://github.com/MadryLab/cifar10_challenge) and decompress it.
```
wget https://www.dropbox.com/s/g4b6ntrp8zrudbz/adv_trained.zip
unzip adv_trained.zip
```

3. Set `DATA_DIR` and `MODEL_DIR` in `cifar10/main.py` to the locations of the dataset and the model respectively.

### ImageNet
1. Download ImageNet validation dataset (images and corresponding labels). Note that the validation images must be contained within a folder named `val` and the filename of validation labels must be `val.txt`.
* For images
```
mkdir val
wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar
tar -xf ILSVRC2012_img_val.tar -C val
```
* For labels
```
wget http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz
tar -xvzf caffe_ilsvrc12.tar.gz val.txt
```

2. Place the directory `val` and the file `val.txt` in the same directory.

3. Download a pretrained Inception-v3 model from [Tensorflow model library](https://github.com/tensorflow/models/tree/master/research/slim) and decompress it.
```
wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
tar -xvzf inception_v3_2016_08_28.tar.gz
```

4. Set `IMAGENET_PATH` in `imagenet/main.py` and `MODEL_DIR` in `imagenet/tools/inception_v3_imagenet.py` to the locations of the dataset and the model respectively.

## How to run
* Cifar-10 untargeted attack
```
cd cifar10
python main.py --epsilon 8 --max_queries 20000
```

* ImageNet untargeted attack
```
cd imagenet
python main.py --epsilon 0.05 --max_queries 10000
```
* ImageNet targeted attack
```
cd imagenet
python main.py --targeted --epsilon 0.05 --max_queries 100000
```

## License
MIT License 
