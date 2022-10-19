## HoD-Net: High-order Differentiable Deep Neural Networks and Applications
 (AAAI 2022)

### Installation

1. Install basic requirements (Python 3.7 or newer). Enter repository directory, run
```
pip install -r requirements.txt
```
2. Install [CuPy](https://docs.cupy.dev/en/stable/install.html):
```
(Binary Package for CUDA 9.2)
$ pip install cupy-cuda92

(Binary Package for CUDA 10.0)
$ pip install cupy-cuda100

(Binary Package for CUDA 10.1)
$ pip install cupy-cuda101

(Binary Package for CUDA 10.2)
$ pip install cupy-cuda102

(Binary Package for CUDA 11.0)
$ pip install cupy-cuda110

(Binary Package for CUDA 11.1)
$ pip install cupy-cuda111

(Binary Package for CUDA 11.2)
$ pip install cupy-cuda112

(Source Package, recommened for Linux environments)
$ pip install cupy
```



### Demo: High-order network training

First, enter the `NewtonKrylov` folder

```
cd NewtonKrylov
```

Run mnist_demo:
This demo trains a standard LeNet-5 on first ~10k samples of MNIST dataset.

```
python main.py --cfg config/LeNet/MNIST10k_demo.yaml
```

Train MNIST on LeNet:

```
python main.py --cfg config/LeNet/MNIST.yaml
```



Notice: The code for speed tests will be release later.





```
@article{shen2022hod,
  title={HoD-Net: High-order Differentiable Deep Neural Networks and Applications},
  author={Shen, Siyuan and Shao, Tianjia and Zhou, Kun and Jiang, Chenfanfu and Luo, Feng and Yang, Yin},
  year={2022}
}
```

