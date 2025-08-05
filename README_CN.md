## 1. 简介
本项目基于PyTorch 1.12.0和Python 3.9。 数据集目前采用MNIST数据集和CIFAR10数据集， 
构建模型实现了基于灰白图片（单通道）和彩色图片（三通道）的FedAvg。

## 2. 使用方法
### 2.1 FedAvg
如果你只是想使用FedAvg，修改conf.json中的noise设置为0, 运行如下代码：
```bash
python server.py -c ./utils/conf.json
```
### 2.2 差分隐私
如果你想使用基于差分隐私的FedAvg，修改conf.json中的noise设置为1（拉普拉斯机制）或者2（高斯机制），sigma用来调节噪声幅度，运行如下代码：
```bash
python server.py -c ./utils/conf.json
```
