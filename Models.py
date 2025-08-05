import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
'''
MNIST的卷积神经网络模型
'''
class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义每一层模型
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(3 * 3 * 64, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, inputs):
        # 构造模型
        tensor = inputs.view(-1, 1, 28, 28)
        tensor = F.relu(self.conv1(tensor))
        tensor = self.pool1(tensor)
        tensor = F.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        tensor = F.relu(self.conv3(tensor))
        tensor = tensor.view(-1, 3 * 3 * 64)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        return tensor


'''
cifar简单卷积神经网络模型
'''
class Cifar_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义每一层模型
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(8 * 8 * 128, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, inputs):
        # 构造模型
        tensor = inputs.view(-1, 3, 32, 32)
        tensor = F.relu(self.conv1(tensor))
        tensor = self.pool1(tensor)
        tensor = F.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        tensor = F.relu(self.conv3(tensor))
        # print(tensor.shape)
        # raise(1)
        tensor = tensor.view(-1, 8 * 8 * 128)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        return tensor


'''
mnist的LeNet-5模型
'''


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(16, 120, kernel_size=4)

        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, inputs):
        tensor = inputs.view(-1, 1, 28, 28)
        tensor = self.pool1(F.relu(self.conv1(tensor)))
        tensor = self.pool2(F.relu(self.conv2(tensor)))
        tensor = F.relu(self.conv3(tensor))
        tensor = tensor.view(tensor.size(0), -1)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        return tensor


'''
resnet18网络模型
'''


class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, inputs):
        tensor = self.conv1(inputs)
        tensor = F.relu(self.bn1(tensor))
        tensor = self.conv2(tensor)
        tensor = self.bn2(tensor)
        return F.relu(inputs + tensor)


class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, input):
        extra_x = self.extra(input)
        tensor = self.conv1(input)
        tensor = F.relu(self.bn1(tensor))
        tensor = self.conv2(tensor)
        tensor = self.bn2(tensor)
        return F.relu(extra_x + tensor)


class RestNet18(nn.Module):
    def __init__(self):
        super(RestNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(RestNetBasicBlock(64, 64, 1),
                                    RestNetBasicBlock(64, 64, 1))

        self.layer2 = nn.Sequential(RestNetDownBlock(64, 128, [2, 1]),
                                    RestNetBasicBlock(128, 128, 1))

        self.layer3 = nn.Sequential(RestNetDownBlock(128, 256, [2, 1]),
                                    RestNetBasicBlock(256, 256, 1))

        self.layer4 = nn.Sequential(RestNetDownBlock(256, 512, [2, 1]),
                                    RestNetBasicBlock(512, 512, 1))

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(512, 10)

    def forward(self, inputs):
        tensor = inputs.view(-1, 3, 32, 32)
        tensor = self.conv1(tensor)
        tensor = self.layer1(tensor)
        tensor = self.layer2(tensor)
        tensor = self.layer3(tensor)
        tensor = self.layer4(tensor)
        tensor = self.avgpool(tensor)
        tensor = tensor.reshape(tensor.shape[0], -1)
        tensor = self.fc(tensor)
        return tensor


'''
resnet20网络模型
'''

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet20(nn.Module):
    """
    ResNet-20 architecture for CIFAR-10.
    Depth = 6n+2, here n=3 -> [3,3,3] BasicBlock per stage.
    """
    def __init__(self, num_classes=10):
        super(ResNet20, self).__init__()
        self.in_planes = 16
        # Initial conv layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        # Residual stages
        self.layer1 = self._make_layer(16, 3, stride=1)
        self.layer2 = self._make_layer(32, 3, stride=2)
        self.layer3 = self._make_layer(64, 3, stride=2)
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * BasicBlock.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)




'''
MIT-BIH数据集的卷积神经网络模型
'''


class SimpleCNN1D(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleCNN1D, self).__init__()
        # 输入: [batch, 187]
        # 第一层：16 个 1D 卷积核，kernel_size=7，padding=3 -> [batch, 16, 187]
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, padding=3)
        # 池化：kernel_size=2 -> [batch, 16, 93]
        self.pool  = nn.MaxPool1d(kernel_size=2)
        # 全连接层：16*93 -> 128
        self.fc1   = nn.Linear(16 * 93, 128)
        # 输出层
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: [batch, 187]
        x = x.unsqueeze(1)           # -> [batch, 1, 187]
        x = F.relu(self.conv1(x))    # -> [batch,16,187]
        x = self.pool(x)             # -> [batch,16, 93]
        x = x.view(x.size(0), -1)    # -> [batch,16*93]
        x = F.relu(self.fc1(x))      # -> [batch,128]
        logits = self.fc2(x)         # -> [batch,num_classes]
        return logits




class LeNet1D(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # 输入: [batch, 1, 187]
        self.conv1 = nn.Conv1d(1, 6, kernel_size=5)    # -> [batch,6,183]
        self.pool1 = nn.AvgPool1d(kernel_size=2)       # -> [batch,6,91]
        self.conv2 = nn.Conv1d(6, 16, kernel_size=5)   # -> [batch,16,87]
        self.pool2 = nn.AvgPool1d(kernel_size=2)       # -> [batch,16,43]
        # 用一个 kernel=43 的 conv3 把时序维度降到 1
        self.conv3 = nn.Conv1d(16, 120, kernel_size=43)  # -> [batch,120,1]
        self.fc1  = nn.Linear(120, 84)
        self.fc2  = nn.Linear(84, num_classes)

    def forward(self, x):
        # x 原始形状: [batch, 187]
        x = x.unsqueeze(1)               # -> [batch,1,187]
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))        # -> [batch,120,1]
        x = x.view(x.size(0), -1)        # -> [batch,120]
        x = F.relu(self.fc1(x))
        return self.fc2(x)