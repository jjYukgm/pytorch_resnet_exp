'''ResNet in PyTorch.

BasicBlock and Bottleneck module is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

PreActBlock and PreActBottleneck module is from the later paper:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
'''
import torch
import models.resnet as rn
rn.test()

'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class BasicBlockD(nn.Module):
    '''
    Dropout version
    '''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, c1d=False, c2d=False):
        super(BasicBlockD, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv1_drop = nn.Dropout2d()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2_drop = nn.Dropout2d()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.c1d = c1d
        self.c2d = c2d

    def forward(self, x):
        if self.c1d:
            out = F.relu(self.bn1(self.conv1_drop(self.conv1(x))))
            out = F.dropout(out, training=self.training)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
        if self.c2d:
            out = self.bn2(self.conv2_drop(self.conv2(out)))
            out = F.dropout(out, training=self.training)
        else:
            out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(PreActBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

class ResNet2(nn.Module):
    def __init__(self, block, num_blocks, num_layers=0, num_classes=10):
        super(ResNet2, self).__init__()
        self.in_planes = 64
        self.num_layers = num_layers
        # in: 32 X 32 X 3
        self.conv1 = conv3x3(3,64)  # 32 X 32 X 64
        self.bn1 = nn.BatchNorm2d(64)
        if num_layers >0:
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)  # 32*32*(64*ex) => 64*64 = 64*2^6
        if num_layers >1:
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # 16*16*(128*ex) => 16*128 = 64*2^5
        if num_layers >2:
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)  # 8*8*(256*ex) => 4*256 = 64*2^4
        if num_layers >3:
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)  # 4*4*(512*ex) => 512 = 64*2^3
        if num_layers ==0:
            channel = 64*pow(2, 6)
        else:
            channel = 64*pow(2, 7 - num_layers)
        self.linear = nn.Linear(channel*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.num_layers >0:
            out = self.layer1(out)
        if self.num_layers >1:
            out = self.layer2(out)
        if self.num_layers >2:
            out = self.layer3(out)
        if self.num_layers >3:
            out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def r_37():
    return ResNet2(BasicBlock, [18,18,18,18], num_layers=1)
def r_37_2():
    return ResNet2(BasicBlock, [18,18,18,18], num_layers=1, num_classes=2)
def r_73():
    return ResNet2(BasicBlock, [18,18,18,18], num_layers=2)
def r_91():
    return ResNet2(BasicBlock, [18,18,9,0], num_layers=3)
def r_110():
    return ResNet2(BasicBlock, [18,18,18,18], num_layers=3)

    
class ResNet3(nn.Module):
    '''
    With dropout every layers
    '''
    def __init__(self, block, num_blocks, num_layers=0, num_classes=10, c1d=False, c2d=False):
        super(ResNet3, self).__init__()
        self.in_planes = 64
        self.num_layers = num_layers
        self.c1d = c1d
        self.c2d = c2d
        
        # in: 32 X 32 X 3
        self.conv1 = conv3x3(3,64)  # 32 X 32 X 64
        self.conv1_drop = nn.Dropout2d()
        self.bn1 = nn.BatchNorm2d(64)
        if num_layers >0:
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, c1d=c1d, c2d=c2d)  # 32*32*(64*ex) => 64*64 = 64*2^6
        if num_layers >1:
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, c1d=c1d, c2d=c2d)  # 16*16*(128*ex) => 16*128 = 64*2^5
        if num_layers >2:
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, c1d=c1d, c2d=c2d)  # 8*8*(256*ex) => 4*256 = 64*2^4
        if num_layers >3:
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, c1d=c1d, c2d=c2d)  # 4*4*(512*ex) => 512 = 64*2^3
        if num_layers ==0:
            channel = 64*pow(2, 6)
        else:
            channel = 64*pow(2, 7 - num_layers)
        self.linear = nn.Linear(channel*block.expansion, num_classes)
        self.sigfc = nn.Linear(channel*block.expansion, num_classes)
        self.sig = None

    def _make_layer(self, block, planes, num_blocks, stride, c1d, c2d):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, c1d=c1d, c2d=c2d))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.c1d:
            out = F.relu(self.bn1(self.conv1_drop(self.conv1(x))))
            out = F.dropout(out, training=self.training)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
        if self.num_layers >0:
            out = self.layer1(out)
        if self.num_layers >1:
            out = self.layer2(out)
        if self.num_layers >2:
            out = self.layer3(out)
        if self.num_layers >3:
            out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # if self.c2d:
        #     out = F.dropout(out, training=self.training)
        out1 = self.linear(out)
        sig = self.sigfc(out)
        return out1, sig
    setdrop(self, c1d, c2d):
        self.c1d = c1d
        self.c2d = c2d
        for module in self.children():
            if isinstance(module, BasicBlockD):
                module.c1d = c1d
                module.c2d = c2d
        

def r_37d():
    return ResNet3(BasicBlockD, [18,18,18,18], num_layers=1, c1d=True, c2d=True)
def r_37d2():
    return ResNet3(BasicBlockD, [18,18,18,18], num_layers=1, c1d=False, c2d=True)
def r_37d3():
    return ResNet3(BasicBlockD, [18,18,18,18], num_layers=1, c1d=False, c2d=False)
def r_110d():
    return ResNet3(BasicBlockD, [18,18,18,18], num_layers=3, c1d=True, c2d=True)

def test():
    net = r_37d()
    y = net(Variable(torch.randn(1,3,32,32)))
    if isinstance(y, tuple):
        print(y[0].size())
    else:
        print(y.size())

# test()