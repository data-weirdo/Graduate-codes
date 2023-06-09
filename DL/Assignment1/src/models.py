import torch
import torch.nn as nn
import torch.nn.functional as F

#########################################
# Three classes are defined in model.py #
# : MLP, VGG, RESNET                    #
#########################################

#####################
######## MLP ########
#####################

class MLP(nn.Module):
    def __init__(self, reg_args, dropout_rate=0):
        super(MLP, self).__init__()
        
        layers = []
        in_dim = 3*32*32
        for dim in [1024, 512, 256, 128]:
            out_dim = dim
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            if reg_args==1:
                layers.append(nn.Dropout(p=dropout_rate))
            in_dim = out_dim
        layers.append(nn.Linear(out_dim, 2))
        self.network = nn.Sequential(*layers)
    
    def forward(self, img):
        logits = self.logit(img)
        probas = F.softmax(logits, dim=1)
        return logits, probas

    def logit(self, img):
        return self.network(img.view(len(img), -1))
    

#######################################################################################################################
#######################################################################################################################

# Source: https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py

'''VGG11/13/16/19 in Pytorch.'''

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, reg_args, dropout_rate=0):
        super(VGG, self).__init__()
        self.reg_args = reg_args
        self.dropout_rate = dropout_rate
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 2)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        probas = F.softmax(logits, dim=1)
        return logits, probas

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                if self.reg_args == 1:
                    layers += [nn.Dropout(p=self.dropout_rate)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def VGG11(reg_args=None, dropout_rate=None):
    return VGG('VGG11', reg_args=reg_args, dropout_rate=dropout_rate)


def VGG13(reg_args=None, dropout_rate=None):
    return VGG('VGG13', reg_args=reg_args, dropout_rate=dropout_rate)


def VGG16(reg_args=None, dropout_rate=None):
    return VGG('VGG16', reg_args=reg_args, dropout_rate=dropout_rate)


def VGG19(reg_args=None, dropout_rate=None):
    return VGG('VGG19', reg_args=reg_args, dropout_rate=dropout_rate)


#######################################################################################################################
#######################################################################################################################


# Source: https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py

'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, reg_args, num_classes=2, dropout_rate=0):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.dropout = nn.Dropout(p=dropout_rate)
        self.reg_args = reg_args

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
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
        if self.reg_args == 1:
            out = self.dropout(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        probas = F.softmax(logits, dim=1)
        return logits, probas

def ResNet18(reg_args, dropout_rate):
    return ResNet(BasicBlock, [2, 2, 2, 2], reg_args=reg_args, dropout_rate=dropout_rate)


def ResNet34(reg_args, dropout_rate):
    return ResNet(BasicBlock, [3, 4, 6, 3], reg_args=reg_args, dropout_rate=dropout_rate)


def ResNet50(reg_args, dropout_rate):
    return ResNet(Bottleneck, [3, 4, 6, 3], reg_args=reg_args, dropout_rate=dropout_rate)


def ResNet101(reg_args, dropout_rate):
    return ResNet(Bottleneck, [3, 4, 23, 3], reg_args=reg_args, dropout_rate=dropout_rate)


def ResNet152(reg_args, dropout_rate):
    return ResNet(Bottleneck, [3, 8, 36, 3], reg_args=reg_args, dropout_rate=dropout_rate)