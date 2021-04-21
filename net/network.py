import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils import checkpoint as cp
import math
from net.droupoutNet import DropoutNet
import numpy as np
from collections import OrderedDict

class Network(nn.Module):
    def construct(self, net, obj):
        targetClass = getattr(self, net)
        instance = targetClass(obj)
        return instance

    ## Wide ResNet
    # Sourced from https://github.com/meliketoy/wide-resnet.pytorch
    def conv3x3(in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

    def conv_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)
        elif classname.find('BatchNorm') != -1:
            nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    class wide_basic(nn.Module):
        def __init__(self, in_planes, planes, dropout_rate, stride=1):
            super(Network.wide_basic, self).__init__()
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
            self.dropout = nn.Dropout(p=dropout_rate)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
                )

        def getDropoutLayer(self):
            return self.conv1

        def getImpLayer(self):
            return self.dropout

        def forward(self, x):
            out = self.dropout(self.conv1(F.relu(self.bn1(x))))
            out = self.conv2(F.relu(self.bn2(out)))
            out += self.shortcut(x)

            return out

    class Wide_ResNet(DropoutNet):
        def __init__(self, obj):
            super(DropoutNet, self).__init__()
            self.in_planes = 16

            assert ((28 - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
            n = 1
            k = 10

            print('| Wide-Resnet %dx%d' % (28, k))
            nStages = [16, 16 * k, 32 * k, 64 * k]

            self.conv1 = Network.conv3x3(obj.input_ch, nStages[0])
            if obj.prune_technique == "Dropout":
                self.layer1 = self._wide_layer(Network.wide_basic, nStages[1], n, obj.probability_list[0], stride=1)
                self.layer2 = self._wide_layer(Network.wide_basic, nStages[2], n, obj.probability_list[1], stride=2)
                self.layer3 = self._wide_layer(Network.wide_basic, nStages[3], n, obj.probability_list[2], stride=2)
            else:
                self.layer1 = self._wide_layer(Network.wide_basic, nStages[1], n, 0, stride=1)
                self.layer2 = self._wide_layer(Network.wide_basic, nStages[2], n, 0, stride=2)
                self.layer3 = self._wide_layer(Network.wide_basic, nStages[3], n, 0, stride=2)
            self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
            self.classifier = nn.Linear(nStages[3], obj.num_classes)

        def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
            strides = [stride] + [1] * (int(num_blocks) - 1)
            layers = []

            for stride in strides:
                layers.append(block(self.in_planes, planes, dropout_rate, stride))
                self.in_planes = planes

            return nn.Sequential(*layers)

        def forward(self, x):
            out = self.conv1(x)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = F.relu(self.bn1(out))
            out = F.avg_pool2d(out, 8)
            out = out.view(out.size(0), -1)
            out = self.classifier(out)

            return out

        def getLinearLayers(self):
            listOfDropout = []
            for block in self.layer1:
                listOfDropout.append(block.getDropoutLayer())
            for block in self.layer2:
                listOfDropout.append(block.getDropoutLayer())
            for block in self.layer3:
                listOfDropout.append(block.getDropoutLayer())
            return listOfDropout

        def get_imp_layers(self):
            listOfImp = []
            for block in self.layer1:
                listOfImp.append(block.getImpLayer())
            for block in self.layer2:
                listOfImp.append(block.getImpLayer())
            for block in self.layer3:
                listOfImp.append(block.getImpLayer())
            return listOfImp

        def getShapeOfImp(self):
            return [163840, 327680, 163840]

    def _bn_function_factory(norm, relu, conv):
        def bn_function(*inputs):
            concated_features = torch.cat(inputs, 1)
            bottleneck_output = conv(relu(norm(concated_features)))
            return bottleneck_output

        return bn_function

    class _DenseLayer(nn.Module):
        def __init__(self, pruneTechnique, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
            super(Network._DenseLayer, self).__init__()
            if efficient:
                momentum = 1 - math.sqrt(0.9)
            else:
                momentum = 0.1
            self.add_module('norm1', nn.BatchNorm2d(num_input_features, momentum=momentum))
            self.add_module('relu1', nn.ReLU(inplace=True))
            self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                                               kernel_size=1, stride=1, bias=False))
            self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
            self.add_module('relu2', nn.ReLU(inplace=True))
            self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                               kernel_size=3, stride=1, padding=1, bias=False))
            if pruneTechnique == "Dropout":
                self.drop_rate = float(drop_rate)
            else:
                self.drop_rate = 0

            self.add_module('drop1', nn.Dropout(p=self.drop_rate))

            self.efficient = efficient

        def forward(self, *prev_features):
            bn_function = Network._bn_function_factory(self.norm1, self.relu1, self.conv1)
            if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
                bottleneck_output = cp.checkpoint(bn_function, *prev_features)
            else:
                bottleneck_output = bn_function(*prev_features)
            new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))

            new_features = self.drop1(new_features)

            return new_features

        def getLayerBeforeDrop(self):
            return [self.conv2]

        def get_relu_layers(self):
            return [self.relu1, self.relu2]

        def get_conv_layers(self):
            return [self.conv1, self.conv2]

        def get_dropout_layers(self):
            return [self.drop1]

    class _Transition(nn.Sequential):
        def __init__(self, num_input_features, num_output_features):
            super(Network._Transition, self).__init__()
            self.add_module('norm', nn.BatchNorm2d(num_input_features))
            self.add_module('relu', nn.ReLU(inplace=True))
            self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                              kernel_size=1, stride=1, bias=False))
            self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

        def get_relu_layers(self):
            return [self.relu]

        def get_conv_layers(self):
            return [self.conv]

    class _DenseBlock(nn.Module):
        def __init__(self, pruneTechnique, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
            super(Network._DenseBlock, self).__init__()
            self.relus = []
            self.convs = []
            self.drops = []
            self.priorLayers = []
            for i in range(num_layers):
                layer = Network._DenseLayer(
                    pruneTechnique,
                    num_input_features + i * growth_rate,
                    growth_rate=growth_rate,
                    bn_size=bn_size,
                    drop_rate=drop_rate,
                    efficient=efficient,
                )
                self.relus += layer.get_relu_layers()
                self.convs += layer.get_conv_layers()
                self.drops += layer.get_dropout_layers()
                self.priorLayers += layer.getLayerBeforeDrop()
                self.add_module('denselayer%d' % (i + 1), layer)

        def forward(self, init_features):
            features = [init_features]
            for name, layer in self.named_children():
                new_features = layer(*features)
                features.append(new_features)
            return torch.cat(features, 1)

        def get_relu_layers(self):
            return self.relus

        def get_conv_layers(self):
            return self.convs

        def getLayerBeforeDrop(self):
            return self.priorLayers

        def get_dropout_layers(self):
            return self.drops

    class DenseNet(DropoutNet):
        r"""Densenet-BC model class, based on
        `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
        Args:
            growth_rate (int) - how many filters to add each layer (`k` in paper)
            block_config (list of 3 or 4 ints) - how many layers in each pooling block
            num_init_features (int) - the number of filters to learn in the first convolution layer
            bn_size (int) - multiplicative factor for number of bottle neck layers
                (i.e. bn_size * k features in the bottleneck layer)
            drop_rate (float) - dropout rate after each dense layer
            num_classes (int) - number of classification classes
            small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
            efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
        """

        def __init__(self, obj, growth_rate=12, block_config=(16, 16, 16), compression=0.5,
                     num_init_features=24, bn_size=4, drop_rate=0,
                     num_classes=100, small_inputs=True, efficient=False):
            super(DropoutNet, self).__init__()

            self.obj = obj

            assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'

            relu0 = nn.ReLU(inplace=True)
            self.relus = [relu0]
            self.drops = []
            self.layersPrior = []

            # First convolution
            self.features = nn.Sequential(OrderedDict([
                ('conv0',
                 nn.Conv2d(obj.input_ch, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
            self.features.add_module('relu0', relu0)

            self.convs = [self.features.conv0]

            # Each denseblock
            num_features = num_init_features
            for i, num_layers in enumerate(block_config):
                block = Network._DenseBlock(
                    obj.prune_technique,
                    num_layers=num_layers,
                    num_input_features=num_features,
                    bn_size=bn_size,
                    growth_rate=growth_rate,
                    drop_rate=drop_rate,
                    efficient=efficient,
                )
                self.relus += block.get_relu_layers()
                if hasattr(obj, 'first_convs') and obj.first_convs:
                    self.convs += [block.get_conv_layers()[0]]
                else:
                    self.convs += block.get_conv_layers()
                self.drops += block.get_dropout_layers()
                self.layersPrior += block.getLayerBeforeDrop()
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                if i != len(block_config) - 1:
                    trans = Network._Transition(num_input_features=num_features,
                                                num_output_features=int(num_features * compression))
                    #                    self.relus += trans.get_relu_layers()
                    #                    self.convs += trans.get_conv_layers()
                    self.features.add_module('transition%d' % (i + 1), trans)
                    num_features = int(num_features * compression)

            # Final batch norm
            self.features.add_module('norm5', nn.BatchNorm2d(num_features))

            # Linear layer
            self.classifier = nn.Linear(num_features, obj.num_classes)

            self.convs += [self.classifier]

            # Initialization
            for name, param in self.named_parameters():
                if 'conv' in name and 'weight' in name:
                    n = param.size(0) * param.size(2) * param.size(3)
                    param.data.normal_().mul_(math.sqrt(2. / n))
                elif 'norm' in name and 'weight' in name:
                    param.data.fill_(1)
                elif 'norm' in name and 'bias' in name:
                    param.data.fill_(0)
                elif 'classifier' in name and 'bias' in name:
                    param.data.fill_(0)

            self.relu = nn.ReLU(inplace=False)
            self.relus += [self.relu]

        def forward(self, x):
            features = self.features(x)
            out = self.relu(features)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            out = self.classifier(out)
            return out

        def getLinearLayers(self):
            return self.layersPrior

        def get_imp_layers(self):
            return self.drops

        def getShapeOfImp(self):
            return [32768, 32768, 8192, 8192, 2048, 2048, 512, 512]

    class DenseNet16(DenseNet):
        def __init__(self, obj):
            super(Network.DenseNet16, self).__init__(obj,
                 num_classes=obj.num_classes,
                 growth_rate=32,
                 block_config=(2, 2, 2, 2),
                 num_init_features=64,
                 drop_rate=obj.probability_list[0])