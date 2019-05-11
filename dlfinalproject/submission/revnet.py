# adapted from torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, expansion=4):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.expansion = expansion
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(inplanes)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, self.expansion * width)
        self.bn3 = norm_layer(width)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.relu(self.bn1(x))
        out = self.conv1(x)

        out = self.relu(self.bn2(out))
        out = self.conv2(out)

        out = self.relu(self.bn3(out))
        out = self.conv3(out)
        return out


class BottleneckRev(nn.Module):
    def __init__(self, inplanes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, expansion=4):
        super(BottleneckRev, self).__init__()
        self.bottleneck = Bottleneck(inplanes // 2, inplanes // (2 * expansion), stride, downsample, groups,
                                     base_width, dilation, norm_layer, expansion)

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        y1 = x1 + self.bottleneck(x2)
        y2 = x2
        x = torch.cat((y2, y1), dim=1)

        return x


class RevNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, filters_factor=4, last_relu=True):
        super(RevNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.filters_factor = filters_factor
        self.inplanes = 64 * self.filters_factor
        self.dilation = 1
        self.last_relu = last_relu
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn_last = norm_layer(512 * self.filters_factor)
        self.fc = nn.Linear(512 * self.filters_factor, num_classes)
        self.pool_double = nn.AvgPool2d(2, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * self.filters_factor:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes *
                        self.filters_factor, stride),
                norm_layer(planes * self.filters_factor),
            )

        layers = []
        layers.append(block(self.inplanes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, expansion=self.filters_factor))
        self.inplanes = planes * self.filters_factor
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, expansion=self.filters_factor))
        self.inplanes = self.inplanes * 2
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.pool_double(x)
        x = F.pad(x, (0, 0, 0, 0, x.size(1) // 2, x.size(1) // 2))
        x = self.layer2(x)
        x = self.pool_double(x)
        x = F.pad(x, (0, 0, 0, 0, x.size(1) // 2, x.size(1) // 2))
        x = self.layer3(x)
        x = self.pool_double(x)
        x = F.pad(x, (0, 0, 0, 0, x.size(1) // 2, x.size(1) // 2))
        x = self.layer4(x)

        x = self.bn_last(x)
        if self.last_relu:
            x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.model = RevNet(BottleneckRev, [3, 4, 6, 3], filters_factor=12)

        # Load pre-trained model
        self.load_weights('weights.pth')

    def load_weights(self, pretrained_model_path, cuda=True):
        # Load pretrained model
        pretrained_model = torch.load(
            f=pretrained_model_path, map_location="cuda" if cuda else "cpu")

        # Load pre-trained weights in current model
        with torch.no_grad():
            self.load_state_dict(pretrained_model, strict=True)

        # Debug loading
        print('Parameters found in pretrained model:')
        pretrained_layers = pretrained_model.keys()
        for l in pretrained_layers:
            print('\t' + l)
        print('')

        for name, module in self.state_dict().items():
            if name in pretrained_layers:
                assert torch.equal(pretrained_model[name].cpu(), module.cpu())
                print('{} have been loaded correctly in current model.'.format(name))
            else:
                raise ValueError("state_dict() keys do not match")

    def forward(self, x, augment=True):
        if augment:
            return F.softmax(self.model(x), dim=1) + F.softmax(self.model(torch.flip(x, [3])), dim=1)
        else:
            return F.softmax(self.model(x), dim=1)
