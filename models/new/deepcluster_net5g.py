import torch.nn as nn
from residual import BasicBlock, ResNet, ResNetTrunk
from make_sobel import make_sobel

# resnet34 and full channels
__all__ = [ 'deepcluster_net5g']


class DeepClusterNet5gTrunk(ResNetTrunk):
    def __init__(self, sobel, input_ch, input_sp_sz):
        super(DeepClusterNet5gTrunk, self).__init__()

        block = BasicBlock
        layers = [3, 4, 6, 3]

        in_channels = input_ch
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if input_sp_sz == 96:
            avg_pool_sz = 7
        elif input_sp_sz == 64:
            avg_pool_sz = 5
        elif input_sp_sz == 32:
            avg_pool_sz = 3
        print("avg_pool_sz %d" % avg_pool_sz)

        self.avgpool = nn.AvgPool2d(avg_pool_sz, stride=1)

        self.use_sobel = sobel
        self.sobel = None

    def forward(self, x):
        if self.use_sobel:
            x = self.sobel(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

class DeepClusterNet5g(ResNet):
    def __init__(self, sobel, out, input_sp_sz, input_ch):
        super(DeepClusterNet5g, self).__init__()

        self.features = DeepClusterNet5gTrunk(sobel, input_ch, input_sp_sz)

        self.classifier = nn.Sequential(
            nn.Linear(512 * BasicBlock.expansion, 4096),
            nn.BatchNorm2d(4096),
            nn.ReLU(True)
        )

        self.top_layer = nn.Linear(4096, out)

        self._initialize_weights()

        if sobel:
            self.features.sobel = make_sobel()

    def forward(self, x, penultimate=False):
        x = self.features(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        # used by assess code
        if penultimate:
            return x

        if self.top_layer:
            x = self.top_layer(x)
        return x

def deepcluster_net5g(sobel=False, out=None, input_sp_sz=None, input_ch=None):
    return DeepClusterNet5g(sobel, out, input_sp_sz, input_ch)