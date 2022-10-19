from layers.layer import Parameter, Module
from layers.linear import Linear
from layers.activation import Sigmoid, ReLU, ELU
from layers.batchnorm import BatchNorm2D, BatchNormalization
from layers.conv import Conv

class FC_ResBlock(Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()

        self.fc1 = Linear(in_features, out_features, bias=bias)
        self.sigmoid = ELU()
        self.fc2 = Linear(out_features, out_features, bias=bias)
        self.fc3 = Linear(in_features, out_features, bias=bias)

        self.inner_layers = [self.fc1, self.fc2, self.fc3]

    def forward(self, X):
        out = self.fc1.forward(X)
        out = self.sigmoid.forward(out)
        out = self.fc2.forward(out)
        shortcut = self.fc3.forward(X)
        out = out + shortcut
        return out

    def backward(self, eta):
        grad1 = self.fc3.backward(eta)
        grad2 = self.fc2.backward(eta)
        grad2 = self.sigmoid.backward(grad2)
        grad2 = self.fc1.backward(grad2)
        return grad1 + grad2

    def cforward(self, X):
        out = self.fc1.cforward(X)
        out = self.sigmoid.cforward(out)
        out = self.fc2.cforward(out)
        shortcut = self.fc3.cforward(X)
        out = out + shortcut
        return out

    def cbackward(self, eta):
        grad1 = self.fc3.cbackward(eta)
        grad2 = self.fc2.cbackward(eta)
        grad2 = self.sigmoid.cbackward(grad2)
        grad2 = self.fc1.cbackward(grad2)
        return grad1 + grad2


class BasicBlock(Module):
    def __init__(self, inplane, outplanes, downsample=False, base_width=64, stride=2):
        super().__init__()
        self.conv1 = Conv((outplanes, 3, 3, inplane), 'SAME', stride=stride)
        self.bn1 = BatchNormalization(outplanes)
        self.relu1 = ELU()
        self.conv2 = Conv((outplanes, 3, 3, outplanes), 'SAME')
        self.bn2 = BatchNormalization(outplanes)
        self.relu2 = ELU()
        self.downsample = downsample
        if downsample:
            self.conv_downsample = Conv((outplanes, 1, 1, inplane),  stride=stride)
            self.inner_layers = [self.conv1, self.bn1, self.relu1, self.conv2, self.bn2,  self.conv_downsample, self.relu2]
        self.inner_layers = [self.conv1, self.bn1, self.relu1, self.conv2, self.bn2, self.relu2]

    def forward(self, X):
        identity = X
        # print(X.shape)
        # print(self.conv1.parameters[0].tensor.shape)
        out = self.conv1.forward(X)
        out = self.bn1.forward(out)
        out = self.relu1.forward(out)

        out = self.conv2.forward(out)
        out = self.bn2.forward(out)

        if self.downsample:
            out += self.conv_downsample.forward(identity)
        else:
            out += identity
        out = self.relu2.forward(out)
        return out

    def backward(self, eta):
        grad1 = self.relu2.backward(eta)
        if self.downsample:
            grad1_in = self.conv_downsample.backward(grad1)

        grad2 = self.bn2.backward(grad1)
        # grad2 = grad1
        grad2 = self.conv2.backward(grad2)
        grad2 = self.relu1.backward(grad2)
        grad2 = self.bn1.backward(grad2)
        grad2 = self.conv1.backward(grad2)
        return grad2 + grad1_in

    def cforward(self, X):
        identity = X

        out = self.conv1.cforward(X)
        out = self.bn1.cforward(out)
        out = self.relu1.cforward(out)

        out = self.conv2.cforward(out)
        out = self.bn2.cforward(out)

        if self.downsample:
            out += self.conv_downsample.cforward(identity)
        else:
            out += identity
        out = self.relu2.cforward(out)

        return out

    def cbackward(self, eta):
        grad1 = self.relu2.cbackward(eta) #* 0
        if self.downsample:
            grad1_in = self.conv_downsample.cbackward(grad1)

        grad2 = self.bn2.cbackward(grad1)
        # grad2 = grad1
        grad2 = self.conv2.cbackward(grad2)
        grad2 = self.relu1.cbackward(grad2)
        grad2 = self.bn1.cbackward(grad2)
        grad2 = self.conv1.cbackward(grad2)
        return grad2 + grad1_in

class _ResNet(Module):
    def __init__(self, num_layers_stack, in_channels, out_channels, stride=2):
        super().__init__()
        self.inner_layers = []
        self._make_layers(num_layers_stack, in_channels=in_channels, out_channels=out_channels, stride=stride)

    def _make_layers(self, num_layers_stack, in_channels, out_channels, stride):
        # downsample = False
        # if in_channels != out_channels:
        downsample = True
        self.inner_layers.append(BasicBlock(in_channels, out_channels, downsample=downsample, stride=stride))
        for _ in range(num_layers_stack - 1):
            self.inner_layers.append(BasicBlock(out_channels, out_channels, downsample=downsample, stride=1))

    def forward(self, X):
        out = X
        for layer in self.inner_layers:
            out = layer.forward(out)
        return out

    def backward(self, eta):
        grad = eta
        for layer in reversed(self.inner_layers):
            grad = layer.backward(grad)
            # print("===============resnet backward", grad.shape)
        return grad

    def cforward(self, X):
        out = X
        for layer in self.inner_layers:
            out = layer.cforward(out)
        return out

    def cbackward(self, eta):
        grad = eta
        for layer in reversed(self.inner_layers):
            grad = layer.cbackward(grad)
        return grad
