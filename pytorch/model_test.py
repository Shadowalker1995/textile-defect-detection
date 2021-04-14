import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential


class Net(nn.Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(3, 32, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(32, 64, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, 128, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, 256, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 256, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 128, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, 64, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64, 1000),)

    def forward(self, x):
        x = self.cnn_layers(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class CNN2(nn.Module):
    def __init__(self, num_classes=1, **kwargs):
        super(CNN2, self).__init__()
        self.conv_1 = BasicConv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv_2 = BasicConv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv_3 = BasicConv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv_4 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv_5 = BasicConv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(512, num_classes)

        # self.conv_1 = Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2))
        # self.conv_2 = Sequential(
        #     nn.Conv2d(32, 64, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2))
        # self.conv_3 = Sequential(
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2))
        # self.conv_4 = Sequential(
        #     nn.Conv2d(128, 256, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2))
        # self.conv_5 = Sequential(
        #     nn.Conv2d(256, 512, kernel_size=3, padding=1),
        #     nn.ReLU())
        # self.fc = Sequential(
        #     nn.Flatten(),
        #     nn.Linear(256*12*12, 256),
        #     nn.Dropout(0.75),
        #     nn.Linear(256, num_classes))

    def forward(self, x):
        # 3 x 200 x 200 -> 32 x 100 x 100
        x = self.conv_1(x)
        # 32 x 100 x 100 -> 64 x 50 x 50
        x = self.conv_2(x)
        # 64 x 50 x 50 -> 128 x 25 x 25
        x = self.conv_3(x)
        # 128 x 25 x 25 -> 256 x 13 x 13
        x = self.conv_4(x)
        # 256 x 13 x 13 -> 512 x 7 x 7
        x = self.conv_5(x)
        # 512 x 7 x 7 -> 512 x 1 x 1
        x = F.avg_pool2d(x, kernel_size=7)
        # 512 x 1 x 1 -> 512 x 1 x 1
        x = F.dropout(x, p=0.75)
        # 512 x 1 x 1 -> 512
        x = x.view(x.size()[0], -1)
        # 512 -> num_classes
        x = self.fc(x)

        return x


if __name__ == '__main__':
    net = CNN2(num_classes=8)
    print(net)
    img = torch.randn(2, 3, 200, 200)
    img = net.forward(img)
    print(img.shape)
