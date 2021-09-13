import torch
import torch.nn as nn

class DR(nn.Module):
    def __init__(self, in_d, out_d):
        super(DR, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.conv1 = nn.Conv2d(self.in_d, self.out_d, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_d)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio, kernel_size):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class DS_layer(nn.Module):
    def __init__(self, in_d, out_d, stride, output_padding, n_class):
        super(DS_layer, self).__init__()

        self.dsconv = nn.ConvTranspose2d(in_d, out_d, kernel_size=3, padding=1, stride=stride,
                                         output_padding=output_padding)
        self.bn = nn.BatchNorm2d(out_d)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=0.2)
        self.outconv = nn.ConvTranspose2d(out_d, n_class, kernel_size=3, padding=1)

    def forward(self, input):
        x = self.dsconv(input)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.outconv(x)
        return x