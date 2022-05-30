from __future__ import print_function

import torch
import torch.nn as nn


class AlexNet(nn.Module):  # GH：inherit nn.Module
    def __init__(self, feat_dim=128):
        super(AlexNet, self).__init__()
        self.encoder = alexnet(feat_dim=feat_dim)
        self.encoder = nn.DataParallel(self.encoder)

    def forward(self, x, layer=8):
        return self.encoder(x, layer)


class alexnet(nn.Module):
    def __init__(self, feat_dim=128, high_feat_dim=128):
        super(alexnet, self).__init__()

        self.l_to_ab = alexnet_half(in_channel=1, feat_dim=feat_dim, high_feat_dim=high_feat_dim)
        self.ab_to_l = alexnet_half(in_channel=2, feat_dim=feat_dim, high_feat_dim=high_feat_dim)
        self.ori = alexnet_half(in_channel=3, feat_dim=feat_dim, high_feat_dim=high_feat_dim)
        self.comp_net = complementarity_factor_net(feat_dim=feat_dim, high_feat_dim=high_feat_dim)
        a=1
        pass


    def forward(self, x, layer=8): # GH:layer=8 means to use all 8 layers
        l, ab = torch.split(x, [1, 2], dim=1)
        high_feat_l, feat_l = self.l_to_ab(l, layer)
        high_feat_ab, feat_ab = self.ab_to_l(ab, layer)
        high_feat_ori, feat_ori = self.ori(x, layer)
        high_feat_comp = self.comp_net(high_feat_l, feat_l, high_feat_ab, feat_ab, high_feat_ori, feat_ori)

        # print("gh alexnet 34 :feat_ori is", feat_ori)

        return high_feat_l, high_feat_ab, high_feat_ori, high_feat_comp


class complementarity_factor_net(nn.Module):
    def __init__(self, feat_dim=128, high_feat_dim=128):
        super(complementarity_factor_net, self).__init__()
        self.fc6 = nn.Sequential(
            nn.Linear(((3 * feat_dim) + (3 * high_feat_dim)) * 6 * 6, 4096 // 2),
            nn.BatchNorm1d(4096 // 2),
            nn.ReLU(inplace=True),
        )
        self.fc7 = nn.Sequential(
            nn.Linear(4096 // 2, 4096 // 2),
            nn.BatchNorm1d(4096 // 2),
            nn.ReLU(inplace=True),
        )
        self.fc8 = nn.Sequential(
            nn.Linear(4096 // 2, high_feat_dim)
        )
        self.l2norm = Normalize(2)

    def forward(self, high_feat_l, feat_l, high_feat_ab, feat_ab, high_feat_ori, feat_ori):
        x = torch.cat([feat_l, feat_ori, feat_ab], 1)
        high_x = torch.cat([high_feat_l, high_feat_ori, high_feat_ab], 1)
        high_x = high_x.unsqueeze(-1).unsqueeze(-1).expand(-1, 384, 6, 6)
        x = torch.cat([x, high_x], 1)
        x = x.view(x.shape[0], -1)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        x = self.l2norm(x)
        return x


class alexnet_half(nn.Module):
    def __init__(self, in_channel=1, feat_dim=128, high_feat_dim=128):
        super(alexnet_half, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channel, 96 // 2, 11, 4, 2, bias=False),
            nn.BatchNorm2d(96 // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(96 // 2, 256 // 2, 5, 1, 2, bias=False),
            nn.BatchNorm2d(256 // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(256 // 2, 384 // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(384 // 2),
            nn.ReLU(inplace=True),
        )
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(384 // 2, 384 // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(384 // 2),
            nn.ReLU(inplace=True),
        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(384 // 2, feat_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.fc6 = nn.Sequential(
            nn.Linear(256 * 6 * 6 // 2, 4096 // 2),
            nn.BatchNorm1d(4096 // 2),
            nn.ReLU(inplace=True),
        )
        self.fc7 = nn.Sequential(
            nn.Linear(4096 // 2, 4096 // 2),
            nn.BatchNorm1d(4096 // 2),
            nn.ReLU(inplace=True),
        )
        self.fc8 = nn.Sequential(
            nn.Linear(4096 // 2, high_feat_dim)
        )
        self.l2norm = Normalize(2)

    def forward(self, x, layer):
        if layer <= 0:
            return x
        x = self.conv_block_1(x)
        if layer == 1:
            return x
        x = self.conv_block_2(x)
        if layer == 2:
            return x
        x = self.conv_block_3(x)
        if layer == 3:
            return x
        x = self.conv_block_4(x)
        if layer == 4:
            return x
        x = self.conv_block_5(x)
        if layer == 5:
            return x
        low_x = x
        x = x.view(x.shape[0], -1)
        x = self.fc6(x)
        if layer == 6:
            return x
        x = self.fc7(x)
        if layer == 7:
            return x
        x = self.fc8(x)
        x = self.l2norm(x)
        return x, low_x


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


if __name__ == '__main__':

    import torch

    model = alexnet().cuda()
    data = torch.rand(10, 3, 224, 224).cuda()
    out = model.compute_feat(data, 5)

    for i in range(10):
        out = model.compute_feat(data, i)
        print(i, out.shape)

