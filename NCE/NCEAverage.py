import torch
from torch import nn
from .alias_multinomial import AliasMethod
import math


class NCEAverage(nn.Module):

    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, use_softmax=False):
        super(NCEAverage, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([K, T, -1, -1, -1, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_l', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_ab', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_ori', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_comp', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, l, ab, ori, comp, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_l = self.params[2].item()
        Z_ab = self.params[3].item()
        Z_ori = self.params[4].item()
        Z_comp = self.params[5].item()

        momentum = self.params[6].item()
        batchSize = l.size(0)
        outputSize = self.memory_l.size(0)
        inputSize = self.memory_l.size(1)

        # score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)
        # sample
        weight_l = torch.index_select(self.memory_l, 0, idx.view(-1)).detach()
        weight_l = weight_l.view(batchSize, K + 1, inputSize)
        out_ab_l = torch.bmm(weight_l, ab.view(batchSize, inputSize, 1))
        # sample
        weight_ori = torch.index_select(self.memory_ori, 0, idx.view(-1)).detach()
        weight_ori = weight_ori.view(batchSize, K + 1, inputSize)
        out_l_ori = torch.bmm(weight_ori, l.view(batchSize, inputSize, 1))
        # sample
        weight_ab = torch.index_select(self.memory_ab, 0, idx.view(-1)).detach()
        weight_ab = weight_ab.view(batchSize, K + 1, inputSize)
        out_ori_ab = torch.bmm(weight_ab, ori.view(batchSize, inputSize, 1))

        # sample
        weight_comp = torch.index_select(self.memory_comp, 0, idx.view(-1)).detach()
        weight_comp = weight_comp.view(batchSize, K + 1, inputSize)
        out_ab_comp = torch.bmm(weight_comp, ab.view(batchSize, inputSize, 1))
        # sample
        weight_comp = torch.index_select(self.memory_comp, 0, idx.view(-1)).detach()
        weight_comp = weight_comp.view(batchSize, K + 1, inputSize)
        out_l_comp = torch.bmm(weight_comp, l.view(batchSize, inputSize, 1))
        # sample
        weight_ori = torch.index_select(self.memory_ori, 0, idx.view(-1)).detach()
        weight_ori = weight_ori.view(batchSize, K + 1, inputSize)
        out_comp_ori = torch.bmm(weight_ori, comp.view(batchSize, inputSize, 1))

        if self.use_softmax:
            out_ab_l = torch.div(out_ab_l, T)
            out_ab_l = out_ab_l.contiguous()
            out_l_ori = torch.div(out_l_ori, T)
            out_l_ori = out_l_ori.contiguous()
            out_ori_ab = torch.div(out_ori_ab, T)
            out_ori_ab = out_ori_ab.contiguous()
            out_ab_comp = torch.div(out_ab_comp, T)
            out_ab_comp = out_ab_comp.contiguous()
            out_l_comp = torch.div(out_l_comp, T)
            out_l_comp = out_l_comp.contiguous()
            out_comp_ori = torch.div(out_comp_ori, T)
            out_comp_ori = out_comp_ori.contiguous()
        else:
            out_ab_l = torch.exp(torch.div(out_ab_l, T))
            out_l_ori = torch.exp(torch.div(out_l_ori, T))
            out_ori_ab = torch.exp(torch.div(out_ori_ab, T))
            out_ab_comp = torch.exp(torch.div(out_ab_comp, T))
            out_l_comp = torch.exp(torch.div(out_l_comp, T))
            out_comp_ori = torch.exp(torch.div(out_comp_ori, T))
            # set Z_0 if haven't been set yet,
            # Z_0 is used as a constant approximation of Z, to scale the probs
            if Z_l < 0:
                self.params[2] = out_l_ori.mean() * outputSize
                Z_l = self.params[2].clone().detach().item()
                print("normalization constant Z_l is set to {:.1f}".format(Z_l))
            if Z_ab < 0:
                self.params[3] = out_ab_l.mean() * outputSize
                Z_ab = self.params[3].clone().detach().item()
                print("normalization constant Z_ab is set to {:.1f}".format(Z_ab))
            if Z_ori < 0:
                self.params[4] = out_ori_ab.mean() * outputSize
                Z_ori = self.params[4].clone().detach().item()
                print("normalization constant Z_ori is set to {:.1f}".format(Z_ori))
            if Z_comp < 0:
                self.params[5] = out_comp_ori.mean() * outputSize
                Z_comp = self.params[5].clone().detach().item()
                print("normalization constant Z_comp is set to {:.1f}".format(Z_comp))
            # compute out
            out_ab_l = torch.div(out_ab_l, Z_ab).contiguous()
            out_l_ori = torch.div(out_l_ori, Z_l).contiguous()
            out_ori_ab = torch.div(out_ori_ab, Z_ori).contiguous()
            out_ab_comp = torch.div(out_ab_comp, Z_ab).contiguous()
            out_l_comp = torch.div(out_l_comp, Z_l).contiguous()
            out_comp_ori = torch.div(out_comp_ori, Z_comp).contiguous()

        # # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_l, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(l, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_l = l_pos.div(l_norm)
            self.memory_l.index_copy_(0, y, updated_l)

            ab_pos = torch.index_select(self.memory_ab, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(ab, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_ab = ab_pos.div(ab_norm)
            self.memory_ab.index_copy_(0, y, updated_ab)

            ori_pos = torch.index_select(self.memory_ori, 0, y.view(-1))
            ori_pos.mul_(momentum)
            ori_pos.add_(torch.mul(ori, 1 - momentum))
            ori_norm = ori_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_ori = ori_pos.div(ori_norm)
            self.memory_ori.index_copy_(0, y, updated_ori)

            comp_pos = torch.index_select(self.memory_comp, 0, y.view(-1))
            comp_pos.mul_(momentum)
            comp_pos.add_(torch.mul(comp, 1 - momentum))
            comp_norm = comp_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_comp = comp_pos.div(comp_norm)
            self.memory_comp.index_copy_(0, y, updated_comp)

        return out_l_ori, out_ab_l, out_ori_ab, out_ab_comp, out_l_comp, out_comp_ori
