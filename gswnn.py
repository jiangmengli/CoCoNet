import numpy as np
import torch
from torch import nn
from torch import optim
from util import set_requires_grad

class GSW_NN():
    def __init__(self, device, train_lr, din=2, nofprojections=10, model_depth=3,
                 num_filters=32, train_wd=0.0005):
        self.nofprojections=nofprojections
        self.device=device
        self.train_lr = train_lr
        self.train_wd = train_wd
        
        self.parameters=None # This is for max-GSW
        self.din=din
        self.dout=nofprojections
        self.model_depth=model_depth
        self.num_filters=num_filters
        self.model=MLP(din=self.din, dout=self.dout, num_filters=self.num_filters, device=device).to(device)
 
    def gsw(self,X,Y,Z,random=True):
        N,dn = X.shape
        M,dm = Y.shape
        P,dp = Z.shape
        assert dn==dm and M==N
        assert dn==dp and P==N
        assert dm==dp and M==P
        
        if random:
            self.model.reset()
        
        Xslices=self.model(X.to(self.device))
        Yslices=self.model(Y.to(self.device))
        Zslices=self.model(Z.to(self.device))

        Xslices_sorted=torch.sort(Xslices,dim=0)[0]
        Yslices_sorted=torch.sort(Yslices,dim=0)[0]
        Zslices_sorted=torch.sort(Zslices,dim=0)[0]

        return_value = torch.sqrt(torch.sum((Xslices_sorted-Yslices_sorted)**2)) + torch.sqrt(torch.sum((Xslices_sorted-Zslices_sorted)**2)) + torch.sqrt(torch.sum((Zslices_sorted-Yslices_sorted)**2))
        return return_value

    def max_gsw(self,X,Y,Z,iterations=50):
        set_requires_grad(self.model, requires_grad=True)
        N,dn = X.shape
        M,dm = Y.shape
        P,dp = Z.shape
        assert dn==dm and M==N
        assert dn==dp and P==N
        assert dm==dp and M==P

        self.model.reset()
        
        optimizer=optim.Adam(self.model.parameters(), lr=self.train_lr, weight_decay=self.train_wd)
        total_loss=np.zeros((iterations,))
        for i in range(iterations):
            optimizer.zero_grad()
            loss=-self.gsw(X.to(self.device),Y.to(self.device),Z.to(self.device), random=False)
            total_loss[i]=loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()
        
        return_value = self.gsw(X.to(self.device),Y.to(self.device),Z.to(self.device), random=False)
        set_requires_grad(self.model, requires_grad=False)
        return return_value


class MLP(nn.Module):
    def __init__(self, din=2, dout=10, num_filters=32, depth=3, device='cuda'):
        super(MLP, self).__init__()
        self.device = device
        self.din = din
        self.dout = dout
        self.init_num_filters = num_filters
        self.depth = depth

        self.features = nn.Sequential()

        for i in range(self.depth):
            if i == 0:
                self.features.add_module('linear%02d' % (i + 1), nn.Linear(self.din, self.init_num_filters))
            else:
                self.features.add_module('linear%02d' % (i + 1),
                                         nn.Linear(self.init_num_filters, self.init_num_filters))
            self.features.add_module('activation%02d' % (i + 1), nn.LeakyReLU(inplace=True))

        self.features.add_module('linear%02d' % (i + 2), nn.Linear(self.init_num_filters, self.dout))

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32).requires_grad_(True).to(self.device)
        return self.features(x)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def reset(self):
        self.features.apply(self.init_weights)
