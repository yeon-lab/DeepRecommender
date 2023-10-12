import torch
import torch.nn as nn
import numpy as np

class _Net(nn.Module):
    def __init__(self, input_size, hidden_unit_sizes):
        super(_Net,self).__init__()
        
        self.layers = nn.Sequential(
                          nn.Linear(input_size, hidden_unit_sizes),
                          nn.BatchNorm1d(hidden_unit_sizes),
                          nn.LeakyReLU(0.33))
        
    def forward(self,x):
        out = self.layers(x)
        return out

class Emb(nn.Module):
    def __init__(self, count, emb):
        super(Emb,self).__init__()
        layers = []     
        for n in for(count):
            EE = nn.Embedding(n, emb, sparse=True)
            W = np.random.uniform(
                low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, emb)).astype(np.float32)
            EE.weight.data = torch.tensor(W, requires_grad=True)
            layers.append(EE)         
            
        self.emb_layer = nn.ModuleList(layers) 
    
    def forward(self,data):   
        for i, layer in enumerate(self.emb_layer):            
            if i == 0:
                out = layer(data[:,i])
            else :
                out = torch.cat((out,layer(data[:,i])),1)
        return out

    
class CrossConvolution(torch.nn.Module):
    def __init__(self, in_channels, out_channels, user_latent,item_latent,bias=False):
        super(CrossConvolution, self).__init__()
        self.item_latent = item_latent
        self.user_latent = user_latent
        self.CNN_row = nn.Conv2d(in_channels,out_channels,(1,self.item_latent),bias=bias)
        self.CNN_col = nn.Conv2d(in_channels,out_channels,(self.user_latent,1),bias=bias)

    def forward(self, x):
        row_calc = self.CNN_row(x)
        col_calc = self.CNN_col(x)
        return torch.cat([row_calc]*self.item_latent, 3) + torch.cat([col_calc]*self.user_latent, 2)
    
    
