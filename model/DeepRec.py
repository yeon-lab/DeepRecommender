import torch
import torch.nn as nn
from torch.nn import functional as F
from model.base import CrossConvolution, Emb, _Net

class Rec_Model(nn.Module):
    def __init__(self, params):
        super(Rec_Model, self).__init__() 
        
        self.user_latent = len(user_count)*params['emb']
        self.item_latent = (len(item_count)+2)*params['emb']
        
        self.UserEmb = Emb(params['user_count'], params['emb'])
        self.ItemEmb = Emb(params['item_count'], params['emb'])
        self.GenEmb  = _Net(params['genr_nfeat'], params['emb'])
        self.TxtEmb  = _Net(params['txt_nfeat'], params['emb'])
        
        layers = [CrossConvolution(1, params['nhid'][0])]
            
        for i in range(len(params['nhid']) - 1):
            layers.append(CrossConvolution(params['nhid'][i], params['nhid'][i + 1]))
            
        if params['pool'] == 'avg':
            layers.append(nn.AvgPool2d(self.user_latent,self.item_latent))
        else:
            layers.append(nn.MaxPool2d(self.user_latent,self.item_latent))

        self.conv = nn.Sequential(*layers)
        
        self.dense = nn.Sequential(
            nn.Linear(params['nhid'][-1],1), nn.ReLU()
        ) 
    
    def forward(self, user, cat, genr, txt):     
        user_out = self.UserEmb.forward(user)        
        cat_out = self.ItemEmb.forward(cat)
        gen_out = self.GenEmb.forward(genr)
        txt_out = self.TxtEmb.forward(txt)
        item_out = torch.cat((cat_out,gen_out,txt_out),1)
        
        out = torch.unsqueeze(torch.bmm(user_out.unsqueeze(2), item_out.unsqueeze(1)),1)
        out = self.conv(out)
        out = out.view(out.size(0), -1)
        out = self.dense(out)
        
        return out    
