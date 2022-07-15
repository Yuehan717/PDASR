import torch
import torch.nn as nn
import model.common as common
from common.wavelet import DWT_Haar,IWT_Haar
def make_model(args):
    return CliqueNet(args)

class ResClique(nn.Module):
    """
    Option 3
    Res-Clique Block with 4 fixed layers.
    In_channel = 4 * g
    """
    def __init__(self, g=32, kernel_size=3, padding=None, bias=True, res_scale=1, n_stage=2):
        super(ResClique, self).__init__()
        self.g = g
        self.res_scale = res_scale
        self.n_stage = n_stage
        ### stage 1 ###
        self.Fll1 = nn.Sequential(nn.Conv2d(g,g,kernel_size,padding=(kernel_size//2), bias=bias),
                                  nn.PReLU(num_parameters=g))
        self.Flh1 = nn.Sequential(nn.Conv2d(2*g,g,kernel_size,padding=(kernel_size//2), bias=bias),
                                  nn.PReLU(num_parameters=g))
        self.Fhl1 = nn.Sequential(nn.Conv2d(2*g,g,kernel_size,padding=(kernel_size//2), bias=bias),
                                  nn.PReLU(num_parameters=g))
        self.Fhh1 = nn.Sequential(nn.Conv2d(4*g,g,kernel_size,padding=(kernel_size//2), bias=bias),
                                  nn.PReLU(num_parameters=g))
        if n_stage == 2:
        ### stage 2 ###
            self.Fll2 = nn.Sequential(nn.Conv2d(4*g,g,kernel_size,padding=(kernel_size//2), bias=bias),
                                    nn.PReLU(num_parameters=g))
            self.Flh2 = nn.Sequential(nn.Conv2d(2*g,g,kernel_size,padding=(kernel_size//2), bias=bias),
                                    nn.PReLU(num_parameters=g))
            self.Fhl2 = nn.Sequential(nn.Conv2d(2*g,g,kernel_size,padding=(kernel_size//2), bias=bias),
                                    nn.PReLU(num_parameters=g))
            self.Fhh2 = nn.Sequential(nn.Conv2d(g,g,kernel_size,padding=(kernel_size//2), bias=bias),
                                    nn.PReLU(num_parameters=g))
        
    def forward(self, x):
        ### stage 1 ###
        g = self.g
        ll0 = x[:,0:g,:,:]
        lh0 = x[:,g:2*g,:,:]
        hl0 = x[:,2*g:3*g,:,:]
        hh0 = x[:,3*g:4*g,:,:]
        ll1 = self.Fll1(ll0)
        lh1 = self.Flh1(torch.cat((ll1,lh0),dim=1))
        hl1 = self.Fhl1(torch.cat((ll1,hl0),dim=1))
        hh1 = self.Fhh1(torch.cat((ll1,lh1,hl1,hh0),dim=1))
        
        if self.n_stage == 2:
        ### stage 2 ###
            hh2 = self.Fhh2(hh1)
            hl2 = self.Fhl2(torch.cat((hl1,hh2),dim=1))
            lh2 = self.Flh2(torch.cat((lh1,hh2), dim=1))
            ll2 = self.Fll2(torch.cat((ll1,lh2,hl2,hh2), dim=1))
        
            ### Residual connection ###
            out = torch.cat((ll2,lh2,hl2,hh2), dim=1).mul(self.res_scale) + x
        else:
            out = torch.cat((ll1,lh1,hl1,hh1), dim=1).mul(self.res_scale) + x      
        return out
    
class CliqueNet(nn.Module):
    def __init__(self, args):
        super(CliqueNet, self).__init__()
        self.n_block = args.nBlocks
        self.n_feats = args.n_feats_p
        self.n_colors = args.n_colors
        self.scale = args.scale[0]
        self.res_scale = args.res_scale_p
        n_stage = args.n_stage
        self.dwt = DWT_Haar()
        self.preConv0 = nn.Conv2d(self.n_colors*4, self.n_feats, kernel_size=3,padding=1,bias=True)
        self.preConv1 = nn.Conv2d(self.n_feats, self.n_feats*4, kernel_size=3, padding=1, bias=True)
        self.CliqueBlock = nn.ModuleList()
        for _ in range(self.n_block):
            self.CliqueBlock.append(ResClique(g=self.n_feats, res_scale=self.res_scale,n_stage=n_stage))
            
        self.funsion = common.Conv2dWithActivation(self.n_feats*4*self.n_block, self.n_feats,
                                                   1,padding=0,activation=nn.LeakyReLU())
        # Old #
        self.tail=common.Conv2dWithActivation(self.n_feats,self.n_colors*4,3,padding=1,activation=None)
        self.iwt = IWT_Haar()
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        
    def forward(self, x):
        x = self.sub_mean(x)
        wvt = self.dwt(x)
        
        lrsb_1 = self.preConv0(wvt)
        lrsb = self.preConv1(lrsb_1)
        
        mid_feats = []
        
        for i in range(self.n_block):
            lrsb = self.CliqueBlock[i](lrsb)
            mid_feats.append(lrsb)
        
        lrsb = torch.cat(mid_feats, dim=1)
        
        srsb = self.funsion(lrsb) + lrsb_1
        srsb = self.tail(srsb)
        
        out = self.iwt(srsb)
        
        out = self.add_mean(out)
        
        return out
    
    