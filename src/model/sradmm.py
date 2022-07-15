from model import common
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from common.wavelet import DWT_Haar, IWT_Haar
from importlib import import_module
import os
def make_model(args):
    return SRADMM(args)
    
class SRADMM(nn.Module):
    """
    Final model
    Input: LR image
    Output: (list of distortion-oriented wavelet channels , list of final images).
    Problem: how to choose the initialization/pre-trained model.
    """
    def __init__(self, args, conv=common.default_conv):
        super(SRADMM, self).__init__()
        
        ### build stage 1 module ###
        module = import_module('model.' + args.DNet.lower())
        self.D = module.make_model(args)
        pretrained = args.traineds1
        if pretrained != '':
            kwargs = {}
            load_from = torch.load(pretrained, **kwargs)
            self.D.load_state_dict(load_from, strict=False)

        ### build refinement module ###
        module = import_module('model.' + args.PNet.lower())
        self.P = module.make_model(args)
            
        self.scale = float(args.scale[0])
        self.pre_train = not args.split_loss
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        if args.Tlf == "wvt":
            r = 2
        else:
            r = 1
        self.S = Variable(torch.full((args.batch_size, args.n_colors,
                                      args.patch_size//r,args.patch_size//r),0,dtype=torch.float32),
                          requires_grad=False).to(self.device)
    
    def forward(self,lr):
        
        sr0 = self.D(lr)
        if self.pre_train:
            sr1 = self.P(sr0)
        else:
            sr1 = self.P(sr0.detach())
        return (sr0, sr1)