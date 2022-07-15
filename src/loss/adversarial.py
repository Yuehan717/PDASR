import common.utility as utility
from types import SimpleNamespace

from model import common
from loss import discriminator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from common.wavelet import DWT_Haar, IWT_Haar
import random

class Adversarial(nn.Module):
    def __init__(self, args, gan_type):
        super(Adversarial, self).__init__()
        self.gan_type = gan_type
        self.gan_k = args.gan_k
        self.count = 0
        self.dis = discriminator.Discriminator(args)
        if gan_type == 'WGAN_GP':
            # see https://arxiv.org/pdf/1704.00028.pdf pp.4
            optim_dict = {
                'optimizer': 'ADAM',
                'betas': (0, 0.9),
                'epsilon': 1e-8,
                'lr': 1e-5,
                'weight_decay': args.weight_decay,
                'decay': args.decay,
                'gamma': args.gamma,
                'split_loss' : False,
                'epochs': args.epochs
            }
            optim_args = SimpleNamespace(**optim_dict)
        elif gan_type == 'WGAN':
            optim_dict = {
                'optimizer': 'RMSprop',
                'alpha': 0.99,
                'epsilon': 1e-8,
                'lr': 5e-5,
                'weight_decay': args.weight_decay,
                'decay': args.decay,
                'gamma': args.gamma,
                'split_loss' : False,
                'epochs': args.epochs
            }
            optim_args = SimpleNamespace(**optim_dict)
        else:
            optim_args = args

        self.optimizer = utility.make_optimizer(optim_args, self.dis)
        self.criterion_rgan = nn.BCEWithLogitsLoss().cuda()
        self.soft = args.soft_label
        self.n_colors = args.n_colors
        
    def forward(self, fake, real):
        # updating discriminator...
        self.loss = 0
        fake_detach = fake.detach()     # do not backpropagate through G
        for _ in range(self.gan_k):
        # if self.count % self.gan_k == 0:
            self.optimizer.zero_grad()
            # d: B x 1 tensor
            d_fake = self.dis(fake_detach)
            # d_fake_detach = d_fake.detach()
            d_real = self.dis(real)
            retain_graph = False
            if self.gan_type == 'GAN':
                loss_d = self.bce(d_real, d_fake)
            elif self.gan_type.find('WGAN') >= 0:
                
                loss_d = d_fake.mean() - d_real.mean()
                
                if self.gan_type.find('GP') >= 0:
                    retain_graph = True
                    gradient_penalty = self.compute_gradient_penalty(real, fake)
                    loss_d += gradient_penalty
            # from ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks
            elif self.gan_type == 'RGAN':
                # d_real = d_real.detach()
                better_real = d_real - d_fake.mean(dim=0, keepdim=True)
                better_fake = d_fake - d_real.mean(dim=0, keepdim=True)
                loss_d = self.bce(better_real, better_fake)
                retain_graph = True

            # Discriminator update
            self.loss += loss_d.item()
            # if self.gan_type.find('WGAN') == -1:
            loss_d.backward(retain_graph=retain_graph)
            if self.count == 0:
                self.optimizer.step()
            
            self.count = (self.count + 1) % self.gan_k

            if self.gan_type == 'WGAN':
                for p in self.dis.parameters():
                    p.data.clamp_(-0.01, 0.01)

        self.loss /= self.gan_k

        # updating generator...
        d_fake_bp = self.dis(fake)      # for backpropagation, use fake as it is
        if self.gan_type == 'GAN':
            label_real = torch.ones_like(d_fake_bp)
            loss_g = F.binary_cross_entropy_with_logits(d_fake_bp, label_real)
        elif self.gan_type.find('WGAN') >= 0:
            loss_g = -d_fake_bp.mean()
        elif self.gan_type == 'RGAN':
            d_real = d_real.detach()
            better_real = d_real - d_fake_bp.mean(dim=0, keepdim=True)
            better_fake = d_fake_bp - d_real.mean(dim=0, keepdim=True)
            loss_g = self.bce(better_fake, better_real)

        # Generator loss
        return loss_g
    
    def _RGAN_forward(self, fake ,real):
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        self.optimizer.zero_grad()
        
        valid = Variable(Tensor(np.ones((fake.size(0), *self.dis.output_shape))), requires_grad=False)
        invalid = Variable(Tensor(np.zeros((fake.size(0), *self.dis.output_shape))), requires_grad=False)
        
        fake_d_pred = self.dis(fake).detach()
        real_d_pred = self.dis(real)
        l_d_real = self.criterion_rgan(real_d_pred - torch.mean(fake_d_pred), valid)
        l_d_real.backward()
        
        fake_d_pred = self.dis(fake.detach())
        l_d_fake = self.criterion_rgan(fake_d_pred - torch.mean(real_d_pred.detach()), invalid)
        l_d_fake.backward()
        
        self.optimizer.step()
        
        # updating generator...
        d_fake_bp = self.dis(fake)      # for backpropagation, use fake as it is
        d_real_bp = self.dis(real).detach()
        better_real = d_real_bp - d_fake_bp.mean(dim=0, keepdim=True)
        better_fake = d_fake_bp - d_real_bp.mean(dim=0, keepdim=True)
        loss_g = self.bce(better_fake, better_real)

        # Generator loss
        return loss_g
    
    def state_dict(self, *args, **kwargs):
        state_discriminator = self.dis.state_dict(*args, **kwargs)
        state_optimizer = self.optimizer.state_dict()

        return dict(**state_discriminator, **state_optimizer)

    def bce(self, real, fake):
        label_real = torch.ones_like(real)
        label_fake = torch.zeros_like(fake)
        if self.soft:
            # label_real*=random.uniform(0.8,1)
            # label_fake+=random.uniform(0,0.2)
            label_real*=0.9
        bce_real = F.binary_cross_entropy_with_logits(real, label_real)
        bce_fake = F.binary_cross_entropy_with_logits(fake, label_fake)
        bce_loss = bce_real + bce_fake
        return bce_loss

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand(real_samples.size(0), 1, 1, 1).cuda()
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.dis(interpolates)
        # fake = Variable(torch.rand(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = torch.ones(d_interpolates.size()).cuda()
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
# Some references
# https://github.com/kuc2477/pytorch-wgan-gp/blob/master/model.py
# OR
# https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
