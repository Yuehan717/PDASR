import os
import math
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import imageio

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch.nn as nn
from common.wavelet import DWT_Haar

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        self.model = args.model.lower()
        self.split_loss = args.split_loss
        
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S') # get current time
        
        os.makedirs(os.path.join('experiment',self.model),exist_ok=True)
        if not args.load:
            if not args.save:
                args.save = now # if user gives name, use it. Otherwise use timestamp instead.
            self.dir = os.path.join('..', 'experiment',self.model ,args.save)
        else: # if load != '', user give names and load model
            self.dir = os.path.join('..', 'experiment',self.model ,args.load)
            if os.path.exists(self.dir):
                self.log = torch.load(self.get_path('psnr_log.pt'))
                print('Continue from epoch {}...'.format(len(self.log)))
            else:
                args.load = ''
    # What if neither of load and save is ''
        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        for d in args.data_test:
            os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt'))else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.n_processes = 8

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self, trainer, epoch, is_bestP=False, is_bestD=False, inter=False):
        trainer.model.save(self.get_path('model'), epoch, is_bestP=is_bestP, is_bestD=is_bestD,inter=inter)
        if self.split_loss:
            trainer.loss_d.save(self.dir)
            trainer.loss_d.plot_loss(self.dir, epoch)
            trainer.loss_p.save(self.dir)
            trainer.loss_p.plot_loss(self.dir, epoch)
        else:   
            trainer.loss.save(self.dir)
            trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        if self.split_loss:
            trainer.optimizer_d.save(self.dir)
            trainer.optimizer_p.save(self.dir)
        else:
            trainer.optimizer.save(self.dir)
        torch.save(self.log, self.get_path('psnr_log.pt'))

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False): # log what also print on terminal
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        for idx_data, d in enumerate(self.args.data_test):
            label = 'SR on {}'.format(d)
            fig = plt.figure()
            plt.title(label)
            for idx_scale, scale in enumerate(self.args.scale):
                plt.plot(
                    axis,
                    self.log[:, idx_data, idx_scale].numpy(),
                    label='Scale {}'.format(scale)
                )
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('PSNR')
            plt.grid(True)
            plt.savefig(self.get_path('test_{}.pdf'.format(d)))
            plt.close(fig)

    def begin_background(self): # multi-process, save images background
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None: break
                    imageio.imwrite(filename, tensor.numpy())
        
        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes) # for each processor, establish a process that execute bg_target
        ]
        
        for p in self.process: p.start()

    def end_background(self):
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()

    def save_results(self, dataset, filename, save_list, scale):
        if self.args.save_results:
            filename = self.get_path(
                'results-{}'.format(dataset.dataset.name),
                '{}_'.format(filename)
            )
            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):
                # print("Size of v {}".format(v.size()))
                normalized = v[0].mul(255 / self.args.rgb_range) #normalize ???
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu() # self.byte() == self.to(torch.uint8)
                self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu)) # use subprocess to save images

def ts_YCrCb2RGB(img, rgb_range):
    # Convert a batch of YCrCb tensor images to RGB color space
    # The formulation refers to Opencv
    # input (B,C,H,W)
    # print(img.dtype)
    if rgb_range==255:
        delta = 128
    elif rgb_range==1:
        delta = 0.5
        
    Y, Cr, Cb = img[:,0,:,:], img[:,1,:,:], img[:,2,:,:]
    R = Y + 1.403*(Cr-delta)
    G = Y - 0.714*(Cr-delta) - 0.344*(Cb-delta)
    B = Y + 1.773*(Cb-delta)
    out = torch.stack((R,G,B),dim=1)
    return out

def ts_RGB2YCrCb(img, rgb_range):
    if rgb_range==255:
        delta = 128
    elif rgb_range==1:
        delta = 0.5
        
    R, G, B = img[:,0,:,:], img[:,1,:,:,1], img[:,2,:,:]
    Y = 0.299*R + 0.587*G + 0.114*B
    Cr = (R - Y)*0.713 + delta
    Cb = (B - Y)*0.564 + delta
    out = torch.stack((Y,Cr,Cb),dim=1)
    return out


def quantize(img, rgb_range, convert=False):
    if convert:
        # print('convert')
        img = ts_YCrCb2RGB(img, rgb_range)
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range) # confused, why do this? get integer value?

def calc_psnr(sr, hr, scale, rgb_range, dataset=None, Y=False):
    if Y:
        return calc_psnrY(sr, hr, scale, rgb_range, dataset)
    else:
        return calc_psnrG(sr, hr, scale, rgb_range, dataset)
    
def calc_psnrG(sr, hr, scale, rgb_range, dataset=None):
    if hr.nelement() == 1: return 0

    diff = (sr - hr) / rgb_range # step 1
    # print("size of diff {}".format(diff.size()))
    if dataset and dataset.dataset.benchmark:
        shave = scale
        if diff.size(1) > 1: #size(0): batch size # not grayscale
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            # print("tensor conevert {}".format(convert))
            diff = diff.mul(convert).sum(dim=1)
    else:
        # print("HERE")
        shave = scale + 6

    valid = diff[..., shave:-shave, shave:-shave]  # shave border, is this psnr calculate procedure official? It's the implementation in Matlab
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse) # already div 255 in step 1

### (13 Sep) Calculate PSNR values on Y channel. 
def calc_psnrY(sr, hr, scale, pixel_range, datset=None):
    if sr.size(1) != 1 :
        sr = ts_RGB2YCrCb(sr)[:,:1,:,:]
    if hr.size(1) != 1:
        hr = ts_RGB2YCrCb(hr)[:,:1,:,:]
    diff = (sr - hr)
    if datset and datset.dataset.benchmark:
        shave = scale
    else:
        shave = scale + 6
    valid = diff[...,shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()/(pixel_range**2)
    
    return 10 * math.log10((pixel_range ** 2)/mse)
    

def make_optimizer(args, target, type_d = True):
    '''
        make optimizer and scheduler together
    '''
    # optimizer
    # parameters() returns iterator over module parameters, is usually passed to optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters()) # filt out params that do not have grad.
    type = None
    if args.split_loss:
        lr = args.lr_d if type_d else args.lr_p
        type = 'd_' if type_d else 'p_'
    else:
        lr = args.lr
        type = ''
    kwargs_optimizer = {'lr': lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    # scheduler
    milestones = list(map(lambda x: int(x), args.decay.split('-')))
    # milestones = list([args.decay * x for x in range(1,min(args.epochs,1000)//args.decay)]) #split the params and convert it to integer
    kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
    scheduler_class = lrs.MultiStepLR # used for learning rate decay

    class CustomOptimizer(optimizer_class): ## derive from certain loss module
        def __init__(self,type,*args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs) # init the loss module
            self.type = type

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()

        def get_dir(self, dir_path):
            return os.path.join(dir_path, self.type+'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_last_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch
    
    optimizer = CustomOptimizer(type, trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer

class GaussianLayer(nn.Module):
    def __init__(self, size=21, p=3, n_color=3):
        super(GaussianLayer, self).__init__()
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(10), 
            nn.Conv2d(n_color, n_color, size, stride=1, padding=0, bias=None, groups=3)
        )
        self.size = size
        self.sigma = p
        self.weights_init()
        
    
    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        n = np.zeros((self.size, self.size))
        n[self.size//2, self.size//2] = 1
        k = scipy.ndimage.gaussian_filter(n,sigma=self.sigma)
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))
            f.requires_grad = False

class FourierLayer(nn.Module):
    def __init__(self, size, p=0.3):
        super(FourierLayer, self).__init__()
        self.mask = torch.zeros((size,size))
        start = size // 2 * p
        end = size //2 * (1+p)
        self.mask[:,:,start:end, start, end] = 1
    def forward(self, x):
        fft2 = torch.fft.fft2(x)
        fft2 = fft2 * self.mask
        return fft2

class conADMM(nn.Module):
    def __init__(self, n_colors, size, p, rho=0.01,filter='wvt', mode="mean"):
        super(conADMM,self).__init__()
        self.rho = rho
        self.n_colors = n_colors
        self.mode = mode
        # self.is_LL = is_LL
        self.filter = filter
        if filter == 'wvt':
            self.Tlf = DWT_Haar()
        elif filter == 'gaussian':
            self.Tlf = GaussianLayer(size, p, n_colors)
        elif filter == 'fourier':
            self.Tlf = FourierLayer(size, p)
    
    def forward(self, sr, y, S=None):
        # print(sr.size(),y.size())
        ll = self.Tlf(sr)
        LL = self.Tlf(y)
        # print(ll.size(),LL.size())
        if self.filter == 'wvt':
            ll = ll[:,:self.n_colors,...]
            LL = LL[:,:self.n_colors,...]
        if S == None:
            update = ll-LL 
        else:
            # print(ll.size(),LL.size(),S.size())
            diff = ll - LL + S
            # print(torch.mean(abs(S)),torch.mean(abs(ll-LL)))
            if self.mode == "mean":
                update = (self.rho/2) * torch.mean(diff * diff)
            elif self.mode == "sqrt":
                update = (self.rho/2) * torch.sqrt(torch.sum(diff*diff))
        return update

def pad(lr, pad, scale):
    times = (2**pad) / scale
    if times <= 1:
        return lr, 0, 0
    padding_h, padding_w = 0,0
    if lr.size(2) % times !=0:
        padding_h = times - lr.size(2) % times
    if lr.size(3) % times !=0:
        padding_w = times - lr.size(3) % times
    lr_pad = torch.zeros((lr.size(0),lr.size(1),
                            lr.size(2)+padding_h,lr.size(3)+padding_w))
    lr_pad[:,:,:lr.size(2),:lr.size(3)] = lr
    lr_pad[:,:,lr.size(2):,lr.size(3):] = lr[:,:,lr.size(2)-padding_h:,lr.size(3)-padding_w:]
    return lr_pad, padding_h, padding_w

def chop(sr,padding_h, padding_w, scale = 4):
    sr = sr[:,:,:sr.size(2)-padding_h*scale,:sr.size(3)-padding_w*scale]
    
    return sr