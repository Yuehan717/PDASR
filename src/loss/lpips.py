from . import networks_basic as networks
import torch
import os
# from .base_model import BaseModel
import torch.nn as nn

class DistModel(nn.Module):

    def __init__(self, model='net-lin', net='alex', pnet_rand=False, pnet_tune=False, model_path=None, colorspace='Lab', use_gpu=True, printNet=False, spatial=False, spatial_shape=None, spatial_order=1, spatial_factor=None,version='0.1'):
        super(DistModel, self).__init__()
        
        '''
        INPUTS
            model - ['net-lin'] for linearly calibrated network
                    ['net'] for off-the-shelf network
                    ['L2'] for L2 distance in Lab colorspace
                    ['SSIM'] for ssim in RGB colorspace
            net - ['squeeze','alex','vgg']
            model_path - if None, will look in weights/[NET_NAME].pth
            colorspace - ['Lab','RGB'] colorspace to use for L2 and SSIM
            use_gpu - bool - whether or not to use a GPU
            printNet - bool - whether or not to print network architecture out
            spatial - bool - whether to output an array containing varying distances across spatial dimensions
            spatial_shape - if given, output spatial shape. if None then spatial shape is determined automatically via spatial_factor (see below).
            spatial_factor - if given, specifies upsampling factor relative to the largest spatial extent of a convolutional layer. if None then resized to size of input images.
            spatial_order - spline order of filter for upsampling in spatial mode, by default 1 (bilinear).
            is_train - bool - [True] for training mode
            lr - float - initial learning rate
            beta1 - float - initial momentum term for adam
            version - 0.1 for latest, 0.0 was original
        '''

        self.model = model
        self.net = net
        self.use_gpu = use_gpu
        # self.is_train = is_train
        self.spatial = spatial
        self.spatial_shape = spatial_shape
        self.spatial_order = spatial_order
        self.spatial_factor = spatial_factor

        self.model_name = '%s [%s]'%(model,net)
        if(self.model == 'net-lin'): # pretrained net + linear layer
            self.net = networks.PNetLin(use_gpu=use_gpu,pnet_rand=pnet_rand, pnet_tune=pnet_tune, pnet_type=net,use_dropout=True,spatial=spatial,version=version)
            kw = {}
            if not use_gpu:
                kw['map_location'] = 'cpu'
            if(model_path is None):
                import inspect
                # model_path = '/weights/v%s/%s.pth'%(version,net)
                model_path = os.path.abspath(os.path.join(inspect.getfile(self.forward), '..', '..', 'weights/v%s/%s.pth'%(version,net)))

            
            print('Loading model from: %s'%model_path)
            self.net.load_state_dict(torch.load(model_path, **kw))

        elif(self.model=='net'): # pretrained network
            assert not self.spatial, 'spatial argument not supported yet for uncalibrated networks'
            self.net = networks.PNet(use_gpu=use_gpu,pnet_type=net)
            self.is_fake_net = True
        elif(self.model in ['L2','l2']):
            self.net = networks.L2(use_gpu=use_gpu,colorspace=colorspace) # not really a network, only for testing
            self.model_name = 'L2'
        elif(self.model in ['DSSIM','dssim','SSIM','ssim']):
            self.net = networks.DSSIM(use_gpu=use_gpu,colorspace=colorspace)
            self.model_name = 'SSIM'
        else:
            raise ValueError("Model [%s] not recognized." % self.model)

        self.parameters = list(self.net.parameters())

        # if self.is_train: # training mode
        #     # extra network on top to go from distances (d0,d1) => predicted human judgment (h*)
        #     self.rankLoss = networks.BCERankingLoss(use_gpu=use_gpu)
        #     self.parameters+=self.rankLoss.parameters
        #     self.lr = lr
        #     self.old_lr = lr
        #     self.optimizer_net = torch.optim.Adam(self.parameters, lr=lr, betas=(beta1, 0.999))
        # else: # test mode
        self.net.eval()

        if(printNet):
            print('---------- Networks initialized -------------')
            networks.print_network(self.net)
            print('-----------------------------------------------')

    def forward(self,in1,in2,retPerLayer=False):
        if(retPerLayer):
            return self.net.forward(in1,in2, retPerLayer=True)
        else:
            return self.net.forward(in1,in2)