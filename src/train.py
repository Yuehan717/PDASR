from decimal import Decimal
import torch.nn.functional as F
import common.utility as utility
import random
import torch
import torch.nn.utils as utils
from tqdm import tqdm
from Metrics.Evaluation import calc_LPIPS
from option import args
import model
import data
import loss

class Trainer():
    def __init__(self, args, loader, my_model, loss_d, loss_p, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.model_name = args.model

        self.conADMM = utility.conADMM(args.n_colors,args.c_size, args.c_p, args.rho, filter=args.Tlf, mode=args.conMode, high=args.xorder)
        device = torch.device('cpu' if args.cpu else 'cuda')
        self.conADMM.to(device)
        self.loss_d = loss_d
        self.loss_p = loss_p
        self.optimizer_d = utility.make_optimizer(args, self.model, type_d=True)
        if self.args.load != '':
            self.optimizer_d.load(ckp.dir, epoch=len(ckp.log)) # function defined by torch
        self.optimizer_p = utility.make_optimizer(args, self.model, type_d=False)
        if self.args.load != '':
            self.optimizer_p.load(ckp.dir, epoch=len(ckp.log)) # function defined by torch

        self.error_last = 1e8 

    def train(self):
        self.loss_d.step()
        self.loss_p.step()
        
        epoch = self.optimizer_d.get_last_epoch() + 1
        lr_d = self.optimizer_d.get_lr()
        lr_p = self.optimizer_p.get_lr()
        
        self.ckp.write_log(
            '[Epoch {}]\tLearning rate (distortion): {:.2e}\tLearning rate (perception): {:.2e}'.format(epoch, Decimal(lr_d), Decimal(lr_p)) 
        )
        self.loss_d.start_log() # start to log loss from each loss item
        self.loss_p.start_log() # start to log loss from each loss item
        
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer() # utility.timer() : {acc, time}
        # TEMP
        idx_scale = random.randint(0, len(self.args.scale) - 1) # added new
        self.loader_train.dataset.set_scale(idx_scale) # idx_scale = 0, modified
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            if self.args.n_colors == 1: # Colorspace = YCrCb
                lr = lr[:,0:1,:,:]
                hr = hr[:,0:1,:,:]
            lr, hr = self.prepare(lr, hr) # assign device and manage fp
            timer_data.hold() # get time passed for data loading
            timer_model.tic() # set t0 to current time

            self.optimizer_d.zero_grad() # what is the function of this? Why called here?
            self.optimizer_p.zero_grad() # what is the function of this? Why called here?
            
            
            if self.model_name.lower().find('admm') >= 0:
                for i in range(3):
                    wvt,sr = self.model(lr, idx_scale) # idx_scale = 0, modified
                    if epoch < self.args.admmStart:
                        lossP = self.loss_p(sr, hr)
                        lossD = self.loss_d(wvt, hr)
                        lossD.backward()
                        lossP.backward()
                        break
                    else:
                        # print(i)
                        if i==0:
                            # update P(.)
                            lossP = self.loss_p(sr,hr) + self.conADMM(sr, wvt.detach(),self.model.model.S.detach())
                            lossP.backward()
                            if self.args.gclip > 0:
                                utils.clip_grad_value_(
                                    self.model.parameters(),
                                    self.args.gclip
                                )
                            self.optimizer_p.step()
                        elif i==1:
                            lossD = self.loss_d(wvt,hr) + self.conADMM(sr.detach(), wvt,self.model.model.S.detach())
                            lossD.backward()
                            if self.args.gclip > 0:
                                utils.clip_grad_value_(
                                    self.model.parameters(),
                                    self.args.gclip
                                )
                            self.optimizer_d.step()
                        else:
                            self.model.model.S += self.conADMM(sr.detach(), wvt.detach())
            
            timer_model.hold() # get time used for training in this batch

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss_d.display_loss(batch),
                    self.loss_p.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss_d.end_log(len(self.loader_train)) # get average on batch
        self.loss_p.end_log(len(self.loader_train)) # get average on batch
        
        self.error_last = self.loss_d.log[-1, -1]
        self.optimizer_d.schedule()
        self.optimizer_p.schedule()
        

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer_d.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1,4,len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test): # What type is d?
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale) # set test scale X2 X3 X4
                for lr, hr, filename in tqdm(d, ncols=80): # show progress bar
                    ### For multi-level wavlet, padding the size
                    if self.args.pad != 0:
                        lr, padding_h, padding_w = utility.pad(lr,self.args.pad, scale)
                    lr,hr = self.prepare(lr, hr)
                    ### End ###
                    # lr, hr = self.prepare(lr, hr)
                    convert = self.args.n_colors ==1 and lr.size(1)==3
                    if convert: # colorspace = ycrcb
                        SrCr = F.interpolate(lr[:,1:2,:,:], mode='bicubic', scale_factor=self.scale[0])
                        SrCb = F.interpolate(lr[:,2:3,:,:], mode='bicubic', scale_factor=self.scale[0])
                        lr = lr[:,0:1,:,:]
                        
                    srD,srP = self.model(lr, idx_scale)

                    #### For multi-level wavelet model, chop the padding ####
                    if self.args.pad != 0:
                        # print(sr.size())
                        srP = utility.chop(srP, padding_h, padding_w, scale)
                        srD = utility.chop(srD, padding_h, padding_w, scale)
                        # print(sr.size())
                    ### End ###
                    if convert:
                        srP = torch.cat((srP,SrCr,SrCb),dim=1)
                        srD = torch.cat((srD,SrCr,SrCb),dim=1)
                        
                    srP = utility.quantize(srP, self.args.rgb_range,convert=convert)
                    srD = utility.quantize(srD, self.args.rgb_range,convert=convert)
                    save_list = [srD, srP]
                    
                    hr = utility.quantize(hr, self.args.rgb_range,convert=convert)
                    
                    self.ckp.log[-1, 1 ,idx_data, idx_scale] += calc_LPIPS(
                        srD, hr, scale)
                    self.ckp.log[-1, 0 ,idx_data, idx_scale] += utility.calc_psnr(
                        srD, hr, scale, self.args.rgb_range, dataset=d
                    )
                    self.ckp.log[-1, 3 ,idx_data, idx_scale] += calc_LPIPS(
                        srP, hr, scale)
                    self.ckp.log[-1, 2 ,idx_data, idx_scale] += utility.calc_psnr(
                        srP, hr, scale, self.args.rgb_range, dataset=d
                    )
                    
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                self.ckp.log[-1, 1 , idx_data, idx_scale] /= len(d) # get average value
                self.ckp.log[-1, 0 , idx_data, idx_scale] /= len(d) # get average value
                self.ckp.log[-1, 3 , idx_data, idx_scale] /= len(d) # get average value
                self.ckp.log[-1, 2 , idx_data, idx_scale] /= len(d) # get average value
                    
                bestD = self.ckp.log[:,2,...].max(0)
                bestP = self.ckp.log[:,3,...].min(0) 
                
                self.ckp.write_log(
                    '[{} x{}]\tStage 1: : PSNR: {:.3f}\tLPIPS: {:.3f}\
                        \nStage 2: PSNR: {:.3f} (Best: {:.3f} @epoch {})\tLPIPS: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1,0,idx_data, idx_scale],
                        self.ckp.log[-1,1,idx_data, idx_scale],
                        self.ckp.log[-1,2,idx_data, idx_scale],
                        bestD[0][idx_data, idx_scale],
                        bestD[1][idx_data, idx_scale] + 1,
                        self.ckp.log[-1,3,idx_data, idx_scale],
                        bestP[0][idx_data, idx_scale],
                        bestP[1][idx_data, idx_scale] + 1
                    )
                )
                

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        inter = (self.ckp.log[-1, 3 ,idx_data, idx_scale] < 0.140) or (self.ckp.log[-1, 2 ,idx_data, idx_scale] >= 28.40)
        self.ckp.save(self, epoch, is_bestP=(bestP[1][0, 0] + 1 == epoch), is_bestD=(bestD[1][0, 0] + 1 == epoch), inter=inter) # only reasonable for training set and single scale, in other cases like MDSR, demo save all internal models 

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer_d.get_last_epoch() + 1
            return epoch >= self.args.epochs

torch.manual_seed(args.seed) #set seed for torch.rand, default seed = 1
checkpoint = utility.checkpoint(args) #set save path and load path

def main():
    global model # a global variable
    torch.autograd.set_detect_anomaly(True)
    if checkpoint.ok: # checkpoint.ok default is True
        loader = data.Data(args)
        _model = model.Model(args, checkpoint)
        _loss_d = loss.Loss(args, checkpoint,type_d=True) if not args.test_only else None
        _loss_p = loss.Loss(args, checkpoint,type_d=False) if not args.test_only else None
        t = Trainer(args, loader, _model, _loss_d, _loss_p, checkpoint)
        while not t.terminate(): # if args.test_only = True, this loop will not be executed, and test() in trainer.py will be executed only once.
            t.train()
            t.test()

        checkpoint.done()

if __name__ == '__main__':
    main()