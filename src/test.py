import torch.nn.functional as F
import torch
import torch.nn.utils as utils
from tqdm import tqdm
from Metrics.Evaluation import calc_LPIPS
import torch
import common.utility as utility
import data
import model
from option import args

class Tester():
    def __init__(self, args, loader, my_model, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_test = loader.loader_test
        self.model = my_model
        self.model_name = args.model

    def test(self):
        torch.set_grad_enabled(False)

        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(2, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale) # set test scale X2 X3 X4
                for lr, hr, filename in tqdm(d, ncols=80): # show progress bar
                    ### For multi-level wavlet, padding the size
                    if self.args.pad != 0:
                        lr, padding_h, padding_w = utility.pad(lr,self.args.pad, scale)
                    lr,hr = self.prepare(lr, hr)
                    ### End ###
                    
                    idx_scale_4model = [2,3,4].index(scale)
                    _,sr = self.model(lr, idx_scale_4model)
                    if self.args.pad != 0:
                        sr = utility.chop(sr, padding_h, padding_w, scale)
                    ### End ###
                   
                    sr = utility.quantize(sr, self.args.rgb_range)
                    save_list = [sr]
                    
                    hr = utility.quantize(hr, self.args.rgb_range)
                    
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    self.ckp.log[-2, idx_data, idx_scale] += calc_LPIPS(
                        sr, hr, scale)
                        
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                
                self.ckp.log[-1, idx_data, idx_scale] /= len(d) # get average value of PSNR
                self.ckp.log[-2, idx_data, idx_scale] /= len(d) # get average value of LPIPS
                    
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} LPIPS: {:.4f}'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        self.ckp.log[-2, idx_data, idx_scale]
                    )
                )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            # if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]



torch.manual_seed(args.seed) #set seed for torch.rand, default seed = 1
checkpoint = utility.checkpoint(args) #set save path and load path

def main():
    global model # a global variable
    # torch.autograd.set_detect_anomaly(False)
    if checkpoint.ok: # checkpoint.ok default is True
        loader = data.Data(args)
        _model = model.Model(args, checkpoint)
        t = Tester(args, loader, _model, checkpoint)
        t.test()

        checkpoint.done()

if __name__ == '__main__':
    main()
