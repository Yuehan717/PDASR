import argparse
import template_D
import template_P

parser = argparse.ArgumentParser(description='EDSR and MDSR')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--templateD', default='.',
                    help='You can set various templates in option.py')
parser.add_argument('--templateP', default='.',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=2,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true', # default is false, when '--cpu' exists in command line, args.cpu is set to True.
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='/home/yuehan/Dataset/',
                    help='dataset directory')
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='DIV2K',
                    help='test dataset name')
parser.add_argument('--data_range', type=str, default='1-800/801-810', # Split data to training set and validation set
                    help='train/test data range')
parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')
parser.add_argument('--scale', type=str, default= '4',
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=48,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')

# Model specifications
parser.add_argument('--model', default='SRADMM',
                    help='model name')
parser.add_argument('--DNet', type=str, default='EDSRwvt',
                    help='stage one model name')
parser.add_argument('--PNet', type=str, default='ResNet',
                    help='stage one model name')
parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='', # download
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks_d', type=int, default=16,
                    help='number of residual blocks')
parser.add_argument('--n_resblocks_p', type=int, default=16,
                    help='number of residual blocks')
parser.add_argument('--n_feats_d', type=int, default=64,
                    help='number of feature maps in distortion-oriented module')
parser.add_argument('--n_feats_p', type=int, default=64,
                    help='number of feature maps in perception-oriented module')
parser.add_argument('--res_scale_d', type=float, default=1, # default no scaling
                    help='residual scaling')
parser.add_argument('--res_scale_p', type=float, default=1, # default no scaling
                    help='residual scaling')
parser.add_argument('--shift_mean', action='store_true',
                    help='subtract pixel mean from the input')
parser.add_argument('--precision', type=str, default='single', # For CUDA optimization. Use fp32 (single) or fp16 (half)
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')
parser.add_argument('--stage1_fix', action='store_true',
                    help="fix stage one model")
parser.add_argument('--freeze_s1', action='store_true',
                    help="freeze the parameter of stage-one model")
parser.add_argument('--traineds1', type=str, default='', 
                    help="trained stage-one model")
parser.add_argument('--pad', type=int, default=0,
                    help='the times of 2 shall the image size follow')

## option for wavelet channels ##
parser.add_argument('--llrescale', type=float, default=1.0,
                    help='Enlarge factor for ll channel coefficients')
parser.add_argument('--hhrescale', type=float, default=1.0,
                    help='Enlarge factor for hh channel coefficients')

# # Option for Residual channel attention network (RCAN)
# parser.add_argument('--n_resgroups_d', type=int, default=10,
#                     help='number of residual groups')
# parser.add_argument('--n_resgroups_p', type=int, default=10,
#                     help='number of residual groups')
# parser.add_argument('--reduction_d', type=int, default=16,
#                     help='number of feature maps reduction')
# parser.add_argument('--reduction_p', type=int, default=16,
#                     help='number of feature maps reduction')

# Option for CliqueNet
parser.add_argument('--nBlocks', type=int, default=10,
                    help='number of blocks in cliquenet')
parser.add_argument('--n_stage', type=int, default=2,
                    help='number of stage in cliqueblock')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training') 
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')
parser.add_argument('--rho',type=float, default=1e-5, 
                    help='rho value for ADMM optimization')
parser.add_argument('--extra', type=float, default=0,
                    help='Add constraints on HL and LH')
parser.add_argument('--keeplf', action='store_true',
                    help='detach HH channel')
parser.add_argument('--lf_measure', type=str, default='l1',
                    help='way of calculating the loss between lr-sized images')
parser.add_argument('--c_size', type=int, default=21, 
                    help='size of Gaussian kernel or patch size for Fourier.')
parser.add_argument('--c_p', type=float, default=3,
                    help='sigma of blur kernel or poportion of remained coefficients.')


# Optimization specifications
parser.add_argument('--lr_d', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--lr_p', type=float, default=8e-5,
                    help='learning rate')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--decay', type=str, default='200',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.75,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--gclip', type=float, default=0,
                    help='gradient clipping threshold (0 = no clipping)')
parser.add_argument('--d_level', type=int, default=2,
                    help='Decomposition level fo LF loss')

# Loss specifications
parser.add_argument('--split_loss', action='store_true',
                    help='whether use different learning rates for losses.')
parser.add_argument('--d_loss', type=str, default='1*L1',
                    help='loss function for distorttion')
parser.add_argument('--p_loss', type=str, default='1*L1',
                    help='loss function for perception')
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function for perception')
parser.add_argument('--num_c', type=int, default=1, 
                    help='number of wavelet channels that used for l1 loss calculation')
parser.add_argument('--inverse', action='store_true', 
                    help='whether num_c is number of channels from position 0 or the start position.')
parser.add_argument('--ratio', type=float, default=1.0,
                    help='In wavelet based L1 loss, the weight of high-frequency channels / the weight of LL channel')
parser.add_argument('--Tlf', type=str, default='wvt', 
                    help='methods for low-frequency subband extraction')

# Log specifications
parser.add_argument('--save', type=str, default='test',
                    help='file name to save') 
parser.add_argument('--load', type=str, default='',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,  # -1 -> load latest model; other <name>, load model_<name>.pt
                    help='resume from specific checkpoint')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')
parser.add_argument('--save_gt', action='store_true',
                    help='save low-resolution and high-resolution images together')

args = parser.parse_args()
# template setted by author, for convenience
template_P.set_template(args)
template_D.set_template(args)

args.scale = list(map(lambda x: int(x), args.scale.split('+')))
args.data_train = args.data_train.split('+')
args.data_test = args.data_test.split('+')

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
# The vars() function returns the __dic__ attribute of an object.
# Use Boolean to substitute string.
