def set_template(args):
    # Set the templates here
    if args.templateD.find('GAN') >= 0:
        args.epochs = 200
        args.lr = 5e-5
        args.decay = '150'

    if args.templateD.find('RCAN') >= 0:
        args.DNet = 'RCAN'
        args.n_resgroups_d = 10
        args.n_resblocks_d = 20
        args.reduction_d = 16
        args.n_feats_d = 64
        args.chop = False
        # args.n_GPUs = 2
        
    if args.templateD.find('D_EDSR_paper') >= 0:
        args.DNet = 'EDSRsp'
        args.n_resblocks_d = 32
        args.n_feats_d = 256
        args.res_scale_d = 0.1
        args.lr_d = 1e-4
    
    if args.templateD.find('D_EDSR_medium') >= 0:
        args.DNet = 'EDSRsp'
        args.n_resblocks_d = 24
        args.n_feats_d = 256
        args.res_scale_d = 0.1
        args.lr_d = 1e-4
    
    if args.templateD == 'NLSN':
        args.DNet = 'NLSN'
        args.n_resblocks_d = 32
        args.n_feats_d = 256
        args.res_scale_d = 0.1
        args.chunk_size = 144
        args.n_hashes = 4
    
    if args.templateD == 'NLSN-':
        args.DNet = 'NLSN'
        args.n_resblocks_d = 24
        args.n_feats_d = 256
        args.res_scale_d = 0.1
        args.chunk_size = 144
        args.n_hashes = 4
        
    if args.templateD.find('HAN') >= 0:
        args.DNet = 'HAN'
        args.n_resgroups_d = 10
        args.n_resblocks_d = 20
        args.reduction_d = 16
        args.n_feats_d = 64
        args.chop = False
        
        