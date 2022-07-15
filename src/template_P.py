def set_template(args):
    # Set the templates here
    
    if args.templateP.find('dense') >= 0:
        args.PNet = 'RRDN'
        args.n_feats_p = 64
        args.dc_growth = 32
        args.nBlock = 1
        args.lr_p = 1e-4
    
    if args.templateP.find('multi-level') >= 0:
        args.PNet = 'MLwvt'
        args.n_feats_p = 64
        args.nBlock = 16
        args.lr_p = 1e-4
        
    if args.templateP.find('RCAN') >= 0:
        args.PNet = 'RCANwvt'
        args.n_resgroups_p = 4
        args.n_resblocks_p = 20
        args.reduction_p = 16
        args.n_feats_p = 64
        args.chop = False
    
    if args.templateP=='Clique':
        args.PNet = 'CliqueNet'
        args.nBlocks = 15
        args.n_feats_p = 32
        args.res_scale_p = 1
        args.n_stage = 2
        
    if args.templateP=='Clique-':
        args.PNet = 'CliqueNet'
        args.nBlocks = 15
        args.n_feats_p = 32
        args.res_scale_p = 1
        args.n_stage = 1
    
    if args.templateP=='ResNet':
        args.PNet = 'ResNet'
        args.n_resblocks_p = 16
        args.n_feats_p = 64
        args.res_scale_p = 1
        
    if args.templateP=='light_Clique-':
        args.PNet = 'CliqueNet'
        args.nBlocks = 10
        args.n_feats_p = 32
        args.res_scale_p = 1
        args.n_stage = 1
    
    if args.templateP=='SRGAN':
        args.PNet = 'SRGAN'
        args.n_resblocks_p = 16
        args.n_feats_p = 64
        args.res_scale_p = 1