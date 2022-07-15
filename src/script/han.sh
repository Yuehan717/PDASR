## training ##
python main.py --model SRADMM --scale 4 --save sradmm_han_clique_x4_l1_CX_run3 --templateD HAN --templateP Clique \
--n_colors 3 --epochs 300 --gamma 0.5 --decay 20-50-70-100-140-200-300 --patch_size 144 \
--split_loss --lr_d 5e-5 --lr_p 5e-5 --d_loss 1*L1 --p_loss 0.1*CX+0.001*GAN+1*LF --rho 1e-4 --gclip 1e-4 \
--pre_train /home/yuehan/WFSN/experiment/sradmm/sradmm_han_clique-_x4_l1_pretrain/model/model_best_distortion.pt