#EDSR
CUDA_VISIBLE_DEVICES=0 python3 main.py --scale 4 \
--k_bits 2 --model EDSR \
--pre_train ./pretrained/EDSR_vanilla_x4.pt --patch_size 192 \
--data_test Urban100+test2k \
--save "output/edsr_x4/3bit" --dir_data ../dataset --print_every 10 --epochs 30 --batch_size 16
