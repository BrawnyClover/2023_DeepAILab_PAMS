#EDSR
CUDA_VISIBLE_DEVICES=1 python3 main.py --scale 4 \
--k_bits 8 --model EDSR \
--pre_train ./pretrained/edsr_baseline_x4.pt --patch_size 192 \
--data_test Urban100+test2k \
--save "output/edsr_x4/8bit" --dir_data ../dataset --print_every 10