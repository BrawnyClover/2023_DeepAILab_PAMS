# EDSR
CUDA_VISIBLE_DEVICES=1
python3 main.py --scale 4 \
--k_bits 4 --model EDSR --test_only \
--data_test Urban100+test2k+test4k \
--save "output/edsr_x4/8bit" --dir_data ../dataset