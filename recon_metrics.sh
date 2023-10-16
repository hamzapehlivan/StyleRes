export CUDA_VISIBLE_DEVICES="0"

REAL="" #Path to real images
FAKE="" #Path to fake images
python -m evaluation.recons_metrics.recons --mode lpips --data_path $FAKE --gt_path $REAL
python -m evaluation.recons_metrics.recons --mode msssim --data_path $FAKE --gt_path $REAL


