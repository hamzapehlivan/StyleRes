export CUDA_VISIBLE_DEVICES="0"

REAL="" #Path to real images
FAKE="" #Path to fake images
python -m evaluation.id.calc_id_loss_parallel --num_threads=8 --output_path=$FAKE --gt_path=$REAL
