export CUDA_VISIBLE_DEVICES="0"

REAL="" #Path to real images
FAKE="" #Path to fake images

python -m evaluation.fid.fid --paths $REAL $FAKE

