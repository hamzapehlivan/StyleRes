#Set GPU
export CUDA_VISIBLE_DEVICES='0'

DATADIR='samples/inference_samples'
OUTDIR='results'
EDITCFG='options/editing_options/template.py'

python inference.py --datadir=$DATADIR --outdir=$OUTDIR --edit_configs=$EDITCFG