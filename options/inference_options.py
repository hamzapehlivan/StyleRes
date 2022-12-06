from argparse import ArgumentParser

class InferenceOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		# arguments for inference script
		self.parser.add_argument('--outdir', type=str, default='results', help='Inference results save path')
		self.parser.add_argument('--checkpoint_path', default='checkpoints/styleres_ffhq.pth', type=str, help='Path to StyleRes model')
		self.parser.add_argument('--datadir', type=str, default='samples/inference_samples', help='Path to input images. ')
		self.parser.add_argument('--resize_outputs', action='store_true', help='Whether to resize outputs to 256x256 or keep at 1024x1024')

		self.parser.add_argument('--test_batch_size', default=1, type=int, help='Batch size for inference')
		self.parser.add_argument('--test_workers', default=0, type=int, help='Number of inference dataloader workers')
		self.parser.add_argument('--aligner_path', default=None, type=str, help="Optional face alignment network.")
		self.parser.add_argument('--n_images', type=int, default=None, help='Number of images to output. If None, run on all data')
		self.parser.add_argument('--edit_configs', type=str, default='options/editing_options/template.py',  help='Which edits to perform on the images. \
																									Specified in template.py file. See this file for more information ')

	def parse(self):
		opts = self.parser.parse_args()
		return opts