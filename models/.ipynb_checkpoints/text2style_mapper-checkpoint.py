import torch
from torch import nn
from mapper import feature_mappers
# from models.stylegan2.model import Generator


def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt


class Text2StyleMapper(nn.Module):

	def __init__(self, opts):
		super(Text2StyleMapper, self).__init__()
		self.opts = opts
		# Define architecture
		self.mapper = feature_mappers.FeatureMapper(self.opts)
		# self.decoder = Generator(self.opts.stylegan_size, 512, 8)
		# self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
		# Load weights if needed
		if self.opts.checkpoint_path is not None:
            self.load_weights()

	def load_weights(self):
        print('Loading from checkpoint: {}'.format(self.opts.checkpoint_path))
        ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
        self.mapper.load_state_dict(get_keys(ckpt, 'mapper'), strict=True)

	def forward(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
	            inject_latent=None, return_latents=False, alpha=None):
		if input_code:
			codes = x
		else:
			codes = self.mapper(x)
            
		return codes
