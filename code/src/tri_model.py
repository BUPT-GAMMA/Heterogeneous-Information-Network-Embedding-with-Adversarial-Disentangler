import torch
import torch.nn as nn
from torch.autograd import Variable



def LoadVAE(parameter, n_nodes, input_dim, embed_dim):
	enc_list = []
	for para in parameter['encoder']:
		if para[0] == 'fc':
			next_dim, bn, act, dropout = para[1:5]
			act = get_act(act)
			enc_list.append((para[0], (input_dim, next_dim, bn, act, dropout)))
			input_dim = next_dim
		else:
			raise NameError('Unknown encoder layer type:' + para[0])

	dec_list = []
	for para in parameter['decoder']:
		if para[0] == 'fc':
			next_dim, bn, act, dropout, insert_code = para[1:6]
			act = get_act(act)
			dec_list.append((para[0], (input_dim, next_dim, bn, act, dropout), insert_code))
			input_dim = next_dim
		else:
			raise NameError('Unknown decoder layer type:' + para[0])
	return HEAD(enc_list, dec_list, n_nodes, embed_dim)


def LoadNN(parameter, input_dim):
	dnet_list = []
	for para in parameter['dnn']:
		if para[0] == 'fc':
			next_dim, bn, act, dropout = para[1:5]
			act = get_act(act)
			dnet_list.append((para[0], (input_dim, next_dim, bn, act, dropout)))
			input_dim = next_dim
		else:
			raise NameError('Unknown nn layer type:' + para[0])
	return Discriminator(dnet_list)


def get_act(name):
	if name == 'LeakyReLU':
		return nn.LeakyReLU(0.2)
	elif name == 'ReLU':
		return nn.ReLU()
	elif name == 'Tanh':
		return nn.Tanh()
	elif name == '':
		return None
	else:
		raise NameError('Unknown activation:' + name)


class HEAD(nn.Module):
	def __init__(self, enc_list, dec_list, n_nodes, embed_dim):
		super(HEAD, self).__init__()

		# Meta-path Specific Embedding
		self.mp_1_embedding = nn.Embedding(n_nodes, embed_dim)
		self.mp_2_embedding = nn.Embedding(n_nodes, embed_dim)
		self.mp_3_embedding = nn.Embedding(n_nodes, embed_dim)

		# Encoder
		self.enc_layers = []
		for l in range(len(enc_list)):
			self.enc_layers.append(enc_list[l][0])
			if enc_list[l][0] == 'fc':
				embed_in, embed_out, norm, act, dropout = enc_list[l][1]
				if l == len(enc_list) - 1:
					setattr(self, 'enc_mu', FC(embed_in, embed_out, norm, act, dropout))
					setattr(self, 'enc_logvar', FC(embed_in, embed_out, norm, act, dropout))
				else:
					setattr(self, 'enc_' + str(l), FC(embed_in, embed_out, norm, act, dropout))
			else:
				raise ValueError('Unreconized layer type')

		# Decoder
		self.dec_layers = []
		for l in range(len(dec_list)):
			self.dec_layers.append((dec_list[l][0], dec_list[l][2]))
			if dec_list[l][0] == 'fc':
				embed_in, embed_out, norm, act, dropout = dec_list[l][1]
				if dec_list[l][2]:
					embed_in += embed_dim
				setattr(self, 'dec_' + str(l), FC(embed_in, embed_out, norm, act, dropout))
			else:
				raise ValueError('Unreconized layer type')

		self.apply(weights_init)

	def encode(self, x):
		for l in range(len(self.enc_layers) - 1):
			if self.enc_layers[l] == 'fc':
				batch_size = x.size()[0]
				x = x.view(batch_size, -1)
			x = getattr(self, 'enc_' + str(l))(x)

		if self.enc_layers[-1] == 'fc':
			batch_size = x.size()[0]
			x = x.view(batch_size, -1)

		mu = getattr(self, 'enc_mu')(x)
		logvar = getattr(self, 'enc_logvar')(x)
		return mu, logvar

	def decode(self, z, insert_code_idx, insert_type):
		if insert_code_idx is not None:
			# Meta-path Code Setting
			if insert_type == 'truth':
				insert_code = torch.cat([self.mp_1_embedding(insert_code_idx),
										 self.mp_2_embedding(insert_code_idx),
										 self.mp_3_embedding(insert_code_idx)], dim = 0)
			# Fake Code Setting
			# A -> B -> C -> A
			elif insert_type == 'forward':
				insert_code = torch.cat([self.mp_2_embedding(insert_code_idx),
										 self.mp_3_embedding(insert_code_idx),
										 self.mp_1_embedding(insert_code_idx)], dim = 0)
			# C -> B -> A -> C
			elif insert_type == 'backward':
				insert_code = torch.cat([self.mp_3_embedding(insert_code_idx),
										 self.mp_1_embedding(insert_code_idx),
										 self.mp_2_embedding(insert_code_idx)], dim = 0)
			else:
				raise NameError('Unknown insert type:' + insert_type)

		for l in range(len(self.dec_layers)):
			if (insert_code is not None) and (self.dec_layers[l][1]):
				z = torch.cat([z, insert_code], dim = 1)
			z = getattr(self, 'dec_' + str(l))(z)
		return z

	def reparameterize(self, mu, logvar):
		if self.training:
			std = logvar.mul(0.5).exp_()
			eps = Variable(std.data.new(std.size()).normal_(std = 10))
			return eps.mul(std).add_(mu)
		else:
			return mu

	def forward(self, x = None, insert_code_idx = None, insert_type = 'truth', return_enc = False, return_mp_embedding = False):
		if return_mp_embedding:
			return self.mp_1_embedding.weight, self.mp_2_embedding.weight, self.mp_3_embedding.weight
		batch_size = x.size()[0]
		x = x.view(batch_size, -1)
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		if return_enc:
			return z
		return self.decode(z, insert_code_idx, insert_type), mu, logvar


def FC(embed_in, embed_out, norm = 'bn', activation = None, dropout = None):
	layers = []
	layers.append(nn.Linear(embed_in, embed_out))
	if dropout is not None:
		if dropout > 0:
			layers.append(nn.Dropout(dropout))
	if norm == 'bn':
		layers.append(nn.BatchNorm1d(embed_out))
	if activation is not None:
		layers.append(activation)
	return nn.Sequential(*layers)


def weights_init(model):
	classname = model.__class__.__name__
	if classname.find('BatchNorm') != -1:
		model.weight.data.normal_(0.0, 0.02)
		model.bias.data.fill_(0)


class Discriminator(nn.Module):
	def __init__(self, layer_list):
		super(Discriminator, self).__init__()

		self.layer_list = []
		for l in range(len(layer_list) - 1):
			self.layer_list.append(layer_list[l][0])
			if layer_list[l][0] == 'fc':
				embed_in, embed_out, norm, act, dropout = layer_list[l][1]
				setattr(self, 'layer_' + str(l), FC(embed_in, embed_out, norm, act, dropout))
			else:
				raise ValueError('Unreconized layer type')

		self.layer_list.append(layer_list[-1][0])
		embed_in, embed_out, norm, act, _ = layer_list[-1][1]
		if not isinstance(embed_out, list):
			embed_out = [embed_out]
		self.output_amount = len(embed_out)

		for idx, d in enumerate(embed_out):
			setattr(self, 'layer_out_' + str(idx), FC(embed_in, d, norm, act, 0))

		self.apply(weights_init)

	def forward(self, x):
		for l in range(len(self.layer_list) - 1):
			x = getattr(self, 'layer_' + str(l))(x)

		output = []
		for d in range(self.output_amount):
			output.append(getattr(self, 'layer_out_' + str(d))(x))

		if self.output_amount == 1:
			return output[0]
		else:
			return tuple(output)
