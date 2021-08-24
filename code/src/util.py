import torch
from torch.autograd import Variable, grad



def calc_grad_penalty(dnet, real_data, fake_data, use_gpu = True, dec_output = 2):
	alpha = torch.rand(real_data.shape[0], 1)
	alpha = alpha.expand(real_data.size())
	if use_gpu:
		alpha = alpha.cuda()

	interpolates = alpha * real_data + ((1 - alpha) * fake_data)

	if use_gpu:
		interpolates = interpolates.cuda()
	interpolates = Variable(interpolates, requires_grad = True)

	if dec_output == 2:
		disc_interpolates, _ = dnet(interpolates)
	else:
		disc_interpolates = dnet(interpolates)

	grads = grad(outputs = disc_interpolates, inputs = interpolates,
				 grad_outputs = torch.ones(disc_interpolates.size()).cuda(),
				 create_graph = True, retain_graph = True, only_inputs = True)[0]
	grad_penalty = ((grads.norm(2, dim = 1) - 1) ** 2).mean()
	return grad_penalty


def vae_loss(reconstruct_x, x, mu, logvar, reconstruct_loss):
	loss = reconstruct_loss(reconstruct_x, x)
	KLD = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	return loss, KLD
