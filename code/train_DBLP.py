from torch.backends import cudnn
import sys
import yaml
import os
import shutil
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import warnings

from src.data import LoadDataset
from src.tri_model import LoadNN, LoadVAE
from src.util import calc_grad_penalty, vae_loss
from evaluate.DBLP_evaluate import DBLP_evaluation



# Ingore Warnings
warnings.filterwarnings('ignore')

# Experimental Setting
cudnn.benchmark = True

change_node_type = False

config_path = sys.argv[1]
config = yaml.load(open(config_path, 'r'))

exp_name = config['exp_setting']['exp_name']
input_dim = config['exp_setting']['input_dim']
n_nodes = config['exp_setting']['n_nodes']

trainer = config['trainer']
if trainer['save_checkpoint']:
	model_path = config['exp_setting']['checkpoint_dir'] + exp_name + '/'
	if not os.path.exists(model_path):
		os.makedirs(model_path)
	model_path = model_path + '{}'
if trainer['save_log']:
	if os.path.exists(config['exp_setting']['log_dir'] + exp_name):
		shutil.rmtree(config['exp_setting']['log_dir'] + exp_name)
	writer = SummaryWriter(config['exp_setting']['log_dir'] + exp_name)

# Fix Seed
np.random.seed(config['exp_setting']['seed'])
torch.manual_seed(config['exp_setting']['seed'])

# Load Dataset
if not change_node_type:
	mp_1 = config['exp_setting']['mp_1']
	mp_2 = config['exp_setting']['mp_2']
	mp_3 = config['exp_setting']['mp_3']
else:
	mp_4 = config['exp_setting']['mp_4']
	mp_5 = config['exp_setting']['mp_5']
	mp_6 = config['exp_setting']['mp_6']

data_root = config['exp_setting']['data_root']
batch_size = trainer['batch_size']

mp_1_data = LoadDataset('DBLP', data_root, mp = mp_1)
mp_2_data = LoadDataset('DBLP', data_root, mp = mp_2)
mp_3_data = LoadDataset('DBLP', data_root, mp = mp_3)

dataset = TensorDataset(torch.LongTensor(list(range(n_nodes))), torch.FloatTensor(mp_1_data), torch.FloatTensor(mp_2_data), torch.FloatTensor(mp_3_data))
data_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

# Load Model
embed_dim = config['model']['vae']['encoder'][-1][1]
vae_lr = config['model']['vae']['lr']
vae_betas = tuple(config['model']['vae']['betas'])
d_lr = config['model']['D']['lr']
d_betas = tuple(config['model']['D']['betas'])
dmp_lr = config['model']['D_mp']['lr']
dmp_betas = tuple(config['model']['D_mp']['betas'])

vae = LoadVAE(config['model']['vae'], n_nodes, input_dim, embed_dim)
d = LoadNN(config['model']['D'], input_dim)
dmp = LoadNN(config['model']['D_mp'], embed_dim)

reconstruct_loss = nn.MSELoss()
clf_loss = nn.BCEWithLogitsLoss()

# Use CUDA
vae = vae.cuda()
d = d.cuda()
dmp = dmp.cuda()

reconstruct_loss = reconstruct_loss.cuda()
clf_loss = clf_loss.cuda()

# Optimizer
opt_vae = optim.Adam(list(vae.parameters()), lr = vae_lr, betas = vae_betas)
opt_d = optim.Adam(list(d.parameters()), lr = d_lr, betas = d_betas)
opt_dmp = optim.Adam(list(dmp.parameters()), lr = dmp_lr, betas = dmp_betas)

# Training
vae.train()
d.train()
dmp.train()

# Loss Weight Setting
loss_lambda = {}
for k in trainer['lambda'].keys():
	init = trainer['lambda'][k]['init']
	final = trainer['lambda'][k]['final']
	step = trainer['lambda'][k]['step']
	loss_lambda[k] = {}
	loss_lambda[k]['cur'] = init
	loss_lambda[k]['inc'] = (final - init) / step
	loss_lambda[k]['final'] = final

# Training
global_step = 0
epoch = 0

if not change_node_type:
	cat_best_NMI = 0
	cat_best_ARI = 0
	# mp1_spc_best_NMI = 0
	# mp1_spc_best_ARI = 0
	# mp2_spc_best_NMI = 0
	# mp2_spc_best_ARI = 0
	# mp3_spc_best_NMI = 0
	# mp3_spc_best_ARI = 0
	cat_best_micro = 0
	cat_best_macro = 0
else:
	best_auc_lr = 0
	best_f1_lr = 0
	best_acc_lr = 0

while global_step < trainer['total_step']:
	for batch_idx, mp_1_fea, mp_2_fea, mp_3_fea in data_loader:
		input_fea = torch.cat([mp_1_fea.type(torch.FloatTensor),
								mp_2_fea.type(torch.FloatTensor),
								mp_3_fea.type(torch.FloatTensor)], dim = 0)
		input_fea = Variable(input_fea.cuda(), requires_grad = False)
		batch_idx = batch_idx.cuda()
		length = batch_idx.size()[0]

		# Meta-path Code Setting
		mp_code = np.concatenate([np.repeat(np.array([[*[1], *[0], *[0]]]), length, axis = 0),
									np.repeat(np.array([[*[0], *[1], *[0]]]), length, axis = 0),
									np.repeat(np.array([[*[0], *[0], *[1]]]), length, axis = 0),], axis = 0)
		mp_code = torch.FloatTensor(mp_code)

		# Fake Code Setting
		# Forword Translation Code: A -> B -> C -> A
		forward_code = np.concatenate([np.repeat(np.array([[*[0], *[1], *[0]]]), length, axis = 0),
										np.repeat(np.array([[*[0], *[0], *[1]]]), length, axis = 0),
										np.repeat(np.array([[*[1], *[0], *[0]]]), length, axis = 0),], axis = 0)
		forward_code = torch.FloatTensor(forward_code)

		# Backword Translation Code: C -> B -> A -> C
		backward_code = np.concatenate([np.repeat(np.array([[*[0], *[0], *[1]]]), length, axis = 0),
										np.repeat(np.array([[*[1], *[0], *[0]]]), length, axis = 0),
										np.repeat(np.array([[*[0], *[1], *[0]]]), length, axis = 0),], axis = 0)
		backward_code = torch.FloatTensor(backward_code)

		code = Variable(mp_code.cuda(), requires_grad = False)
		invert_code = 1 - code

		if global_step % 2 == 0:
			trans_code = Variable(torch.FloatTensor(forward_code).cuda(), requires_grad = False)
		else:
			trans_code = Variable(torch.FloatTensor(backward_code).cuda(), requires_grad = False)

		# Train Meta-path Discriminator
		opt_dmp.zero_grad()
		embedding = vae(input_fea, return_enc = True).detach()
		code_pred = dmp(embedding)

		dmp_loss = clf_loss(code_pred, code)
		dmp_loss.backward()
		opt_dmp.step()

		# Training Discriminator
		opt_d.zero_grad()
		real_pred, real_code_pred = d(input_fea)

		if global_step % 2 == 0:
			fake_fea = vae(input_fea, batch_idx, insert_type = 'forward')[0].detach()
		else:
			fake_fea = vae(input_fea, batch_idx, insert_type = 'backward')[0].detach()

		fake_pred = d(fake_fea)[0]
		real_pred = real_pred.mean()
		fake_pred = fake_pred.mean()

		gp = loss_lambda['gp']['cur'] * calc_grad_penalty(d, input_fea.data, fake_fea.data)
		real_code_pred_loss = clf_loss(real_code_pred, code)

		d_loss = real_code_pred_loss + fake_pred - real_pred + gp
		d_loss.backward()
		opt_d.step()

		# Train VAE
		opt_vae.zero_grad()

		## Reconstruction Phase
		reconstruct_batch, mu, logvar = vae(input_fea, batch_idx, insert_type = 'truth')
		mse, kl = vae_loss(reconstruct_batch, input_fea, mu, logvar, reconstruct_loss)
		reconstruct_batch_loss = loss_lambda['reconstruct']['cur'] * mse + loss_lambda['kl']['cur'] * kl
		reconstruct_batch_loss.backward()

		## Meta-path Discriminator Adversarial Phase
		embedding = vae(input_fea, return_enc = True)
		code_pred = dmp(embedding)
		adv_code_pred_loss = clf_loss(code_pred, invert_code)

		adv_mp_clf_loss = loss_lambda['adv_mp_clf']['cur'] * adv_code_pred_loss
		adv_mp_clf_loss.backward()

		## Discriminator Adversarial Phase
		embedding = vae(input_fea, return_enc = True).detach()

		if global_step % 2 == 0:
			fake_fea = vae.decode(embedding, batch_idx, insert_type = 'forward')
		else:
			fake_fea = vae.decode(embedding, batch_idx, insert_type = 'backward')

		adv_d_loss, d_code_pred = d(fake_fea)
		adv_d_loss = adv_d_loss.mean()
		d_clf_loss = clf_loss(d_code_pred, trans_code)

		d_loss = - loss_lambda['d_adv']['cur'] * adv_d_loss + loss_lambda['d_clf']['cur'] * d_clf_loss
		d_loss.backward()
		opt_vae.step()

		# End of Step
		print('Step ', global_step, end = '\r', flush = True)
		global_step += 1

		# Records
		if trainer['save_log'] and (global_step % trainer['verbose_step'] == 0):
			writer.add_scalar('MSE', mse.item(), global_step)
			writer.add_scalar('KL', kl.item(), global_step)
			writer.add_scalar('Gradient Penalty', gp.item(), global_step)
			writer.add_scalars('Real_Fake Discriminator', {'Real':real_pred.item(),
															'Fake':fake_pred.item()}, global_step)
			writer.add_scalars('Meta-path Discriminator', {'Real':real_code_pred_loss.item(),
															'Fake':d_clf_loss.item()}, global_step)
			writer.add_scalars('Adversarial Meta-path Discriminator', {'Classifier':dmp_loss.item(),
																		'Adversarial Classifier':adv_mp_clf_loss.item()}, global_step)

		# Update Lambda
		for k in loss_lambda.keys():
			if not loss_lambda[k]['cur'] > loss_lambda[k]['final']:
				loss_lambda[k]['cur'] += loss_lambda[k]['inc']

	# Test
	epoch += 1

	vae.eval()

	mp1_inv_embedding_matrix = torch.empty(n_nodes, embed_dim).cuda()
	mp2_inv_embedding_matrix = torch.empty(n_nodes, embed_dim).cuda()
	mp3_inv_embedding_matrix = torch.empty(n_nodes, embed_dim).cuda()

	mp1_spc_embedding_matrix, mp2_spc_embedding_matrix, mp3_spc_embedding_matrix = vae(return_mp_embedding = True)

	for batch_idx, mp_1_fea, mp_2_fea, mp_3_fea in data_loader:
		mp_1_fea = Variable(mp_1_fea.type(torch.FloatTensor).cuda(), requires_grad = False)
		mp_2_fea = Variable(mp_2_fea.type(torch.FloatTensor).cuda(), requires_grad = False)
		mp_3_fea = Variable(mp_3_fea.type(torch.FloatTensor).cuda(), requires_grad = False)

		mp1_inv_embedding_matrix[batch_idx] = vae(mp_1_fea, return_enc = True)
		mp2_inv_embedding_matrix[batch_idx] = vae(mp_2_fea, return_enc = True)
		mp3_inv_embedding_matrix[batch_idx] = vae(mp_3_fea, return_enc = True)

	cat_embedding_matrix = torch.cat((mp1_inv_embedding_matrix, mp2_inv_embedding_matrix, mp3_inv_embedding_matrix), dim = 1).cuda()

	evaluation = DBLP_evaluation()

	if not change_node_type:
		# # Cluster
		# print('>>***************     Cluster     ***************<<')
		# cat_NMI, cat_ARI = evaluation.evaluate_cluster(cat_embedding_matrix)
		# print('<Epoch %d> CAT		NMI = %.4f, ARI = %.4f' % (epoch, cat_NMI, cat_ARI))
		# # mp1_spc_NMI, mp1_spc_ARI = evaluation.evaluate_cluster(mp1_spc_embedding_matrix)
		# # print('<Epoch %d> MP1 SPC	NMI = %.4f, ARI = %.4f' % (epoch, mp1_spc_NMI, mp1_spc_ARI))
		# # mp2_spc_NMI, mp2_spc_ARI = evaluation.evaluate_cluster(mp2_spc_embedding_matrix)
		# # print('<Epoch %d> MP2 SPC	NMI = %.4f, ARI = %.4f' % (epoch, mp2_spc_NMI, mp2_spc_ARI))
		# # mp3_spc_NMI, mp3_spc_ARI = evaluation.evaluate_cluster(mp3_spc_embedding_matrix)
		# # print('<Epoch %d> MP3 SPC	NMI = %.4f, ARI = %.4f' % (epoch, mp3_spc_NMI, mp3_spc_ARI))

		# if cat_NMI > cat_best_NMI:
		# 	cat_best_NMI = cat_NMI
		# 	cat_best_ARI = cat_ARI
		# 	if trainer['save_best_only']:
		# 		torch.save(cat_embedding_matrix, model_path + '.cat.cluster.embedding')
		# if mp1_spc_NMI > mp1_spc_best_NMI:
		# 	mp1_spc_best_NMI = mp1_spc_NMI
		# 	mp1_spc_best_ARI = mp1_spc_ARI
		# 	if trainer['save_best_only']:
		# 		torch.save(mp1_spc_embedding_matrix, model_path + '.mp1_spc.cluster.embedding')
		# if mp2_spc_NMI > mp2_spc_best_NMI:
		# 	mp2_spc_best_NMI = mp2_spc_NMI
		# 	mp2_spc_best_ARI = mp2_spc_ARI
		# 	if trainer['save_best_only']:
		# 		torch.save(mp2_spc_embedding_matrix, model_path + '.mp2_spc.cluster.embedding')
		# if mp3_spc_NMI > mp3_spc_best_NMI:
		# 	mp3_spc_best_NMI = mp3_spc_NMI
		# 	mp3_spc_best_ARI = mp3_spc_ARI
		# 	if trainer['save_best_only']:
		# 		torch.save(mp3_spc_embedding_matrix, model_path + '.mp3_spc.cluster.embedding')

		# # if trainer['save_log']:
		# # 	writer.add_scalars('NMI', {'Invariant Embedding':cat_NMI,
		# # 								'MP1 Specific Embedding':mp1_spc_NMI,
		# # 								'MP2 Specific Embedding':mp2_spc_NMI,
		# # 								'MP3 Specific Embedding':mp3_spc_NMI}, epoch)

		# Classification
		print('>>*************** Classification ***************<<')
		cat_micro, cat_macro = evaluation.evaluate_clf(cat_embedding_matrix)
		print('<Epoch %d> CAT 		Micro-F1 = %.4f, Macro-F1 = %.4f' % (epoch, cat_micro, cat_macro))

		if cat_micro > cat_best_micro:
			cat_best_micro = cat_micro
			cat_best_macro = cat_macro
			if trainer['save_best_only']:
				torch.save(cat_embedding_matrix, model_path + '.cat.clf.embedding')

	else:
		# Link Prediction
		auc_lr, f1_lr, acc_lr = evaluation.evaluate_lp_lr(embedding_matrix)
		print('<Epoch %d><Link Prediction> AUC = %.4f, F1 = %.4f, Accuracy = %.4f' % (epoch, auc_lr, f1_lr, acc_lr))

		if auc_lr > best_auc_lr:
			best_auc_lr = auc_lr
			best_f1_lr = f1_lr
			best_acc_lr = acc_lr
			if trainer['save_best_only']:
				torch.save(embedding_matrix, model_path + '.lp_lr.embedding')

if not change_node_type:
	# print('\n<Cluster> CAT		Best NMI = %.4f, Best ARI = %.4f' % (cat_best_NMI, cat_best_ARI))
	# # print('<Cluster> MP1 SPC	Best NMI = %.4f, Best ARI = %.4f' % (mp1_spc_best_NMI, mp1_spc_best_ARI))
	# # print('<Cluster> MP2 SPC	Best NMI = %.4f, Best ARI = %.4f' % (mp2_spc_best_NMI, mp2_spc_best_ARI))
	# # print('<Cluster> MP3 SPC	Best NMI = %.4f, Best ARI = %.4f' % (mp3_spc_best_NMI, mp3_spc_best_ARI))
	print('<Classification> CAT		Best Micro-F1 = %.4f, Best Macro-F1 = %.4f' % (cat_best_micro, cat_best_macro))
else:
	print('<Link Prediction> Best AUC = %.4f, Best F1 = %.4f, Best Accuracy = %.4f' % (best_auc_lr, best_f1_lr, best_acc_lr))
