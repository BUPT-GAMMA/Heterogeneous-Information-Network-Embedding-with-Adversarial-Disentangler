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
from src.bi_model import LoadNN, LoadVAE
from src.util import calc_grad_penalty, vae_loss
from evaluate.ACM_evaluate import ACM_evaluation



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
else:
	mp_1 = config['exp_setting']['mp_3']
	mp_2 = config['exp_setting']['mp_4']

data_root = config['exp_setting']['data_root']
batch_size = trainer['batch_size']

mp_1_data = LoadDataset('ACM', data_root, mp = mp_1)
mp_2_data = LoadDataset('ACM', data_root, mp = mp_2)

dataset = TensorDataset(torch.LongTensor(list(range(n_nodes))), torch.FloatTensor(mp_1_data), torch.FloatTensor(mp_2_data))
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
	cat_best_micro = 0
	cat_best_macro = 0
else:
	best_auc_lr = 0
	best_f1_lr = 0
	best_acc_lr = 0

while global_step < trainer['total_step']:
	for batch_idx, mp_1_fea, mp_2_fea in data_loader:
		input_fea = torch.cat([mp_1_fea.type(torch.FloatTensor),
								mp_2_fea.type(torch.FloatTensor)], dim = 0)
		input_fea = Variable(input_fea.cuda(), requires_grad = False)
		batch_idx = batch_idx.cuda()
		length = batch_idx.size()[0]

		# Meta-path Code Setting
		mp_code = np.concatenate([np.repeat(np.array([[*[1], *[0]]]), length, axis = 0),
									np.repeat(np.array([[*[0], *[1]]]), length, axis = 0),], axis = 0)
		mp_code = torch.FloatTensor(mp_code)

		# Fake Code Setting
		# A <-> B
		fake_code = np.concatenate([np.repeat(np.array([[*[0], *[1]]]), length, axis = 0),
									np.repeat(np.array([[*[1], *[0]]]), length, axis = 0),], axis = 0)
		fake_code = torch.FloatTensor(fake_code)

		code = Variable(mp_code.cuda(), requires_grad = False)
		invert_code = 1 - code

		trans_code = Variable(fake_code.cuda(), requires_grad = False)

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

		fake_fea = vae(input_fea, batch_idx, insert_type = False)[0].detach()
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
		reconstruct_batch, mu, logvar = vae(input_fea, batch_idx, insert_type = True)
		mse, kl = vae_loss(reconstruct_batch, input_fea, mu, logvar, reconstruct_loss)
		reconstruct_batch_loss = (loss_lambda['reconstruct']['cur'] * mse + loss_lambda['kl']['cur'] * kl)
		reconstruct_batch_loss.backward()

		## Meta-path Discriminator Adversarial Phase
		embedding = vae(input_fea, return_enc = True)
		code_pred = dmp(embedding)
		adv_code_pred_loss = clf_loss(code_pred, invert_code)

		adv_mp_clf_loss = loss_lambda['adv_mp_clf']['cur'] * adv_code_pred_loss
		adv_mp_clf_loss.backward()

		## Discriminator Adversarial Phase
		embedding = vae(input_fea, return_enc = True).detach()

		fake_fea = vae.decode(embedding, batch_idx, insert_type = False)
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

	mp1_spc_embedding_matrix, mp2_spc_embedding_matrix = vae(return_mp_embedding = True)

	for batch_idx, mp_1_fea, mp_2_fea in data_loader:
		mp_1_fea = Variable(mp_1_fea.type(torch.FloatTensor).cuda(), requires_grad = False)
		mp_2_fea = Variable(mp_2_fea.type(torch.FloatTensor).cuda(), requires_grad = False)

		mp1_inv_embedding_matrix[batch_idx] = vae(mp_1_fea, return_enc = True)
		mp2_inv_embedding_matrix[batch_idx] = vae(mp_2_fea, return_enc = True)

	cat_embedding_matrix = torch.cat((mp1_inv_embedding_matrix, mp2_inv_embedding_matrix), dim = 1).cuda()

	evaluation = ACM_evaluation()

	if not change_node_type:
		# Cluster
		# print('>>***************     Cluster     ***************<<')
		# cat_NMI, cat_ARI = evaluation.evaluate_cluster(cat_embedding_matrix)
		# print('<Epoch %d> CAT		NMI = %.4f, ARI = %.4f' % (epoch, cat_NMI, cat_ARI))

		# if cat_NMI > cat_best_NMI:
		# 	cat_best_NMI = cat_NMI
		# 	cat_best_ARI = cat_ARI
		# 	if trainer['save_best_only']:
		# 		torch.save(cat_embedding_matrix, model_path + '.cat.cluster.embedding')

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
		# Logistic Regression
		auc_lr, f1_lr, acc_lr = evaluation.evaluate_lp_lr(embedding_matrix)
		print('<Epoch %d><Logistic Regression> AUC = %.4f, F1 = %.4f, Accuracy = %.4f' % (epoch, auc_lr, f1_lr, acc_lr))

		if auc_lr > best_auc_lr:
			best_auc_lr = auc_lr
			best_f1_lr = f1_lr
			best_acc_lr = acc_lr
			if trainer['save_best_only']:
				torch.save(vae, model_path + '.lp_lr.vae')
				torch.save(embedding_matrix, model_path + '.lp_lr.embedding')

if not change_node_type:
	# print('\n<Cluster> CAT		Best NMI = %.4f, Best ARI = %.4f' % (cat_best_NMI, cat_best_ARI))
	print('\n<Classification> CAT 		Best Micro-F1 = %.4f, Best Macro-F1 = %.4f' % (cat_best_micro, cat_best_macro))
else:
	print('<Link Prediction><Logistic Regression> Best AUC = %.4f, Best F1 = %.4f, Best Accuracy = %.4f' % (best_auc_lr, best_f1_lr, best_acc_lr))
	print('<Link Prediction><Dot> Best AUC = %.4f, Best F1 = %.4f, Best Accuracy = %.4f' % (best_auc_dot, best_f1_dot, best_acc_dot))
