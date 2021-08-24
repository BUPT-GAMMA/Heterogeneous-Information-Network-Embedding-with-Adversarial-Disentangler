import pickle



def LoadDataset(name, root, mp = None):
	assert mp != None
	data_root = root + name + '/' + mp +'_pre_train_embedding.p'
	with open(data_root, 'rb') as data:
		return pickle.load(data)
