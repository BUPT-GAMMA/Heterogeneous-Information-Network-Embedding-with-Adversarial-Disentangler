import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import torch
import pickle
import heapq
import math



class Yelp_evaluation():
	def __init__(self):
		# Load Business Label
		self.business_label = {}
		data = pickle.load(open('../datasets/Yelp/business_label.p', 'rb'))
		for item in range(len(data)):
			 (id, label) = data[item]
			 self.business_label[id] = label

	def evaluate_cluster(self, embedding_matrix):
		embedding_list = embedding_matrix.tolist()

		X = []
		Y = []
		for p in self.business_label:
			X.append(embedding_list[p])
			Y.append(self.business_label[p])

		Y_pred = KMeans(4).fit(np.array(X)).predict(X)
		nmi =  normalized_mutual_info_score(np.array(Y), Y_pred)
		ari = adjusted_rand_score(np.array(Y), Y_pred)
		return nmi, ari

	def evaluate_clf(self, embedding_matrix):
		embedding_list = embedding_matrix.tolist()

		X = []
		Y = []
		for p in self.business_label:
			X.append(embedding_list[p])
			Y.append(self.business_label[p])

		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

		LR = LogisticRegression()
		LR.fit(X_train, Y_train)
		Y_pred = LR.predict(X_test)

		micro_f1 = f1_score(Y_test, Y_pred, average = 'micro')
		macro_f1 = f1_score(Y_test, Y_pred, average = 'macro')
		return micro_f1, macro_f1
