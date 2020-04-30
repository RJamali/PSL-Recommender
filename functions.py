import os
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from itertools import product
import math

def load_matrices(dataset, folder):
    observation_mat = np.load(os.path.join(folder, dataset+"_protein_locals.npy"))  # protein subcellular localization observation matrix
    proteins_pssm_sim = np.load(os.path.join(folder, dataset+"_pssm_sim.npy")) # proteins pssm similarity matrix
    proteins_bp_sim = np.load(os.path.join(folder, dataset+"_bp_sim.npy")) # proteins bp terms similarity matrix
    proteins_mf_sim = np.load(os.path.join(folder, dataset+"_mf_sim.npy")) # proteins mf terms similarity matrix

    if dataset == "Deeploc":
        proteins_sim=  ((1 * proteins_pssm_sim)  + (3 * proteins_bp_sim) + (1 * proteins_mf_sim)) / (5) #compute weighted average of similarities
    else:
        proteins_cc_sim = np.load(os.path.join(folder, dataset+"_cc_sim.npy")) # proteins cc terms similarity matrix
        if dataset != "HumT":
            proteins_sim=  ((1 * proteins_pssm_sim)  + (3 * proteins_bp_sim) + (4 * proteins_cc_sim) + (1 * proteins_mf_sim)) / (9) #compute weighted average of similarities
        else:
            proteins_string_sim = np.load(os.path.join(folder, dataset+"_string_sim.npy"))  # proteins string similarity matrix
            proteins_sim =  ((1 * proteins_pssm_sim) + (1 * proteins_string_sim) + (3 * proteins_bp_sim) + (4 * proteins_cc_sim) + (1 * proteins_mf_sim)) / (10) #compute weighted average of similarities

    return observation_mat, proteins_sim

def apply_threshold(scores, thre):
    prediction = np.zeros_like(scores)
    for i in range(len(scores)):
        threshold = max(scores[i]) - ((max(scores[i]) - min(scores[i])) * (thre))
        for j in range(len(scores[i])):
            if scores[i][j] >= threshold:
                prediction[i][j] = 1
    return prediction

def compute_evaluation_criteria(true_result, prediction):
    pres = np.zeros(len(true_result))
    recs = np.zeros(len(true_result))
    ACC, ACC2, AVG = 0.0, 0.0, 0.0
    fmeas=[]
    #computineach protein's ACC2 and AVG
    for i in range(true_result.shape[1]):
        tn, fp, fn, tp = confusion_matrix(true_result[:, i], prediction[:, i]).ravel()
        AVG += tp / ((tp + fn) * (true_result.shape[1]))
        ACC2 += (tp + tn) / (true_result.shape[0] * true_result.shape[1])
    # computing ACC, Precision, and recall for each protein
    for i in range(true_result.shape[0]):
        tn, fp, fn, tp = confusion_matrix(true_result[i], prediction[i]).ravel()
        ACC += (tp / (tp + fp + fn))
        pres[i] = precision_score(true_result[i], prediction[i])
        recs[i] = recall_score(true_result[i], prediction[i])
    ACC = ACC / (len(true_result))
    # Computing Precision, Recall, and F1 for each subcellular location
    for i in range(true_result.shape[1]):
        recall,  precision, r_loc, p_loc = 0, 0, 0, 0
        for j in range(0, len(true_result)):
            if true_result[j][i] == 1:
                recall += recs[j]
                r_loc += 1
            if prediction[j][i] == 1:
                precision += pres[j]
                p_loc += 1
        if r_loc != 0:
            recall = recall / r_loc
        else:
            recall = 0
        if p_loc != 0:
            precision = precision / p_loc
        else:
            precision = 0
        if (precision == 0 and recall == 0):
            fmeas.append(0)
        else:
            fmeas.append(round((2 * recall * precision) / (precision + recall), 4))
    finalF = np.mean(fmeas)
    return round(finalF,2), round(ACC,2), round(AVG,2), round(ACC2,2)


# ***this part of code extracted from Deeploc paper (Deeploc evaluation metrics)***
class ConfusionMatrix:

	def __init__(self, num_classes, class_names=None):
		self.n_classes = num_classes
		if class_names is None:
			self.class_names = map(str, range(num_classes))
		else:
			self.class_names = class_names

		# find max class_name and pad
		max_len = max(map(len, self.class_names))
		self.max_len = max_len
		for idx, name in enumerate(self.class_names):
			if len(self.class_names) < max_len:
				self.class_names[idx] = name + " "*(max_len-len(name))

		self.mat = np.zeros((num_classes,num_classes),dtype='int')

	def __str__(self):
		# calucate row and column sums
		col_sum = np.sum(self.mat, axis=1)
		row_sum = np.sum(self.mat, axis=0)

		s = []

		mat_str = self.mat.__str__()
		mat_str = mat_str.replace('[','').replace(']','').split('\n')

		for idx, row in enumerate(mat_str):
			if idx == 0:
				pad = " "
			else:
				pad = ""
			class_name = self.class_names[idx]
			class_name = " " + class_name + " |"
			row_str = class_name + pad + row
			row_str += " |" + str(col_sum[idx])
			s.append(row_str)

		row_sum = [(self.max_len+4)*" "+" ".join(map(str, row_sum))]
		hline = [(1+self.max_len)*" "+"-"*len(row_sum[0])]

		s = hline + s + hline + row_sum

		# add linebreaks
		s_out = [line+'\n' for line in s]
		return "".join(s_out)

	def add_data(self, targets, preds):
		assert targets.shape == preds.shape
		assert len(targets) == len(preds)
		assert max(targets) < self.n_classes
		assert max(preds) < self.n_classes
		targets = targets.flatten()
		preds = preds.flatten()
		for i in range(len(targets)):
			self.mat[targets[i], preds[i]] += 1

	def ret_mat(self):
		return self.mat

	def get_errors(self):
		tp = np.asarray(np.diag(self.mat).flatten(),dtype='float')
		fn = np.asarray(np.sum(self.mat, axis=1).flatten(),dtype='float') - tp
		fp = np.asarray(np.sum(self.mat, axis=0).flatten(),dtype='float') - tp
		tn = np.asarray(np.sum(self.mat)*np.ones(self.n_classes).flatten(),
						dtype='float') - tp - fn - fp
		return tp, fn, fp, tn

	def accuracy(self):
		tp, _, _, _ = self.get_errors()
		n_samples = np.sum(self.mat)
		return np.sum(tp) / n_samples

def gorodkin(z):

    k = z.shape[0]
    n = np.sum(z)

    t2 = sum(np.dot(z[i,:], z[:,j]) for i, j in product(range(k), range(k)))
    t3 = sum(np.dot(z[i,:], z.T[:,j]) for i, j in product(range(k), range(k)))
    t4 = sum(np.dot(z.T[i,:], z[:,j]) for i, j in product(range(k), range(k)))

    return (n * np.trace(z) - t2) / (math.sqrt(n**2 - t3) * math.sqrt(n**2 - t4))
