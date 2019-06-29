import os
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score


def load_matrices(dataset, folder):
    with open(os.path.join(folder, dataset+"_protein_locals.tab"), "r") as raw:
        next(raw)
        observation_mat = [line.strip("\n").split()[1:] for line in raw]

    with open(os.path.join(folder, dataset+"_pssm_sim.tab"), "r") as raw:
        next(raw)
        proteins_pssm_sim = [line.strip("\n").split()[1:] for line in raw]

    with open(os.path.join(folder, dataset+"_bp_sim.tab"), "r") as raw:
        next(raw)
        proteins_bp_sim = [line.strip("\n").split()[1:] for line in raw]

    with open(os.path.join(folder, dataset+"_cc_sim.tab"), "r") as raw:
        next(raw)
        proteins_cc_sim = [line.strip("\n").split()[1:] for line in raw]

    with open(os.path.join(folder, dataset+"_mf_sim.tab"), "r") as raw:
        next(raw)
        proteins_mf_sim = [line.strip("\n").split()[1:] for line in raw]


    observation_mat = np.array(observation_mat, dtype=np.float64)    # protein subcellular localization observation matrix
    proteins_pssm_sim = np.array(proteins_pssm_sim, dtype=np.float64)      # proteins pssm similarity matrix
    proteins_bp_sim = np.array(proteins_bp_sim, dtype=np.float64)  # proteins bp terms similarity matrix
    proteins_cc_sim = np.array(proteins_cc_sim, dtype=np.float64)  # proteins cc terms similarity matrix
    proteins_mf_sim = np.array(proteins_mf_sim, dtype=np.float64)  # proteins mf terms similarity matrix

    if dataset=="HumT":
        with open(os.path.join(folder, dataset+"_string_sim.tab"), "r") as raw:
            next(raw)
            proteins_string_sim = [line.strip("\n").split()[1:] for line in raw]
        proteins_string_sim = np.array(proteins_string_sim, dtype=np.float64)  # proteins string similarity matrix
        proteins_sim=  ((1 * proteins_pssm_sim) + (1 * proteins_string_sim) + (3 * proteins_bp_sim) + (4 * proteins_cc_sim) + (1 * proteins_mf_sim)) / (10) #compute weighted average of similarities
    else :
        proteins_sim=  ((1 * proteins_pssm_sim)  + (3 * proteins_bp_sim) + (4 * proteins_cc_sim) + (1 * proteins_mf_sim)) / (9) #compute weighted average of similarities
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
    for i in range(true_result.shape[1]):
        tn, fp, fn, tp = confusion_matrix(true_result[:, i], prediction[:, i]).ravel()
        AVG += tp / ((tp + fn) * (true_result.shape[1]))
        ACC2 += (tp + tn) / (true_result.shape[0] * true_result.shape[1])
    for i in range(true_result.shape[0]):
        tn, fp, fn, tp = confusion_matrix(true_result[i], prediction[i]).ravel()
        ACC += (tp / (tp + fp + fn))
        pres[i] = precision_score(true_result[i], prediction[i])
        recs[i] = recall_score(true_result[i], prediction[i])
    ACC = ACC / (len(true_result))

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
