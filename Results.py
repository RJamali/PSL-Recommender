import os
import sys
from sklearn.model_selection import  RepeatedKFold
from functions import load_matrices, apply_threshold, compute_evaluation_criteria
from PSLRecommender import PSLR
import numpy as np

print(os.path.dirname(os.path.realpath('__file__')))
dataset= input("Please enter Dataset's name (HumT, Bacello, Hoglund, DBMloc): ") #Datasets: "HumT", "Bacello", "Hoglund", "DBMloc"
if dataset not in {"HumT", "Bacello", "Hoglund", "DBMloc"}:
    print ("Wrong dataset name!!")
    sys.exit()
elif dataset=="DBMloc":
    num_repeats =int(input("Please enter number of 5-fold cross validation repeats: "))


data_folder = os.path.join(os.path.dirname(os.path.realpath('__file__')), 'Datasets') # All dataset's matrices folder

observation_mat, proteins_sim =load_matrices(dataset,data_folder) # load protein localization matrix, proteins similarity matrix and locations similarity matrix
F1, ACC, AVG, ACC2 = 0.0, 0.0, 0.0, 0.0
seed = [80162,45929]
model = PSLR(c=11, K1=4, K2=10, r=10, lambda_p=0.25, lambda_l=0.5, alpha=2.0, theta=1.0, max_iter=50)  # setting model parameters
if dataset!="DBMloc":
    if dataset=="HumT":
        train_index = np.arange(0, 3122)
        test_index = np.arange(3122, 3501)
    elif dataset=="Bacello":
        train_index = np.arange(0, 2595)
        test_index = np.arange(2595, 3170)
    else:
        train_index = np.arange(0, 2682)
        test_index = np.arange(2682, 2840)
    test_location_mat = np.array(observation_mat)
    test_location_mat[train_index] = 0
    train_location_mat = np.array(observation_mat - test_location_mat)
    true_result = np.array(test_location_mat[test_index])

    x = np.repeat(test_index, len(observation_mat[0]))
    y = np.arange(len(observation_mat[0]))
    y = np.tile(y, len(test_index))
    model.fix_model(train_location_mat, train_location_mat, proteins_sim, seed)
    scores = np.reshape(model.predict_scores(zip(x, y)), true_result.shape)
    prediction = apply_threshold(scores, 0.36)
    F1, ACC, AVG, ACC2 = compute_evaluation_criteria(true_result, prediction)

else:
    kf = RepeatedKFold(n_splits=5, n_repeats=num_repeats)
    for train_index, test_index, in kf.split(proteins_sim, observation_mat):
        test_location_mat = np.array(observation_mat)
        test_location_mat[train_index] = 0
        train_location_mat = np.array(observation_mat - test_location_mat)
        true_result = np.array(test_location_mat[test_index])
        x = np.repeat(test_index, len(observation_mat[0]))
        y = np.arange(len(observation_mat[0]))
        y = np.tile(y, len(test_index))
        model.fix_model(train_location_mat, train_location_mat, proteins_sim, seed)
        scores = np.reshape(model.predict_scores(zip(x, y)), true_result.shape)
        prediction = apply_threshold(scores, 0.36)
        fold_f1, fold_acc, fold_avg, fold_acc2  = compute_evaluation_criteria(true_result, prediction)
        F1+=fold_f1
        ACC+=fold_acc
        AVG += fold_avg
        ACC2 += fold_acc2
    F1 = round(F1/(5*num_repeats),2)
    ACC = round(ACC/(5*num_repeats),2)
    AVG = round(AVG / (5 * num_repeats), 2)
    ACC2 = round(ACC2 / (5 * num_repeats), 2)


print ("F1-mean for this dataset:",F1," \nACC for this dataset:" , ACC, "\nAVG for this dataset:" , AVG, "\nACC2 for this dataset:" ,ACC2)
