from __future__ import division
import os
import sys
from sklearn.model_selection import  RepeatedKFold
from functions import load_matrices, extract_names, apply_threshold, compute_evaluation_criteria
from PSLRecommender import PSLR
import numpy as np


dataset= raw_input("Please enter Dataset's name (HumT, Bacello, Hoglund, DBMloc): ") #Datasets: "HumT", "Bacello", "Hoglund", "DBMloc"
if dataset not in {"HumT", "Bacello", "Hoglund", "DBMloc"}:
    print "Wrong dataset name!!"
    sys.exit()
elif dataset=="DBMloc":
    num_repeats =int(raw_input("Please enter number of 5-fold cross validation repeats: "))


data_folder = os.path.join(os.path.pardir, 'Datasets') # All dataset's matrices folder

observation_mat, proteins_sim =load_matrices(dataset,data_folder) # load protein localization matrix, proteins similarity matrix and locations similarity matrix
location_names, proteins_names = extract_names(dataset, data_folder)    # extract proteins names and location names

seed = [80162,45929]
if dataset!="DBMloc":
    if dataset=="HumT":
        model =  PSLR(c=43, K1=16, K2=10, r=10, lambda_p=1.0, lambda_l=2.0, alpha=1.0, theta=1.0, max_iter=50) #setting model parameters
        train_index = np.arange(0, 3122)
        test_index = np.arange(3122, 3501)
        test_location_mat = np.array(observation_mat)
        test_location_mat[train_index] = 0
        train_location_mat = np.array(observation_mat - test_location_mat)
        true_result = np.array(test_location_mat[test_index])
    elif dataset=="Bacello":
        model =  PSLR(c=17, K1=41, K2=3, r=4, lambda_p=0.25, lambda_l=0.25, alpha=0.5,  theta=1.0, max_iter=50) #setting model parameters
        train_index = np.arange(0, 2595)
        test_index = np.arange(2595, 3170)
        test_location_mat = np.array(observation_mat)
        test_location_mat[train_index] = 0
        train_location_mat = np.array(observation_mat - test_location_mat)
        true_result = np.array(test_location_mat[test_index])
    else:
        model = PSLR(c=46, K1=54, K2=3, r=6, lambda_p=0.5, lambda_l=0.5, alpha=0.5, theta=0.5, max_iter=50) #setting model parameters
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
    prediction = apply_threshold(scores, dataset, 0.36)
    F1, ACC, location_F1 = compute_evaluation_criteria(true_result, prediction)

else:
    model = PSLR(c=8, K1=13, K2=5, r=6, lambda_p=0.5, lambda_l=0.0625, alpha=1.0, theta=1.0, max_iter=50) #setting model parameters
    kf = RepeatedKFold(n_splits=5, n_repeats=num_repeats)
    F1, ACC = 0.0, 0.0
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
        prediction = apply_threshold(scores, dataset, 0.37)
        fold_f1, fold_acc, fold_loc_f1 = compute_evaluation_criteria(true_result, prediction)
        F1+=fold_f1
        ACC+=fold_acc
    F1=round(F1/(5*num_repeats),2)
    ACC=round(ACC/(5*num_repeats),2)


print "F1-mean for this dataset:",F1,"  ACC for this dataset:" ,ACC

