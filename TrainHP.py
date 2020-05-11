import os
import sys
from sklearn.model_selection import  RepeatedKFold
from functions import load_matrices, apply_threshold, compute_evaluation_criteria
from PSLRecommender import PSLR
import numpy as np
from multiprocessing import Pool
import time
from multiprocessing import Process, Queue, current_process, freeze_support


def calculate(func, args):
    result = func(*args)
    return '%s says that %s%s = %s' % (current_process().name, func.__name__, args, result)

def calculatestar(args):
    return calculate(*args)


dataset = "HumT"
data_folder = os.path.join(os.path.dirname(os.path.realpath('__file__')), 'Datasets') # All dataset's matrices folder
observation_mat, proteins_sim =load_matrices(dataset,data_folder) # load protein localization matrix, proteins similarity matrix and locations similarity matrix
proteins_sim = proteins_sim[:3122,:3122]
observation_mat = observation_mat[:3122]
seed = [80162,45929]

Hyper_params = {'r': np.arange(1,13), 'c': np.arange(2, 31), 'K1': np.arange(1,31),
                'K2': np.arange(1,31), 'lambda_p': np.arange(-5,2), 'lambda_l': np.arange(-5,2),
                 'alpha': np.arange(-5,3), 'theta': np.arange(-5,1) }

Nums = {'r': 0, 'c': 1, 'K1': 2, 'K2': 3,'lambda_p': 4, 'lambda_l': 5, 'alpha': 6, 'theta': 7 }

def _compute(job):
    start = time.time()
    model = PSLR(r=job[0], c=job[1], K1=job[2], K2=job[3], lambda_p = 2.0**job[4], lambda_l = 2.0**job[5], alpha=2.0**job[6], theta=2.0**job[7], max_iter=50)  # setting model parameters
    F1, ACC, AVG, ACC2 = 0.0, 0.0, 0.0, 0.0

    kf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=1)
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
    end = time.time()
    return [np.round(F1/25, 5), np.round(ACC/25, 5), np.round(AVG/25, 5), np.round(ACC2/25, 5), job[0], job[1], job[2], job[3], job[4], job[5], job[6], job[7]]

def create_jobs (param, HP, Hyper_params=Hyper_params):
    j = []
    temp = HP
    for val in Hyper_params[param]:
        temp[Nums[param]] = val
        j.append(np.array(temp))
    return np.array(j)

if __name__ == "__main__":
    epoch = 0
    np.random.seed(101)
    HP = [np.random.randint(1,12,1)[0], np.random.randint(1,31,1)[0], np.random.randint(1,31,1)[0], np.random.randint(1,31,1)[0],
        np.random.randint(-2,2,1)[0], np.random.randint(-2,2,1)[0], np.random.randint(-2,2,1)[0], np.random.randint(-2,1,1)[0]]

    print ('Start')
    Best = np.array(HP)
    while True:
        for param in Hyper_params:
            jobs = create_jobs(param, Best)
            print ("Tuning " + param + ":",'\t',len(jobs), " values considered")
            p = Pool(processes = 8)
            result=p.map(_compute , jobs)
            result=np.array(result,dtype=np.float32)
            res=result[(-result[:,0]).argsort(kind='mergesort')]

            Best = res[0][4:]
            if epoch > 1:
                if (np.array(Best) == np.array(HP)).all():
                    break
            p.close()
        print('EPOCH params: ',Best, HP, '\n', 'Best F1: ', res[0][0])
        if (np.array(Best) == np.array(HP)).all():
            break
        else:
            HP = np.array(Best)
        epoch +=1
    print('Best params: ', Best, '\n', 'Best F1: ', res[0][0])
    print ('end')
