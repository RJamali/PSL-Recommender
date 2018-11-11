# PSL-Recommender - Protein Subcellular Localization Prediction using Recommender System

PSL-Recommender is a python package for prediction of proteins subcellular locations.
PSL-Recommender uses logistic matrix factorization to build a recommender system for protein subcellular localization problem.
This package is based on previous implementation of logistic matrix factorization by [Logistic matrix factorization for implicit feedback data](https://github.com/MrChrisJohnson/logistic-mf) and [Neighborhood regularized logistic matrix factorization for drug-target interaction prediction](https://github.com/stephenliu0423/PyDTI).

### Prerequisites

PSL-Recommender is implemented in python 2.7.12 and requiers [Numpy](http://www.numpy.org/) package.

### Using PSL-Recommender
First step is to define a model (PSLR predictor) with parameters like the following example:
```
model = PSLR(c=46, K1=54, K2=3, r=6, lambda_p=0.5, lambda_l=0.5, alpha=0.5, theta=0.5, max_iter=50)
```
All model parameters are explained in the paper but here is a breif description of parameters:

 * c:  weighting factor for positive observations
 * K1: number of nearest neighbors used for latent matrix construction
 * K2: number of nearest neighbors used for score prediction
 * r: dimention of the latent space
 * theta: Gradient descent learning rate
 * lambda_p: variance controlling parameter for proteins
 * lambda_l: : variance controlling parameter for subcellular locations
 * alpha: impact factor of nearest neighbors used for neighborhood regularization
 * max_iter: maximum number of iteration for gradient descent
        
The model must be trained on the training dataset like the following example:
```
model.fix_model(train_interaction, train_interaction, proteins_features, seed)
```
Where "train_interaction" is the binary matrix of the protein-subcellular location observations, "proteins_features" is the similarity matrix for training proteins, and 'seed' is the seed for random number generator used in gradient descent. For producing the same result seed should be fixed number.

Finally, model.predic_scores estimeates the probability of residing test proteins in subcellular locations.

## Running on benchmark datasets
In Results.py, it is possible to test PSL-Recommender on four well-known benchmark datasets ([Hum-mPLoc 3.0](https://academic.oup.com/bioinformatics/article/33/6/843/2623045), [BaCelLo](https://academic.oup.com/bioinformatics/article/22/14/e408/228072), [Höglund](https://academic.oup.com/bioinformatics/article/22/10/1158/236546), and [DBMLoc](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-9-127) ). 

For running Results.py [scikit-learn](http://scikit-learn.org/stable/) package is needed.

Datasets for runnung Results.py are available at: https://drive.google.com/open?id=1ied6kbSF9PByGoGHVUZkW7hyd81oW5nj

## Authors
**Ruhollah Jamali**
Email: ruhi.jamali(at sign)gmail.com

School of Biological Sciences, Institute for Research in Fundamental Sciences(IPM), Tehran, Iran

If you have any problem running the code, please do not hesitate to contact me.

