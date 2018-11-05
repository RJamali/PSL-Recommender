# PSL-Recommender -- a protein subcellular location predictor based on logistic matrix factorization

PSL-Recommender is a python package for prediction of proteins subcellular locations 

### Prerequisites

PSL-Recommender is implemented by python 2.7.12 and it requiers [Numpy](http://www.numpy.org/) package.

### Using PSL-Recommender
First step is to define a model (PSLR predictor) with parameters like following example:
```
model = PSLR(c=46, K1=54, K2=3, r=6, lambda_p=0.5, lambda_l=0.5, alpha=0.5, theta=0.5, max_iter=50)
```
All model parameters explained in the paper but here is the summary of definition of them:

 * c:  importance level for positive observations
 * K1: number of nearest neighbors used for latent matrix construction
 * K2: number of nearest neighbors used for score prediction
 * r: latent matrices's dimension
 * theta: Gradien descent learning rate
 * lambda_p: reciprocal of proteins's variance
 * lambda_l: reciprocal of subcellulars's variance
 * alpha: impact factor of nearest neighbor in constructing predictor
 * max_iter: maximum number of iteration for gradient descent
        
After that model should be trained on the training dataset like following example:
```
model.fix_model(train_interaction, train_interaction, proteins_features, seed)
```
which "train_interaction" is  binary matrix of the protein subcellular locations obsevation with zero valuee for all test protein's subcellular location. "proteins_features" is a matrix of features for same proteins in "train_interaction". One step of our method is gradien descent, so we used fixed random seeds for replicating the results.

Finally, model.predic_scores could estimeate probability of residing test proteins in subcellular locations.

## Running the tests
In the Results.py, It is possible to test PSL-Recommender on the four of the well-known datasets ([Hum-mPLoc 3.0](https://academic.oup.com/bioinformatics/article/33/6/843/2623045), [BaCelLo](https://academic.oup.com/bioinformatics/article/22/14/e408/228072), [Höglund](https://academic.oup.com/bioinformatics/article/22/10/1158/236546), and [DBMLoc](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-9-127) ) with available protein-protein similaritie which is also uploade in this repository. This code could take name of one of the mentioned datasets and report F1-mean and ACC of the PSL-Recommender on that dataset. (All used parameters in Result.py are learned paractically for each dataset)

For running Results.py [scikit-learn](http://scikit-learn.org/stable/) package is needed.

## Authors
**Ruhollah Jamali**
Email: ruhi.jamali@gmail.com
School of Biological Sciences, Institute for Research in Fundamental Sciences(IPM), Tehran, Iran
Do not hesitate to ask any question about the repository.

