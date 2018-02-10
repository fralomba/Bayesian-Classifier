# Genes Classification

## Introduction
This project aims to predict through Bernoulli Naive Bayes algorithm the essentiality of a gene for some bacterium's life.

## Run
1. Download the [repository](https://github.com/fralomba/Bayesian-Classifier.git)

2. Run main.py

3. The script will output *S.Mikatae* dataset classification results and plot the ROC curve of the prediction about *S.Cerevisiae* dataset.

## Implementation
The datasets have been read with [*Pandas*](https://pandas.pydata.org/) and then converted in [*Numpy*](http://www.numpy.org) arrays. After they have been discretized and one-hot-encoded.
Some features in *S.Mikatae* contained unknown values, that have been changed to '0'.

**10-fold cross validation** has been used to evaluate the classifier's accuracy.

## Results
Both classification and ROC curve computation seem to be consistent with the compared [reference](http://genome.cshlp.org/content/16/9/1126).

![ROC](/img/roc.png)

## Requirements
| Software                                                 | Version         | Required |
| ---------------------------------------------------------|-----------------| ---------|
| **Python**                                               |     >= 3        |    Yes   |
| **Numpy** (Python Package)                               |Tested on v1.13.3|    Yes   |
| **Scikit-learn** (Python Package)                        |Tested on v0.19.1|    Yes   |
| **Pandas** (Python Package)                              |Tested on v0.21.1|    Yes   |
| **Matplotlib**			                               |Tested on v2.1.1 |    Yes   |