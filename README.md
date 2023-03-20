# Few-samples-many-attributes-January-2023
## Overview
The goal is to find informative subsets of genes that allow to efficiently solve classification problems defined for a number of microarray data sets.

## Task description
The provided data consist of ten microarray sets with a various number of instances and attributes. Microarray data is a typical example of a problem called "few-samples-many-attributes".

The data tables are provided as CSV files with the ',' (coma) separator sign. In each set, the last column is called "target" and contains class labels for samples. There's only an access to the training parts of the data sets. The task is to (for each set) identify the optimal subset of attributes for an SVM classifier with a linear kernel and the cost parameter set to 1. No additional regularization will be used for the model. The less indicated attributes the better, the expected number of attributes is between 2 and 102.

The evaluation metric will be balanced accuracy (BAC).

## Data visualization
All 10 sets were visualizated using PCA method. They were saved in one list.

## PCA
The visualization function that uses PCA shows not only scatter plot of data but also diagram of standard deviation of eigenvectors to chart the accuracy of data plots. Unfortunately, it turns out that in each case first two eigenvectors don't dominate enough to trust plots unconditionally.

## First plot
### Data visualization
![image](https://user-images.githubusercontent.com/121680361/226287083-cb88b65b-a48e-42cc-a501-75b93e8aa3e1.png)
