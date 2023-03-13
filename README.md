# Few-samples-many-attributes-January-2023
## Overview
The goal is to find informative subsets of genes that allow to efficiently solve classification problems defined for a number of microarray data sets.

## Task description
The provided data consist of ten microarray sets with a various number of instances and attributes. Microarray data is a typical example of a problem called "few-samples-many-attributes".

The data tables are provided as CSV files with the ',' (coma) separator sign. In each set, the last column is called "target" and contains class labels for samples. There's only an access to the training parts of the data sets. The task is to (for each set) identify the optimal subset of attributes for an SVM classifier with a linear kernel and the cost parameter set to 1. No additional regularization will be used for the model. The less indicated attributes the better, the expected number of attributes is between 2 and 102.

The evaluation metric will be balanced accuracy (BAC).
