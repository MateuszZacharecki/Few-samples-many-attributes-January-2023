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
The types of points are so mixed that good prediction is not expected.
![image](https://user-images.githubusercontent.com/121680361/226288747-7c42e77c-7c62-4d8a-821c-476617adb51e.png)
![image](https://user-images.githubusercontent.com/121680361/226288848-978b710e-6b73-4e75-b6c4-d71efbf824c1.png)

## Second plot
SVM with linear kernel will probably clasify points of type 'involved' very well but it will also mix points of types 'normal' and 'uninvolved'.
![image](https://user-images.githubusercontent.com/121680361/226292926-15e26b5b-6478-42c7-b983-c8f341c1eca9.png)
![image](https://user-images.githubusercontent.com/121680361/226292963-cfff7b8f-013a-469e-ad6f-70e6d1d21b70.png)

## Third plot
The points of type 'oligodendroglioma' slightly gather in a group as well as points of type 'glioblastoma' but classification of other points might be difficult.
![image](https://user-images.githubusercontent.com/121680361/226294560-41402f94-37a6-44bc-81a6-3276728470db.png)
![image](https://user-images.githubusercontent.com/121680361/226294608-84f4906f-b423-41d8-8e6f-ea41c37ef778.png)

## Fourth plot
The blue points substantially accumulate in upper part of plot, the green ones at the bottom, and the red ones in between, and due to that classification of red points might be especially difficult.
![image](https://user-images.githubusercontent.com/121680361/226295976-7dc7a49f-96b3-4813-87b8-a66e7b7df17c.png)
![image](https://user-images.githubusercontent.com/121680361/226296038-674e19dd-2dcf-4437-95c2-1232a540394f.png)

## Fifth plot
The plot gives the impression that the straight line y=0 might separate data quite well, thus good predictions are expected.
![image](https://user-images.githubusercontent.com/121680361/226297157-7ba3e1a1-b599-47ef-91aa-fa8c44369da5.png)
![image](https://user-images.githubusercontent.com/121680361/226297207-0fdd4eff-6507-43d7-8dbe-e000c8baf141.png)

## Sixth plot
Types of points are significantly mixed. Green and red ones seem to concentrate around point (0,0). Good prediction is not expected.
![image](https://user-images.githubusercontent.com/121680361/226301176-13b4c5b1-d32a-4826-b6f2-24365651744c.png)
![image](https://user-images.githubusercontent.com/121680361/226301225-d912b8bb-93b7-4a47-91fd-664450473d81.png)

## Seventh plot
The red points gather in the left part of plot, purple ones create small group around point (0,75), blue ones accumulate at the bottom, some of them are noticed in other parts of plot, green ones seem to be hard to separate.
![image](https://user-images.githubusercontent.com/121680361/226302197-1ae39ebb-0344-4562-bd7f-73a8335f6162.png)
![image](https://user-images.githubusercontent.com/121680361/226302261-9478a0f0-3ef3-48e6-90ad-4d7e365f4136.png)

## Eighth plot
The significant majority is represented by blue points. The other ones are so scattered that it will be surely difficult to clasify them using SVM.
![image](https://user-images.githubusercontent.com/121680361/226303662-dd6f8915-976f-442b-bc51-4c62993dfc68.png)
![image](https://user-images.githubusercontent.com/121680361/226303709-c66541dd-1295-4963-ac1c-7cdc895ab95b.png)

## Ninth plot
The blue points make up the majority, the red ones slightly accumulate in the left part of plot, the green ones gather insignificantly in the centre of blue points (due to that the model will probably confuse green points as blue). Nevertheless, in view of small number of green points good prediction is expected.
![image](https://user-images.githubusercontent.com/121680361/226306012-60694cac-073e-4064-995a-e5efac55af15.png)
![image](https://user-images.githubusercontent.com/121680361/226306075-e6bbc21b-ae3c-4daa-9876-3cef2d5c0f0e.png)

## Tenth plot
Due to the number of types of data and the lack of gathering of points, good prediction is not expected.
![image](https://user-images.githubusercontent.com/121680361/226306856-d53fbdfe-d793-4186-bc4d-9c68c0693ad0.png)
![image](https://user-images.githubusercontent.com/121680361/226307096-35e7ee1d-d313-4b02-9957-5bdcb2832065.png)

All of these observations are as exact as the quality of data visualizations in reference to real data.

## Solution
The first attempt to this task (written in R) is applying the package 'mlr3filters' to filter about 600-1000 best features in each of 10 sets and then searching the final set of features using mlr3selector package (wrapper method). Generic search is mainly tested in this attempt. However, results of this method aren't safisfying enough.

The second (and final) attempt is quite similar to the first one. At first the features are filtered in each of 10 sets. Their importance is estimated by statistical chi-square test and then best 800 features are chosen (the number established empirically, gives best predictions). Then there is applied a wrapper method, the used heuristics is recursive feature elimination. The importance of remaining features is estimated repeatedly by SVM model and the weakest features are rejected in every step. The procedure is repeated as long as the established number of features isn't reached (finally this number is 80, it gives the best results). Forward and backward search were also applied, but it gave worse results and moreover computing using these methods lasted long time (about 15-20 minutes on each set).
