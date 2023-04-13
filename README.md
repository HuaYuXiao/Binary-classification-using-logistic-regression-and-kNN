# Binary-classification-problem-using-logistic-regression-and-kNN
SUSTech EE271 Artificial Intelligence and Machine Learning 

## Data set description: 

There are two data sets named “origin_breast_cancer_data.csv” and “breast_cancer_data_357B_100M.csv”.
The data set “origin_breast_cancer_data.csv” contains 357 benign and 212 malignant samples.
The data set “breast_cancer_data_357B_100M.csv” contains 357 benign and 100 malignant samples, which is unbalanced for positive and negative samples. 

Both data sets have 32 columns, starting from 1) to 23), which are explained below.
1. ID number (You can ignore it)
2. Diagnosis (M = malignant, B = benign) 

Attribute Information: 

Ten real-valued features are computed for each cell nucleus.

1. radius (mean of distances from center to points on the perimeter)
2. texture (standard deviation of gray-scale values)
3. perimeter
4. area
5. smoothness (local variation in radius lengths)
6. compactness (perimeter^2 / area - 1.0)
7. concavity (severity of concave portions of the contour)
8. concave points (number of concave portions of the contour)
9. symmetry
10. fractal dimension ("coastline approximation" - 1)

The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius. All feature values are recorded with four significant digits.

## Project description
1. Use the logistic regression method and kNN method for the binary classification problem (to predict whether it is malignant or benign) with the data set “origin_breast_cancer_data.csv”. Try your best to pre-process the data and tune possible hyper-parameters, to get your best binary classification results for each method. Show the metrics (recall, precision, and F1 score) for both training set and validation set. Compare these two methods and discuss for the conclusions that you have.   
2. Considering the unbalanced sample data set “breast_cancer_data_357B_100M.csv”, try the two methods with a new training process again to see what will happen. If the performance is degraded, then try some modification (your input and your creative ideas) for these two methods to see if you can have some improvement. Detail your ideas and show the performance you obtained. 
