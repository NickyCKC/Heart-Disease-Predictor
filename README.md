# Predicting Heart Disease Using Machine Learning

SC1015 Lab Group FCSI - Group 1
- Aloysius Lee
- Faybeata
- Nicholas Chang

## Table of Contents
[About this project](#about-this-project)

[Contributions](#contributions)

[Problem Definition](#problem-definition)

[Data Preparation + Exploratory Data Analysis](#data-preparation--exploratory-data-analysis)

[Machine Learning](#machine-learning)

[Conclusion](#conclusion)

[References](#references)

## About this project
In this SC1015 mini-project, we have created machine learning models to predict heart disease based on a [diverse kaggle dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset/data).

To view the code in its' entirety, please view the [jupyter notebook](./project_files/project_notebook.ipynb) here.

The jupyter notebook is organized following the machine learning pipeline, which includes:
1. Data Preparation and Cleaning
2. Exploratory Data Analysis
3. Machine Learning
4. Conclusion

### Packages that were used:
- Pandas
- Seaborn
- NumPy
- Matplotlib
- Sklearn
- Scipy
- Tensorflow

## Contributions
### Aloysius Lee 
- Data Prep, Cleaning, and Exploratory Data Analysis (col. 9-12)
- Logistic Regression

### Faybeata 
- Data Prep, Cleaning, and Exploratory Data Analysis (col. 1-4)
- Decision Tree & Neural Network
- Conclusion

### Nicholas Chang
- Data Prep, Cleaning, and Exploratory Data Analysis (col. 5-8)
- Random Forest

## Problem Definition
- Identify and analyze the key variables that are most likely to contribute to the development of heart disease.
- Find which model is able to predict heart disease best given the dataset.

## Data Preparation + Exploratory Data Analysis
- Cleaned columns for invalid values.
- Explored target column which showed a balanced dataset.
- Explored each independent variable and visualising in the form of various graphs.
- Calculated correlation values for both numerical independent variables (Using Point Biserial Correlation) and categorical independent variables (Using the Chi-test to test for independence between two categorical variables)

### Insights from the data include:
1. Younger individuals were more likely to be diagnosed with heart disease.
2. Weaker than expected correlation between cholesterol and heart disease.
3. Very strong correlation between gender and heart disease.

## Machine Learning

The general pipeline for these models are:
1. Creating x & y (x are all the independent variables, y is the dependent variable; target)
2. Splitting train and test data (stratifying by y)
3. Fitting the model to the train data
4. Predict y by using test data
5. Compare predicted and actual y value
6. Use metrics such as accuracy, precision, recall, and f1 to quantify how the model performs

We used 4 models:
1. [Decision Tree](#decision-tree---max-tree-depth-7)
2. [Random Forest Regression](#random-forest-regression---max-tree-depth-7)
3. [Logistic Regression](#logistic-regression)
4. [Neural Network](#neural-network)

### Decision Tree - Max Tree Depth 7
1. **Accuracy:** 0.9455
2. **Precision:** 0.9338
3. **Recall:** 0.9621
4. **F1 Score:** 0.9478

<img src="./img/Decision%20Tree.png" height="400">

### Random Forest Regression - Max Tree Depth 7
1. **Accuracy:** 0.9650
2. **Precision:** 0.9362
3. **Recall:** 1.0
4. **F1 Score:** 0.9670

<img src="./img/Random Forest.png" height="300">

### Logistic Regression
1. **Accuracy:** 0.8390
2. **Precision:** 0.7857
3. **Recall:** 0.9429
4. **F1 Score:** 0.8571

<img src="./img/Logistic%20Regression.png" height="300">

### Neural Network

We decided to experiment on different amount of hidden layers; 1 - 3. 
- **1 hidden layer:** loss: 0.4904 - accuracy: 0.9400
- **2 hidden layers:** loss: 0.4122 - accuracy: 0.9720
- **3 hidden layers:** loss: 0.3830 - accuracy: 0.9240

As observed, the accuracy decreases from 2 to 3 hidden layers. This may be the result of overfitting the training data. The optimal amount of hidden layers is 2.

<img src="./img/Neural Network (2).png" height="300">

Image is describing the model accuracy of the Neural Network with 2 hidden layers.

## Conclusion
From the accuracy scores, it is clear to see that the Neural Network model with 2 hidden layers provided us with the best prediction with an **accuracy score of 0.9720**.

Some of the most important factors when it comes to predicting heart disease are **thalassemia, chest pain type, and number of major vessels coloured by fluoroscopy**.

Some considerations that we could take into account to improve the model are:
1. Tweak the activation functions for each hidden layer in the neural network
2. Experiment on which variables to include to hopefully optimize the model
3. Change and experiment on model optimizers for the neural network model

## References
- [Point Biserial Correlation](https://datatab.net/tutorial/point-biserial-correlation) 
- [Chi-Square Test](https://www.analyticsvidhya.com/blog/2021/06/decoding-the-chi-square-test-use-along-with-implementation-and-visualization/)
- [Neural Network with TensorFlow](https://www.tensorflow.org/tutorials/quickstart/beginner)
- [Logistic Regression](https://www.w3schools.com/python/python_ml_logistic_regression.asp#:~:text=Logistic%20regression%20aims%20to%20solve,tumor%20is%20malignant%20or%20benign.)
- [Random Forest Regression](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
