# Credit-Risk-Detection

## Objective
This notebook explores techniques to improve the accuracy of a machine learning model for credit card fraud detection. The dataset used for this project contains anonymized credit card transactions that occurred in September 2013, with 492 fraudulent transactions out of a total of 284,807 transactions. The dataset is highly unbalanced, with frauds accounting for only 0.172% of all transactions.

## Overview
The objective of this project is to develop a machine learning model that can accurately detect fraudulent credit card transactions. To achieve this, we explore balancing the data using undersampling and oversampling (using SMOTE) techniques, and evaluate their impact on the accuracy of the model.


## Data
The dataset used for this project contains credit card transactions that occurred in September 2013, made by European cardholders which is available on [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The dataset includes 31 features, of which 28 features have been transformed using PCA for confidentiality reasons, and the remaining features are 'Time', 'Amount', and 'Class.

The features considered in the dataset are as follow:

* **28 features:** obtained by PCA transformation (denoted as V1, V2, ... V28)
* **2 non-transformed features**: 'Time' and 'Amount'.
* **Target variable**: Payment variables: 'Class' (1 for fraud, 0 for genuine).

## Methods
The notebook follows the following steps:

* Exploratory Data Analysis (EDA) to understand the distribution of the features and their relationship with the target variable.
* Data Preprocessing to handle missing or outlier data, and scaling the features.
* Model Selection and Hyperparameter Tuning using various classification algorithms such as Logistic Regression, Decision Trees, Gradient Boosting and Naive Bayes.
* Implementing undersampling and oversampling (using SMOTE) techniques to balance the data and improve the model's performance.
* Model Evaluation using AUPRC, F1 Score, and other relevant metrics.
* Implementation of undersampling and oversampling on Neural Networks.
* EModel Evaluation using AUPRC, F1 Score, and other relevant metrics.

## Results
The notebook concludes that implementing SMOTE on our imbalanced dataset helped balance the labels of our data. However, our oversampled model sometimes predicts fewer correct fraud transactions than our undersampled model. We should keep in mind that our undersampled model is unable to detect non-fraud transactions correctly, leading to misclassification of non-fraud transactions as fraud transactions. This can cause a lot of customer complaints and dissatisfaction, which is a disadvantage for the financial institution. Our next step will be to perform outlier removal on our oversampled dataset and check if it improves the accuracy of our model on the test set.

## Next Steps
The following steps will be taken to further improve the performance of the model:

* Perform outlier removal on our oversampled dataset using the Interquartile Range (IQR) method.
* Retrain and evaluate our oversampled model on the cleaned dataset.
* Compare the performance of our new model with the previous models to see if outlier removal improves the accuracy of our model.
* Explore other techniques such as feature engineering, hyperparameter tuning, and model ensemble to further improve the performance of our model.
