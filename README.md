# credit-risk-classification
# Credit Risk Classification Challenge

## Overview
In this challenge, we used various techniques to train and evaluate a machine learning model for assessing loan risk. The goal was to build a model that can predict the creditworthiness of borrowers using historical lending activity data from a peer-to-peer lending services company.

## Repository Structure
- The project is organized into a GitHub repository named `credit-risk-classification`.
- Inside the repository, there is a folder titled `Credit_Risk` where you can find the `credit_risk_classification.ipynb` notebook and the `lending_data.csv` dataset.

## Steps

### 1. Split the Data into Training and Testing Sets
- We started by reading the dataset from the `lending_data.csv` file into a Pandas DataFrame.
- We created the labels set (`y`) from the "loan_status" column and the features (`X`) DataFrame from the remaining columns.
- We checked the balance of the labels variable (`y`) using the `value_counts` function.
- Finally, we split the data into training and testing datasets using the `train_test_split` function.

### 2. Create a Logistic Regression Model with the Original Data
- In this step, we fitted a logistic regression model using the training data (`X_train` and `y_train`).
- We saved the model's predictions on the testing data labels (`X_test`).
- We evaluated the model's performance by generating a confusion matrix and printing the classification report.

### 3. Predict a Logistic Regression Model with Resampled Training Data
- We used the `RandomOverSampler` module from the imbalanced-learn library to resample the data to address class imbalance.
- After resampling, we fitted a logistic regression model using the resampled training data and made predictions on the testing data.
- Again, we evaluated the model's performance by calculating the accuracy score, generating a confusion matrix, and printing the classification report.

## Results

### Logistic Regression Model with Original Data
- Accuracy Score: 0.9520479254722232
- Confusion Matrix: [18663   102]
 [   56   563]]
- Classification Report:     precision    recall  f1-score   support

           0       1.00      0.99      1.00     18765
           1       0.85      0.91      0.88       619

    accuracy                           0.99     19384
   macro avg       0.92      0.95      0.94     19384
weighted avg       0.99      0.99      0.99     19384


### Logistic Regression Model with Resampled Data
- Accuracy Score: 0.9936781215845847
- Confusion Matrix: [[18649   116]
 [    4   615]]
- Classification Report:  precision    recall  f1-score   support
...
   macro avg       0.92      0.99      0.95     19384
weighted avg       0.99      0.99      0.99     19384

## Summary and Recommendation
- In this analysis, we explored the credit risk classification problem using logistic regression models.
- The logistic regression model with resampled training data shows improved performance in handling class imbalance compared to the model with original data.
- We recommend using the logistic regression model with resampled data for credit risk assessment, as it provides better accuracy and precision in identifying high-risk loans.


