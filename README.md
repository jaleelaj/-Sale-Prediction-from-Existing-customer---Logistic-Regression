# -Sale-Prediction-from-Existing-customer---Logistic-Regression
# Sale Prediction from Existing Customer - Logistic Regression

![Sale Prediction](sale_prediction.jpg)

This repository provides a simple implementation of sale prediction using logistic regression. Logistic regression is a commonly used statistical method for binary classification, and it can be applied to various scenarios, including predicting whether an existing customer will make a purchase.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Training the Model](#training-the-model)
- [Making Predictions](#making-predictions)
- [Evaluating the Model](#evaluating-the-model)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Sale prediction from existing customers is a valuable task for businesses, as it helps them understand customer behavior and make informed marketing decisions. In this project, we use logistic regression to predict whether an existing customer will make a purchase based on their age and salary.

## Dataset

The dataset used for this project is stored in a CSV file named `ad_dataset.csv`. This dataset contains information about existing customers, including their age and salary, as well as whether they made a purchase (1 for "Yes" and 0 for "No").

## Usage

1. Import the required libraries:

   ```python
   import pandas as pd # Useful for loading the dataset
   import numpy as np  # To perform array operations
   ```

2. Choose the dataset file from your local directory, if you are using Google Colab:

   ```python
   from google.colab import files
   uploaded = files.upload()
   ```

3. Load the dataset:

   ```python
   dataset = pd.read_csv('ad_dataset.csv')
   ```

4. Segregate the dataset into input (X) and output (Y) variables:

   ```python
   X = dataset.iloc[:, :-1].values
   Y = dataset.iloc[:, -1].values
   ```

5. Split the dataset into training and testing sets:

   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
   ```

## Data Preprocessing

6. Perform feature scaling to ensure that all features contribute equally to the result:

   ```python
   from sklearn.preprocessing import StandardScaler
   sc = StandardScaler()
   X_train = sc.fit_transform(X_train)
   X_test = sc.transform(X_test)
   ```

## Training the Model

7. Train a logistic regression model using the training data:

   ```python
   from sklearn.linear_model import LogisticRegression
   model = LogisticRegression(random_state=0)
   model.fit(X_train, y_train)
   ```

## Making Predictions

8. Predict whether a new customer with age and salary will make a purchase:

   ```python
   age = int(input("Enter New Customer Age: "))
   sal = int(input("Enter New Customer Salary: "))
   newCust = [[age, sal]]
   result = model.predict(sc.transform(newCust))
   print(result)
   if result == 1:
     print("Customer will Buy")
   else:
     print("Customer won't Buy")
   ```

## Evaluating the Model

9. Evaluate the model's performance using a confusion matrix and accuracy score:

   ```python
   from sklearn.metrics import confusion_matrix, accuracy_score
   y_pred = model.predict(X_test)
   cm = confusion_matrix(y_test, y_pred)

   print("Confusion Matrix: ")
   print(cm)

   print("Accuracy of the Model: {0}%".format(accuracy_score(y_test, y_pred) * 100))
   ```

## Contributing

Contributions are welcome! If you find any issues or have ideas for improvements, feel free to create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Please customize this README according to your project's specifics. Good luck with your sale prediction using logistic regression project!
