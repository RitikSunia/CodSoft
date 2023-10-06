# CodSoft
Data Science Projects for Codsoft Internship

# Titanic Survival Prediction Model
# Aim 
The aim of this project is to build a model that predicts whether a passenger on the Titanic survived or not.
# Data 
The dataset for this project is imported from a CSV file. The dataset contains information about individual passengers, such as their age, gender, ticket class, fare, cabin, and whether or not they survived.
# Libraries used
The following important libraries were used for this project:

- numpy
- pandas
- seaborn
- matplotlib.pyplot
- sklearn.preprocessing.LabelEncoder
- sklearn.model_selection.train_test_split
- sklearn.linear_model.LogisticRegression

# Model Training
 - The feature matrix **a** and target vector **b** were created using relevant columns from the DataFrame.
 - The dataset was split into training and testing sets using train_test_split from sklearn.model_selection.
 - A logistic regression model was initialized and trained on the training data using LogisticRegression from sklearn.linear_model.

# Prediction
 - The model was used to predict the survival status of passengers in the test set.
 - The predicted results were printed using loRe.predict(X_test).
 - The actual target values in the test set were printed using Y_test.

# Important
This model was developed in PyCharm Community Edition and then copied to Jupyter Notebook
