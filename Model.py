#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Titnic Survival Prediction Model - This model aims to predict if a passenger will survive on the ship

#Importing the necessary Libraries required.
import pandas as pd
import numpy as num
import seaborn as sea
import matplotlib.pyplot as mat
from pandas import DataFrame


#After successfully importing the libraries , its time to load the Data.
TitanicDATA: DataFrame=pd.read_csv('D:/Internships/Codsoft/task 1/tested.csv')
print(TitanicDATA)


# In[3]:


#Checking the data
print(TitanicDATA.describe()) #it checks the statistics of the data
print(TitanicDATA.shape) #Counts the number of rows and columns


# In[4]:


#Now counting the number of Survivors and visualizing it
print(TitanicDATA['Survived'].value_counts())
print(sea.countplot(x='Survived', data= TitanicDATA))
#this visualization show the number of people survived out of total passengers is less as compared to the passenger
#who did not survive


# In[5]:


#Now, we compare the number of males and females survived
print(sea.countplot(x='Survived', hue='Sex',data=TitanicDATA))
print(TitanicDATA.groupby('Sex')['Survived'].mean()) #(this is the survival rate)
#From this analysis we found out that more females are likely to survive as compared to males


# In[6]:


#Now checking the survival rate according to the Passenger class
print(sea.countplot(x='Survived', hue='Pclass',data=TitanicDATA, palette='pastel'))
print(TitanicDATA.pivot_table('Survived', index='Sex', columns='Pclass').plot()) #this shows the survival rate by Sex and class
#From this plot, we found that Ist class passenger are more likely to survive as compared to IInd or IIIrd class,
#and the majority of passengers who did not survive belongs to IIIrd class


# In[7]:


#Age group Analysis
print(TitanicDATA['Age'].plot.hist())
#This histogram plot shows that young people (20-40) are travelling in highest number as compared to elder people (60-70)


# In[8]:


#Checking how many passengers are travelling with their siblings
print(sea.countplot(x='SibSp', data=TitanicDATA))
#this plot implies that most of the passenger are not travelling with their siblings


# In[9]:


#DATA CLEANING
x = TitanicDATA.isnull().sum() #this shows all the null values
print(x)


# In[10]:


#dropping the unnecessary columns and removing all the null values
TitanicDATA.drop(['Cabin', 'Name', 'PassengerId','Ticket'],axis=1, inplace=True)
TitanicDATA.dropna(inplace=True) #(Replaces all the Null values)
print(TitanicDATA)


# In[11]:


#checking data types
print(TitanicDATA.dtypes)


# In[12]:


#converting the data type
print(TitanicDATA['Sex'].unique())

from sklearn.preprocessing import LabelEncoder
ln = LabelEncoder()

TitanicDATA['Sex'] = ln.fit_transform(TitanicDATA['Sex'])
sex = TitanicDATA['Sex']

TitanicDATA['Embarked'] = ln.fit_transform(TitanicDATA['Embarked'])
embark = TitanicDATA['Embarked']

TitanicDATA['Pclass'] = ln.fit_transform(TitanicDATA['Pclass'])
PCL = TitanicDATA['Pclass']


# In[13]:


#creating new data set
titanic = pd.concat([TitanicDATA, sex, embark, PCL], axis=1)


# In[14]:


#dropping columns
titanic.drop(['Pclass', 'Sex','Embarked'],axis=1,inplace=True)
print(titanic)
print(titanic.dtypes)


# In[15]:


#TRAINING THE MODEL
a = titanic.drop('Survived', axis=1)
b = titanic['Survived']
from sklearn.model_selection import train_test_split
#splitting the data into training and testing
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.2, random_state=0)


# In[16]:


#machine learning model
from sklearn.linear_model import LogisticRegression
loRe = LogisticRegression(random_state=0)
loRe.fit(a_train,b_train)
prediction = loRe.predict(a_test)


# In[17]:


#MODEL PREDICTION
print(prediction)


# In[18]:


print(b_test)


# In[ ]:




