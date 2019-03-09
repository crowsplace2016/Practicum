#Getting the data ready - Data Preparation
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset_heart = pd.read_csv('heart.csv')

#Check for null values in data
dataset_heart.isnull().sum()

X1 = dataset_heart.iloc[:, 0:13].values #Independent variable
y1 = dataset_heart.iloc[:, 13].values #Dependent variable

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train1 = sc.fit_transform(X_train1)
X_test1 = sc.transform(X_test1)

#The following charts are based on Kaggle Kernels "What causes heart disease?" by Nitin Datta
dataset_heart['target'].value_counts()

dataset_heart.head()

#Creating the ANN
#import Machine Learning library Keras
import keras
from keras.models import Sequential  #Sequence of layers
from keras.layers import Dense  #initial weight to 0

#Build the ANN
ANN = Sequential()

#Add Input layer and Hidden layer
ANN.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 13))

#Output Layer
ANN.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

#Compile the ANN
ANN.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Now we need to fit the model
ANN.fit(X_train1, y_train1, batch_size = 1, nb_epoch = 50)


# Predicting the Test set results
y_pred = ANN.predict(X_test1)

#Change the output from decimal to True or False.  Anything greater that .5 is True
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test1, y_pred)

#A True outcome
new_prediction = ANN.predict(sc.transform(np.array([[49,1,1,130,266,0,1,171,0,0.6,2,0,2]])))
new_prediction = (new_prediction > 0.5)

#A False outcome
new_prediction1 = ANN.predict(sc.transform(np.array([[45.0,1,0,142,309,0,0,147,1,0,1,3,3]])))
new_prediction1 = (new_prediction1 > 0.5)


sns.distplot(dataset_heart['age'],color='Green')

fig,ax=plt.subplots(figsize=(24,6))
plt.subplot(1, 3, 1)
age_bins = [20,30,40,50,60,70,80]
dataset_heart['Age']=pd.cut(dataset_heart['age'], bins=age_bins)
g1=sns.countplot(x='Age',data=dataset_heart ,hue='target',palette='hot')
g1.set_title("Age vs Heart Disease")
#The number of people with heart disease are more from the age 41-55
#Also most of the people fear heart disease and go for a checkup from age 55-65 and dont have heart disease (Precautions)

fig,ax=plt.subplots(figsize=(24,6))
plt.subplot(1, 3, 2)
cho_bins = [100,150,200,250,300,350,400,450]
dataset_heart['Cholesterol']=pd.cut(dataset_heart['chol'], bins=cho_bins)
g2=sns.countplot(x='Cholesterol',data=dataset_heart,hue='target',palette='hot')
g2.set_title("Cholestoral vs Heart Disease")
#Most people get the heart disease with 200-250 cholestrol 
#The others with cholestrol of above 250 tend to think they have heart disease but the rate of heart disease falls

fig,ax=plt.subplots(figsize=(24,6))
plt.subplot(1, 3, 3)
thal_bins = [60,80,100,120,140,160,180,200,220]
dataset_heart['bin_thal']=pd.cut(dataset_heart['thalach'], bins=thal_bins)
g3=sns.countplot(x='bin_thal',data=dataset_heart,hue='target',palette='hot')
g3.set_title("Thal vs Heart Disease")

fig,ax=plt.subplots(figsize=(16,6))
plt.subplot(121)
s1=sns.boxenplot(x='sex',y='age',hue='target',data=dataset_heart,palette='hot')
s1.set_title("Figure 1")
#Figure 1 says most of females having heart disease range from 40-70yrs and men from 40-60yrs

fig,ax=plt.subplots(figsize=(24,6))
plt.subplot(131)
x1=sns.countplot(x='cp',data=dataset_heart,hue='target',palette='hot')
x1.set_title('Chest pain type')
#Chest pain type 2 people have highest chance of heart disease

fig,ax=plt.subplots(figsize=(24,6))
plt.subplot(132)
x2=sns.countplot(x='thal',data=dataset_heart,hue='target',palette='hot')
x2.set_title('Thal')
#People with thal 2 have the highest chance of heart disease

fig,ax=plt.subplots(figsize=(24,6))
plt.subplot(133)
x3=sns.countplot(x='slope',data=dataset_heart,hue='target',palette='hot')
x3.set_title('slope of the peak exercise ST segment')
#Slope 2 people have higher chance of heart disease

