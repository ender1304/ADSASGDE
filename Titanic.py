
# coding: utf-8

# In[2]:

#standard modules for *.csv in python
import pandas as pd
import numpy as np


# In[4]:

#read local files
data_train=pd.read_csv('c:\\kaggle\\Titanic CSV\\train.csv')
data_test=pd.read_csv('c:\\kaggle\\Titanic CSV\\test.csv')


# In[5]:

data_train.info()
data_test.info()
data_train.head()


# In[6]:

#create a plot of class versus chance of survival
import matplotlib as plt
get_ipython().magic('matplotlib inline')
data_train.groupby('Pclass').Survived.mean().plot(kind='bar')


# In[7]:

#followed an example using Decision Tree
#first train model on 2/3 of data set
from sklearn.tree import DecisionTreeClassifier


# In[8]:

dtree = DecisionTreeClassifier()


# In[9]:

X_train=data_train[['Pclass']]
y = data_train['Survived']
X_test=data_test[['Pclass']]


# In[10]:

passengerid=data_test['PassengerId']


# In[11]:

#make prediction on remaining 1/3 of data set
dtree.fit(X_train,y)
prediction=dtree.predict(X_test)
data_prediction=pd.DataFrame(data=np.column_stack(
        [passengerid,prediction]),index=data_test.index.values,
        columns=['PassengerId','Survived'])


# In[12]:

data_prediction


# In[13]:

#check format of prediction, save to file
data_prediction.to_csv('c:\\kaggle\\myprediction.csv',index=False)


# In[14]:

testingit=pd.read_csv('c:\\kaggle\\myprediction.csv')
testingit


# In[15]:

compare=pd.read_csv('c:\\kaggle\\Titanic CSV\\gender_submission.csv')


# In[12]:

compare

