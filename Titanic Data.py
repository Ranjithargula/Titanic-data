#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


import warnings 
warnings.filterwarnings('ignore')


# ## Reading the training file of titanic survivors data

# In[3]:


tid=pd.read_csv('titanictrain.csv')


# In[4]:


tid.head()


# ## Cleaning the Data

# In[5]:


tid.info()


# In[6]:


tid.isnull().sum()

#### there are lot of null values in Cabin and Age . Cabin is dropped because it doesn't contibute to data
# In[7]:


del tid['Cabin']

#### converting male and female in sex to 1 and 0 respectively
# In[8]:


sex=lambda x : 1 if x=='male' else 0
tid['Sex']=tid['Sex'].apply(sex)


# In[9]:


tid.sample(20)


# In[10]:


tid['Age']=tid['Age'].fillna(tid['Age'].interpolate())


# In[11]:


tid.isnull().sum()


# In[12]:


del tid['Embarked']


# In[13]:


tid.sample(20)


# In[14]:


tid=tid.set_index('PassengerId')


# In[15]:


X=tid[['Fare','Sex','Age','Pclass','SibSp','Parch']]


# In[16]:


y=tid['Survived']


# ## Validating Data

# In[17]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[18]:


X=tid[['Fare','Sex','Age','Pclass','SibSp','Parch']]
y=tid['Survived']
for i in range(10):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

    model1=LogisticRegression()
    model1=model1.fit(X_train,y_train)
    y_pred=model1.predict(X_test)

    print(metrics.accuracy_score(y_test,y_pred))


# ## Reading Test File of Titanic Dataset

# In[19]:


test_tid=pd.read_csv('titanictest.csv')


# In[20]:


test_tid.head()


# In[21]:


test_tid.info()


# ## Cleaning Data

# In[22]:


test_tid['Age']=test_tid['Age'].fillna(test_tid['Age'].interpolate())


# In[23]:


del test_tid['Cabin']


# In[24]:


test_tid=test_tid.set_index('PassengerId')


# In[25]:


test_tid['Fare']=test_tid['Fare'].fillna(test_tid['Fare'].mean())


# In[26]:


sex=lambda x : 1 if x=='male' else 0
test_tid['Sex']=test_tid['Sex'].apply(sex)


# # Creating Models using Train Data

# In[27]:


from sklearn.linear_model import LogisticRegression


# In[28]:


X_train=tid[['Pclass','Fare','Sex','Age','SibSp','Parch']]
y_train=tid['Survived']


# In[29]:


from sklearn.tree import DecisionTreeClassifier


# In[30]:


from sklearn.ensemble import RandomForestClassifier


# In[31]:


from sklearn.neighbors import KNeighborsClassifier


# In[32]:


X=tid[['Fare','Sex','Age','Pclass','SibSp','Parch']]
y=tid['Survived']
for i in range(10):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

    model1=DecisionTreeClassifier(criterion='entropy',min_samples_split=5)
    model1=model1.fit(X_train,y_train)
    y_pred=model1.predict(X_test)

    print(metrics.accuracy_score(y_test,y_pred))


# In[33]:


X=tid[['Fare','Sex','Age','Pclass','SibSp','Parch']]
y=tid['Survived']
for i in range(10):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

    model1=RandomForestClassifier(1000)
    model1=model1.fit(X_train,y_train)
    y_pred=model1.predict(X_test)

    print(metrics.accuracy_score(y_test,y_pred))


# In[34]:


from sklearn.svm import SVC


# In[35]:


X=tid[['Fare','Sex','Age','Pclass','SibSp','Parch']]
y=tid['Survived']
for i in range(10):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

    model1=SVC(kernel='linear')
    model1=model1.fit(X_train,y_train)
    y_pred=model1.predict(X_test)

    print(metrics.accuracy_score(y_test,y_pred))


# In[36]:


X=tid[['Fare','Sex','Age','Pclass','SibSp','Parch']]
y=tid['Survived']
for i in range(10):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

    model1=SVC(kernel='rbf')
    model1=model1.fit(X_train,y_train)
    y_pred=model1.predict(X_test)

    print(metrics.accuracy_score(y_test,y_pred))


# In[37]:


X=tid[['Fare','Sex','Age','Pclass','SibSp','Parch']]
y=tid['Survived']
for i in range(10):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

    model1=KNeighborsClassifier()
    model1=model1.fit(X_train,y_train)
    y_pred=model1.predict(X_test)

    print(metrics.accuracy_score(y_test,y_pred))


# In[ ]:





# #### Random Forest Had maximum Accuracy so it is used to predict for the test data

# In[38]:


X_train=tid[['Pclass','Fare','Sex','Age','SibSp','Parch']]
y_train=tid['Survived']
X_test=test_tid[['Pclass','Fare','Sex','Age','SibSp','Parch']]


# In[39]:


model=RandomForestClassifier(1000)
model=model.fit(X_train,y_train)
y_test=model.predict(X_test)


# In[40]:


test_tid['Survived']=y_test


# In[41]:


test_tid.to_csv('titanicpredicted.csv')


# In[ ]:




