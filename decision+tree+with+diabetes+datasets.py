
# coding: utf-8

# In[1]:

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix,accuracy_score
filepath="E:/DS_ML/ML/week5/"


# In[2]:

dfdiabetes=pd.read_csv(filepath+"Assignment_of_DT_RF/Diabetes_data.csv")


# In[3]:

dfdiabetes.describe()


# In[4]:

# check null values in dataframe
# dfdiabetes.isnull().values.any
null_column=dfdiabetes.columns[dfdiabetes.isnull().any()]
dfdiabetes[null_column].isnull().sum()


# In[5]:

# chwck overall null value in dataframe
# dfdiabetes.isnull()
# fillna to fill null values with mean of respective column in dataframe
dfdiabetes=dfdiabetes.fillna(dfdiabetes.mean())


# In[6]:

dfdiabetes.head()


# In[7]:

featurecol=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
targetcol=["Outcome"]
feature=dfdiabetes[featurecol]
target=dfdiabetes[targetcol]


# In[8]:

import seaborn as sns
get_ipython().magic('matplotlib inline')
sns.pairplot(dfdiabetes,x_vars=featurecol,y_vars=targetcol,kind="reg")
plt.show()


# In[9]:

import numpy as np
correlation=np.corrcoef(feature,rowvar=0)
correlation


# In[10]:

print(np.linalg.det(correlation))


# In[11]:

# determinnat of coffecinet is only 28% means above feature have only 28% probablity to predict outcome(diabetes yes/no) which is very poor
# we will remove feature which are tightly corelated

# insulin and skinthickness tighly corelated. skin thickness have less variance in data compare to isulin hence we drop skin thickness
# feature column


featurecol=["Pregnancies","Glucose","BloodPressure","Insulin","BMI","DiabetesPedigreeFunction"]
targetcol=["Outcome"]
feature=dfdiabetes[featurecol]
target=dfdiabetes[targetcol]


# In[12]:

correlation=np.corrcoef(feature,rowvar=0)
correlation


# In[13]:

print(np.linalg.det(correlation))
# determinnat of coffecinet is only 68% means above feature have  68% probablity to predict outcome(diabetes yes/no)


# In[48]:

from sklearn.cross_validation import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(feature,target,test_size=0.3,random_state=1)


# In[38]:

dstreemodel=DecisionTreeClassifier(criterion="entropy")
dstreemodel.fit(xtrain,ytrain)


# In[39]:

dstreemodel.feature_importances_
list(zip(featurecol,dstreemodel.feature_importances_))


# In[40]:

predicted=dstreemodel.predict(xtest)
print(accuracy_score(predicted,ytest))


# In[41]:

dfconfusion_matrx=confusion_matrix(ytest,predicted)
dfconfusion_matrx


# In[28]:

rfmodel=RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 25, random_state = 123)
rfmodel.fit(xtrain,ytrain)


# In[29]:

rfprediction=rfmodel.predict(xtest)
print(accuracy_score(rfprediction,ytest))


# In[30]:

rf_confusion_matrix=confusion_matrix(ytest,rfprediction)
rf_confusion_matrix


# In[51]:

# # using gridsearch to tune put decisiontree model
# from sklearn.grid_search import GridSearchCV
# # tuning_parameter={"criterion":['gini','entropy'],"max_depth":[3,7],"max_leaf_nodes":[10,20]}
# dt_parameters={"criterion":['gini','entropy'],"max_depth":[3,7],"max_leaf_nodes": [10,20]}
# gridsearchmodel=GridSearchCV(DecisionTreeClassifier(),dt_parameters)
# gridsearchmodel.fit(xtrain,ytrain)
# # print(xtrain.shape)
# # print(ytrain.shape)


# In[52]:

import pickle
import requests
import json


# In[54]:

pickle.dump(rfmodel,open('E:/DS_ML/ML/week6/rfmodel.pkl','wb'))


# In[ ]:



