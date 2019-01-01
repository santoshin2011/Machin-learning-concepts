
# coding: utf-8

# In[3]:

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score


# In[4]:

filepath="E:/DS_ML/ML/week9/Naive bayes/attachment_naivebayeshandson1/"
filename="sms_spam.csv"
ftype="csv"
fsep=','


# In[5]:

def Read_file(filepath,filename,ftype,fsep):
    if ftype=='csv':
        return pd.read_csv(filepath+filename,sep=fsep)
        if ftype=='xls':
            return  pd.read_excel(filepath+filename)
          
   


# In[6]:

df=Read_file(filepath,filename,ftype,fsep)


# In[7]:

df.head()


# In[13]:

nltk.download('stopwords')
stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)


# In[14]:

vectorizer.fit(df)


# In[16]:

df.replace('spam',1,inplace=True)
df.replace('ham',0,inplace=True)


# In[18]:

df.text
x=vectorizer.fit_transform(df.text)


# In[37]:

split_str=df.text[10].split(' ')
split_str


# In[53]:

a=split_str
len(a)


# In[59]:

print(x[10])


# In[58]:

vectorizer.get_feature_names()[5000]


# In[62]:

y=df.type


# In[64]:

xtrain ,xtest,ytrain,ytest=train_test_split(x,y,random_state=10)


# In[69]:

df.type


# In[66]:

clf=naive_bayes.MultinomialNB()
model=clf.fit(xtrain,ytrain)


# In[70]:

predicted_class=model.predict(xtest[])
predicted_class


# In[76]:

df.loc[[100,101]]


# In[80]:

predicted_class[100]
predicted_class[101]


# In[82]:

##Check model's accuracy
roc_auc_score(ytest, clf.predict_proba(xtest)[:,1])


# In[ ]:



