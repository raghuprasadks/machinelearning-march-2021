#!/usr/bin/env python
# coding: utf-8

# <h3 style='color:purple' align='center'>Random Forest Python Tutorial</h3>

# <img src="forest.jpg" width="500" height="600" />

# **Digits dataset from sklearn**

# In[1]:


import pandas as pd
from sklearn.datasets import load_digits
digits = load_digits()


# In[2]:


dir(digits)


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[4]:


plt.gray() 
for i in range(4):
    plt.matshow(digits.images[i]) 


# In[5]:


df = pd.DataFrame(digits.data)
df.head()
df.shape


# In[6]:


df['target'] = digits.target


# In[7]:


df[0:12]


# **Train and the model and prediction**

# In[8]:


X = df.drop('target',axis='columns')
y = df.target


# In[9]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


# In[10]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=20)
model.fit(X_train, y_train)


# In[11]:


model.score(X_test, y_test)


# In[12]:


y_predicted = model.predict(X_test)


# **Confusion Matrix**

# In[17]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
cm


# In[18]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# **Exercise**

# <img src='iris.png' width=200 height='100'/>

# Use famous iris flower dataset from sklearn.datasets to predict flower species using random forest classifier.
# 1. Measure prediction score using default n_estimators (10)
# 2. Now fine tune your model by changing number of trees in your classifer and tell me what best score you can get using how many trees
