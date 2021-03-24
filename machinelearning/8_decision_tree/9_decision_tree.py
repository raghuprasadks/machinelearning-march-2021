#!/usr/bin/env python
# coding: utf-8

# <h2 style='color:blue' align="center">Decision Tree Classification</h2>

# In[19]:


import pandas as pd


# In[20]:


df = pd.read_csv("salaries.csv")
df.head()


# In[21]:


inputs = df.drop('salary_more_then_100k',axis='columns')


# In[22]:


target = df['salary_more_then_100k']


# In[38]:


from sklearn.preprocessing import LabelEncoder
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()


# In[24]:


inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_job.fit_transform(inputs['job'])
inputs['degree_n'] = le_degree.fit_transform(inputs['degree'])


# In[25]:


inputs


# In[26]:


inputs_n = inputs.drop(['company','job','degree'],axis='columns')


# In[27]:


inputs_n


# In[28]:


target


# In[33]:


from sklearn import tree
model = tree.DecisionTreeClassifier()


# In[34]:


model.fit(inputs_n, target)


# In[31]:


inputs_n


# In[32]:


target


# In[35]:


model.score(inputs_n,target)


# **Is salary of Google, Computer Engineer, Bachelors degree > 100 k ?**

# In[36]:


model.predict([[2,1,0]])


# **Is salary of Google, Computer Engineer, Masters degree > 100 k ?**

# In[37]:


model.predict([[2,1,1]])


# **Exercise: Build decision tree model to predict survival based on certain parameters**

# <img src="titanic.jpg" height=200 width=400/>

# CSV file is available to download at https://github.com/codebasics/py/blob/master/ML/9_decision_tree/Exercise/titanic.csv
# 
# ##### In this file using following columns build a model to predict if person would survive or not,
# 
# 1. Pclass
# 1. Sex
# 1. Age
# 1. Fare
# 
# ##### Calculate score of your model
