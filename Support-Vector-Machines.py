#!/usr/bin/env python
# coding: utf-8

# 
# # Support Vector Machines with Python
# 

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Get the Data
# 
# We'll use the built in breast cancer dataset from Scikit Learn. We can get with the load function:

# In[11]:


from sklearn.datasets import load_breast_cancer


# In[12]:


cancer = load_breast_cancer()


# The data set is presented in a dictionary form:

# In[13]:


cancer.keys()


# In[14]:


print(cancer['data'])


# We can chech each elemnt of above list here:

# In[51]:


print(cancer['feature_names'])


# ## Set up DataFrame

# In[52]:


df_feat = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df_feat.head(3)


# In[53]:


df_feat.info()


# In[54]:


cancer['target_names']


# In[55]:


cancer['target']


# In[56]:


df_target = pd.DataFrame(cancer['target'],columns=['Cancer'])


# # Exploratory Data Analysis
# 
# 

# In[57]:


df = pd.concat([df_feat, df_target], axis=1)


# In[58]:


df.head(1)


# In[59]:


sns.scatterplot(x='mean radius',y='mean texture',hue='Cancer',data=df)


# In[60]:


sns.countplot(x='Cancer',data=df)


# ## Train Test Split

# In[61]:


from sklearn.model_selection import train_test_split


# In[62]:


X_train, X_test, y_train, y_test = train_test_split(df_feat, np.ravel(df_target), test_size=0.30, random_state=101)


# # Train the Support Vector Classifier

# In[70]:


from sklearn.svm import SVC


# In[71]:


model = SVC()


# In[72]:


model.fit(X_train,y_train)


# ## Predictions and Evaluations
# 
# Now let's predict using the trained model.

# In[74]:


predictions = model.predict(X_test)


# In[75]:


from sklearn.metrics import classification_report,confusion_matrix


# In[76]:


print(confusion_matrix(y_test,predictions))


# In[77]:


print(classification_report(y_test,predictions))


# # Gridsearch
# 
# 

# In[78]:


param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 


# In[79]:


from sklearn.model_selection import GridSearchCV


# In[80]:


grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)


# In[81]:


# May take awhile!
grid.fit(X_train,y_train)


# You can inspect the best parameters found by GridSearchCV in the best_params_ attribute:

# In[82]:


grid.best_params_


# Then you can re-run predictions on this grid object just like you would with a normal model.

# In[83]:


grid_predictions = grid.predict(X_test)


# In[84]:


print(confusion_matrix(y_test,grid_predictions))


# In[85]:


print(classification_report(y_test,grid_predictions))

