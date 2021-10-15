#!/usr/bin/env python
# coding: utf-8

# In[2]:


import warnings
warnings.simplefilter("ignore")


# In[3]:


import pandas as pd
import numpy as np


# In[4]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[74]:


dataset = pd.read_csv('Fish.csv')
dataset


# In[8]:


dataset.shape


# In[9]:


dataset.head()


# In[32]:


dataset=dataset.loc[::,['Weight','Length1','Length2','Length3','Height','Width']]


# In[33]:


dataset


# In[34]:


x = dataset.iloc[:,0]


# In[35]:


x


# In[36]:


x.shape


# In[37]:


x = dataset.iloc[:,0].values.reshape(-1,1)


# In[38]:


x.shape


# In[39]:


y = dataset.iloc[:,-1].values.reshape(-1,1)


# In[40]:


y.shape


# In[41]:


y


# In[42]:


plt.scatter(x,y)
plt.show


# In[43]:


from sklearn.model_selection import train_test_split


# In[44]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)


# In[45]:


x_train.shape


# In[46]:


x_test.shape


# In[47]:


y_train.shape


# In[48]:


y_test.shape


# In[49]:


from sklearn.linear_model import LinearRegression


# In[50]:


lm = LinearRegression()


# In[51]:


lm.fit(x_train,y_train)


# In[52]:


y_pred = lm.predict(x_test)


# In[73]:


y_pred


# In[70]:


plt.scatter(x,y,color='blue')
plt.plot(x_test,y_pred,color='red')


# In[ ]:




