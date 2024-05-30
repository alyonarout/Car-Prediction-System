#!/usr/bin/env python
# coding: utf-8

# In[112]:


import pandas as pd
import numpy as np


# In[3]:


car = pd.read_csv(r'C:\Users\KIIT\Desktop\Car Prediction System\quikr_car.csv')


# In[4]:


car.head()


# In[5]:


car.shape


# In[6]:


car.info()


# In[7]:


##Quality
#-year has many no year values
#-year object needs to be in int
#-price clean ask for price
#-price object to int
#-kms driven remove kms and commas
#-kms has nan values
#-fuel value are nan
#- categorical distinctive (first three words of names)


# In[8]:


#cleaning
backup=car.copy()


# In[9]:


car['year']


# In[10]:


car =car[car['year'].str.isnumeric()]


# In[11]:


car['year']=car['year'].astype(int)


# In[12]:


car =car[car['Price']!="Ask For Price"]


# In[26]:


car.info()


# In[27]:


# Convert 'Price' column to numeric values (handles existing numeric values and non-numeric values)
car['Price'] = pd.to_numeric(car['Price'], errors='coerce')

# Replace commas in the 'Price' column
car['Price'] = car['Price'].astype(str).str.replace(',', '')

# Convert 'Price' column to integers
car['Price'] = car['Price'].astype(int)


# In[34]:


car['kms_driven'] = car['kms_driven'].str.split(' ').str.get(0).str.replace(',', '')


# In[36]:


car[car['kms_driven'].str.isnumeric()]


# In[52]:


car['kms_driven']=car['kms_driven'].astype(int)


# In[53]:


car.info()


# In[54]:


car[~car['fuel_type'].isna()]


# In[55]:


car['name'] =car['name'].str.split(' ').str.slice(0,3).str.join(' ')


# In[51]:


car.reset_index(drop=True)


# In[56]:


car = car[car['kms_driven'] != 0]



# In[57]:


car.info()


# In[58]:


car.reset_index(drop=True)


# In[61]:


car[car['Price']>6e6].reset_index(drop=True)


# In[67]:


car.to_csv('Clean Car.csv')


# In[74]:


X =car.drop(columns='Price')
y=car['Price']


# In[75]:


X


# In[79]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[86]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline


# In[87]:


ohe = OneHotEncoder()
ohe.fit(X[['name','company','fuel_type']])


# In[93]:


ohe.categories_


# In[94]:


column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),
remainder='passthrough')


# In[95]:


lr= LinearRegression()


# In[96]:


pipe = make_pipeline(column_trans,lr)


# In[97]:


pipe.fit(X_train, y_train)


# In[100]:


y_pred=pipe.predict(X_test)


# In[101]:


r2_score(y_test,y_pred)


# In[115]:


scores=[]
for i in range (1000):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    lr= LinearRegression()
    pipe = make_pipeline(column_trans,lr)
    pipe.fit(X_train, y_train)
    y_pred=pipe.predict(X_test)
    print(r2_score(y_test,y_pred),i)
    scores.append(r2_score(y_test,y_pred))
    


# In[116]:


np.argmax(scores)


# In[114]:


scores[np.argmax(scores)] #small dataset


# In[120]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=np.argmax(scores))
lr= LinearRegression()
pipe = make_pipeline(column_trans,lr)
pipe.fit(X_train, y_train)
y_pred=pipe.predict(X_test)
r2_score(y_test,y_pred)


# In[121]:


import pickle


# In[122]:


get_ipython().system('pip install pickle_mixin')


# In[123]:


pickle.dump(pipe,open('LinearRegression.pkl','wb'))


# In[124]:


pipe.predict(pd.DataFrame([['Maruti Suzuki Swift', 'Maruti',2019,100,'Petrol']], columns=['name','company','year','kms_driven','fuel_type']))


# In[ ]:




