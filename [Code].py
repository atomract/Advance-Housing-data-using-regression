#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import sklearn
import time
import sys
import os


# In[2]:


DeprecationWarning('ignore')
warnings.filterwarnings('ignore')


# In[3]:


os.chdir("E:/data_science/housing")
os.listdir()
df=pd.read_csv("train.csv")


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


mean=df['LotFrontage'].mean


# In[7]:


mean


# In[8]:


sns.distplot(df['MSSubClass'])


# In[10]:


plt.boxplot(df['LotArea'])


# In[11]:


plt.boxplot(df['OverallQual'])


# In[12]:


plt.boxplot(df['OverallCond'])


# In[13]:


plt.hist(df['OverallQual'])


# In[14]:


sns.scatterplot(x=df['OverallQual'],y=df['SalePrice'])


# In[15]:


sns.countplot(x=df['OverallQual'],hue=df['MSZoning'])


# In[16]:


sns.countplot(x=df['OverallQual'],hue=df['LotShape'])


# In[17]:


sns.countplot(x=df['OverallQual'],hue=df['LandSlope'])


# In[18]:


sns.countplot(x=df['OverallQual'])


# In[19]:


sns.countplot(x=df['OverallCond'])


# In[20]:


df['SalePrice'].plot.hist()


# In[21]:


sns.distplot(df.LotFrontage.dropna())


# In[22]:


sns.distplot(df.LotFrontage.fillna(df.LotFrontage.median(),inplace=False))


# In[23]:


df.LotFrontage.fillna(df.LotFrontage.median(),inplace=True)


# In[24]:


df['Alley'].dtypes     #o means object type


# In[25]:


df.Alley.fillna("No Class",inplace=True)


# In[26]:


df.isnull().sum()


# In[27]:


df['LotShape'].value_counts()


# In[28]:


sns.scatterplot(x=df['LotFrontage'],y=df['SalePrice'])


# In[29]:


df.dtypes


# In[30]:


sns.scatterplot(x=df['MSSubClass'],y=df['SalePrice'])


# In[31]:


sns.scatterplot(x=df['LotArea'],y=df['SalePrice'])


# In[32]:


sns.scatterplot(x=df['YearBuilt'],y=df['SalePrice'])


# In[33]:


corr=df.corr()


# In[34]:


corr


# In[35]:


plt.hist(df['OverallQual'])
plt.hist(df['OverallCond'])
plt.show()


# In[36]:


plt.hist(df['OverallQual'])
plt.hist(df['MSSubClass'])
plt.show()


# In[37]:


from sklearn.linear_model import Ridge as slr


# In[38]:


from sklearn.linear_model import Lasso as sll


# In[39]:


column=df.columns
for col in column:
    if df[col].isnull().sum()>0:
        print(col,df[col].isnull().sum())


# all the columns having null values

# In[40]:


df['MasVnrType'].value_counts()


# In[41]:


df['MasVnrType'].dtypes


# In[42]:


df['MasVnrType'].fillna("NoClass",inplace=True)


# In[43]:


df['BsmtQual'].value_counts()


# In[44]:


df['BsmtQual'].fillna("Na",inplace=True)


# In[45]:


df['BsmtCond'].value_counts()


# In[46]:


df['BsmtCond'].fillna("Na",inplace=True)


# In[47]:


df['BsmtExposure'].value_counts()


# In[48]:


df['BsmtExposure'].fillna("Na",inplace=True)


# In[49]:


column=df.columns
for col in column:
    if df[col].isnull().sum()>0:
        print(col,df[col].isnull().sum())


# In[50]:


df['BsmtFinType1'].value_counts()


# In[51]:


df['BsmtFinType1'].fillna("Na",inplace=True)


# In[52]:


df['BsmtFinType2'].value_counts()


# In[53]:


df['BsmtFinType2'].fillna("Na",inplace=True)


# In[54]:


df['FireplaceQu'].value_counts()


# In[55]:


df['Fireplaces']


# In[56]:


df['Fireplaces'].value_counts()


# In[57]:


df['FireplaceQu'].fillna("Na",inplace=True)


# In[58]:


df['GarageType'].value_counts()


# In[59]:


df['GarageType'].fillna("Na",inplace=True)


# In[60]:


df['GarageYrBlt'].fillna(0,inplace=True)


# In[61]:


df['GarageFinish'].value_counts()


# In[62]:


df['GarageFinish'].fillna("Na",inplace=True)


# In[63]:


df['GarageQual'].fillna("Na",inplace=True)


# In[64]:


df['GarageCond'].fillna("Na",inplace=True)


# In[65]:


column=df.columns
for col in column:
    if df[col].isnull().sum()>0:
        print(col,df[col].isnull().sum())


# In[66]:


df['PoolArea'].value_counts()


# In[67]:


df['PoolQC'].fillna("Na",inplace=True)


# In[68]:


df['Fence'].value_counts()


# In[69]:


df['Fence'].fillna("Na",inplace=True)


# In[70]:


df['MiscFeature'].value_counts()


# In[71]:


df['MiscFeature'].fillna("Na",inplace=True)


# In[72]:


df['Electrical'].value_counts()


# In[73]:


df['Electrical'].fillna("No Class",inplace=True)


# In[74]:


column=df.columns
for col in column:
    if df[col].isnull().sum()>0:
        print(col,df[col].isnull().sum())


# In[75]:


sns.distplot(df.MasVnrArea.dropna())


# In[76]:


df.MasVnrArea.fillna(df.MasVnrArea.median(),inplace=True)


# In[77]:


from sklearn.preprocessing import LabelEncoder
cat=df.select_dtypes(include=object)
def labels(df):
    for col in cat.columns:
        label=LabelEncoder()
        df[col]=label.fit_transform(df[col])
    return df


# In[78]:


df=labels(df)


# In[79]:


df.isnull().sum().sum()


# In[80]:


for i in cat.columns:
    print(i, df[i].unique())


# In[81]:


df['MasVnrArea'].unique()


# In[82]:


from sklearn.model_selection import train_test_split
train,test=train_test_split(df,test_size=0.2,random_state=112)
del df


# In[83]:


def x_and_y(df):
    x=df.drop(['SalePrice'],axis=1)
    y=df['SalePrice']
    return x,y
x_train,y_train=x_and_y(train)
x_test,y_test=x_and_y(test)


# In[84]:


lin_model=LinearRegression()
lin_model.fit(x_train,y_train)


# In[85]:


prediction=lin_model.predict(x_train)
prediction


# In[86]:


score=r2_score(y_train,prediction)
score


# In[87]:


test_pred=lin_model.predict(x_test)
test_score=r2_score(y_test,test_pred)
test_score


# In[ ]:




