#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


#reading csv file into dataframe
df=pd.read_csv('http://bit.ly/w-data')


# In[3]:


df


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.info


# In[7]:


df.dtypes


# In[8]:


#plotting distribution of study hours
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# histogram of the Hours
df.Hours.plot(kind='hist', color='purple', edgecolor='black', figsize=(10,7))
plt.title('Distribution of Hours', size=24)
plt.xlabel('Hours', size=18)
plt.ylabel('Frequency', size=18)


# In[9]:


# histogram of the Scores
df.Scores.plot(kind='hist', color='purple', edgecolor='black', figsize=(10,7))
plt.title('Distribution of Percentage', size=24)
plt.xlabel('Scores', size=18)
plt.ylabel('Frequency', size=18);


# In[10]:


#alloting X as independent variable and Y as dependent variable
X = df["Hours"].values.reshape(-1,1)
Y = df["Scores"].values.reshape(-1,1)


# In[ ]:





# In[11]:


#splitting data for training and testing model
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                            test_size=0.2, random_state=0) 


# In[12]:


#linear regression
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 


# In[13]:


#y=b0+xb1
#fitting a line to show the linear relation between indepedent variable and dependent variable
line = regressor.coef_*X+regressor.intercept_


# In[16]:


#visualise the training data
plt.scatter(X_train, y_train,color="red")
plt.plot(X, line,color='b');
plt.show()


# In[17]:


#predicting using test data
Y_pred = regressor.predict(X_test)
Y_pred


# In[18]:


#visualise the test data
plt.scatter(X_test,y_test , color = "red")
plt.plot(X_test,Y_pred , color = "b")
plt.show()


# In[20]:


#actual scores vs predicted scores
df_predict = pd.DataFrame({"Hours": X_test.reshape(1,-1)[0] , "Actual Scores" : y_test.reshape(1,-1)[0] , "Predicted Scores" : Y_pred.reshape(1,-1)[0]})
df_predict


# In[21]:


#sorting the hours
df_sorted = df_predict.sort_values(by = "Hours")
df_sorted


# In[24]:


#visulising actual vs predicted scores
import seaborn as sns
title = "Actual Values Vs Predicted Values"
ax1 = sns.distplot(df_sorted["Actual Scores"], hist = False , color = "green" , label = "Actual Scores")
sns.distplot(df_sorted["Predicted Scores"] , hist = False , color = "yellow" , label = "Predicted Scores" , ax = ax1)
plt.legend()
plt.grid()
plt.title(title)
plt.show()


# In[27]:


#computing mean absolute error correlation accuracy
from sklearn.metrics import r2_score
from sklearn import metrics

mean_absolute_error=metrics.mean_absolute_error(y_test,Y_pred)
print('Mean absolute error:',mean_absolute_error)

corr=r2_score(y_train,regressor.predict(X_train))
print('correlation:',corr)

acc=r2_score(y_test,Y_pred)
print('Accuracy:',acc)SSs


# In[31]:


#prediction on new data
hours = 9.25
pred = regressor.predict([[9.25]])
pred


# In[ ]:





# In[ ]:




