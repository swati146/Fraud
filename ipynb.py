#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Detection

# ### Introduction
# Everyday billions of credit card transactions are made all around the world. Considering the widespread use of smartphones and the Internet throughout the earth, more and more people are using their credit cards to make purchases online, making payments through apps,etc...
# 
# In a scenario such as this one, it is extremely important that credit card companies are able to easily recognize when a transaction is a result of a fraud or a genuine purchase, avoiding that customers end up being charged for items they did not acquire.
# 
# In this project, I'll use the scikit-learn library to develop a prediction model that is able to learn and detect when a transaction is a fraud or a genuine purchase. I intend to use two classification algorithms, Decision Tree and Random Forest, to identify which one of them achieve the best results with our dataset.

# In[46]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import plot_confusion_matrix


# In[14]:


df=pd.read_csv(r"C:\Users\91805\OneDrive\Desktop\creditcard.csv", encoding='latin-1')


# In[15]:


df


# In[16]:


df.dtypes


# In[17]:


df.describe()


# In[18]:


df.isna().values.any()


# In[19]:


df.Amount.describe().round(2)


# In[20]:


df.Amount.max()


# In[21]:


fig = px.scatter(df, x = 'Amount', y =df.index, color = df.Class,
                title = 'Distribution of Amount Values')
fig.update_layout(xaxis_title='Transaction amount')
fig.update_layout(yaxis_title='Transaction')


# In[22]:


df.query("Class==1").sort_values('Amount',ascending = False)


# In[23]:


df.query("Class==1").sort_values('Amount',ascending=False)


# In[24]:


sns.set(rc={'figure.figsize':(12,9)})
df.Class.value_counts().plot(kind='pie',autopct='%.2f%%',explode=(0,0.1))


# In[25]:


scaler=StandardScaler()
df['Normalized_Amount']=scaler.fit_transform(df.Amount.values.reshape(-1,1))
df.drop(['Amount','Time'], inplace = True, axis=1)
df


# In[26]:


Y=df.Class
X=df.drop(['Class'],axis = 1)
(train_x,test_x,train_y,test_y) = train_test_split(X,Y, test_size=0.3,random_state=42)

print("train_x size:", train_x.shape)
print("test_x size", test_x.shape)


# In[27]:


decision_tree = DecisionTreeClassifier()
random_forest = RandomForestClassifier(n_estimators=100)


decision_tree.fit(train_x,train_y)
prediction_dt = decision_tree.predict(test_x)
decision_tree_score = round(decision_tree.score(test_x,test_y)*100,2)

random_forest.fit(train_x,train_y)
prediction_rf = random_forest.predict(test_x)
random_forest_score = round(random_forest.score(test_x,test_y)*100,2)


print('Decision Tree Performace:' , decision_tree_score)
print('Random Forest Performance:', random_forest_score)


# In[28]:


metrics_decision_tree = [['Accuracy',(accuracy_score(test_y, prediction_dt))],
                         ['Precision',precision_score(test_y, prediction_dt)],
                         ['Recall', recall_score(test_y, prediction_dt)],
                         ['F1_score',f1_score(test_y, prediction_dt)]]
metric_values_decision_tree = pd.DataFrame(metrics_decision_tree, columns = ['Metrics', 'Result'])
metric_values_decision_tree


# In[29]:


# Confusion Matrix
confusion_matrix_decision_tree = confusion_matrix(test_y, prediction_dt)
# Visualization
ax = plt.subplot()
sns.heatmap(confusion_matrix_decision_tree, annot=True, fmt='g', ax = ax)
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values')
ax.set_title('Confusion Matrix - Decision Tree')
ax.xaxis.set_ticklabels(['Genuine','Fraud'])
ax.yaxis.set_ticklabels(['Genuine','Fraud'])


# In[47]:





# In[ ]:





# In[ ]:





# In[44]:





# In[ ]:




