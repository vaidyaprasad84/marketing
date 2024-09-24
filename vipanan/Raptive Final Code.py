#!/usr/bin/env python
# coding: utf-8

# ### Importing all the required libraries

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Lasso, LinearRegression
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse, r2_score

import catboost as cb
import lightgbm as lgb
import xgboost as xgb

from scipy.stats import norm,zscore


# ## Problem I Solution

# ### Reading the data file

# In[ ]:


data = pd.read_csv('testdata (1) (1).csv')
data.head()


# ### Exploratory Data Analysis (EDA)

# #### EDA.1 Checking missing values

# In[ ]:


data.info()


# #### EDA.2 Outlier Detections

# In[ ]:


# Box plot based outlier detection
nrows = 1
ncols = 2

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 5))
sns.boxplot(ax=axes[0],y = data['revenue'])
sns.boxplot(ax=axes[1],y = data['top'])
fig.tight_layout()
plt.savefig('outlier.png',bbox_inches='tight',dpi = 1200)
plt.show()


# In[ ]:


# Box plot based outlier detection split across dimensions
nrows = 2
ncols = 3

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))

for i in range(len(num_cols)):
    for j in range(len(cat_cols)):
        sns.boxplot(ax=axes[i,j],x = cat_cols[j], y = num_cols[i], data = data)
        axes[i,j].tick_params(axis='x', labelsize=14)
        axes[i,j].tick_params(axis='y', labelsize=14)
        axes[i,j].xaxis.label.set(fontsize=16)
        axes[i,j].yaxis.label.set(fontsize=16)
fig.tight_layout()
plt.savefig('eda.png',bbox_inches='tight',dpi = 1200)
plt.show()


# In[ ]:


# IQR based outlier detection
Q1 = np.percentile(data['revenue'], 25, method='midpoint')
Q3 = np.percentile(data['revenue'], 75, method='midpoint')
IQR = Q3 - Q1
upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR

print(len(data[(data['revenue']> upper) | (data['revenue']< lower)]))

Q1 = np.percentile(data['top'], 25, method='midpoint')
Q3 = np.percentile(data['top'], 75, method='midpoint')
IQR = Q3 - Q1
upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR

print(len(data[(data['top']> upper) | (data['top']< lower)]))


# In[ ]:


# Zscore based outlier detection
z_revenue= np.abs(zscore(data['revenue']))
z_top= np.abs(zscore(data['top']))
z = pd.DataFrame({'rev':z_revenue, 'top':z_top})
print(len(z[z['rev']>=3]))
print(len(z[z['top']>=3]))


# ### Visualizations for Relationships between Revenue and top 

# #### Visual.1 Overall Revenue vs top

# In[ ]:


sns.lmplot(x = 'top',y = 'revenue',data = data, markers = '+', truncate = False)
plt.xlabel('Time on Page (top)',fontsize=14)
plt.ylabel('Revenue',fontsize=14)
plt.title('Revenue vs Time on Page',fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(bottom = 0,top = 0.024)
print(data.corr(numeric_only=True)['revenue'])
plt.savefig('overall.png',bbox_inches='tight',dpi = 1200)


# #### Visual.2 Revenue vs top (Controlled by platform)

# In[ ]:


sns.lmplot(x = 'top',y = 'revenue',data = data, hue = 'platform', markers = '+', truncate = False,
           legend_out=False)
plt.xlabel('Time on Page (top)',fontsize=14)
plt.ylabel('Revenue',fontsize=14)
plt.legend(fontsize=14)
plt.title('Revenue vs Time on Page',fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(bottom = 0,top = 0.024)

df = data[data['platform']=='mobile']
print(df.corr(numeric_only=True))
df = data[data['platform']=='desktop']
print(df.corr(numeric_only=True))
plt.savefig('plat.png',bbox_inches='tight',dpi = 1200)


# #### Visual.3 Revenue vs top (Controlled by browser)

# In[ ]:


sns.lmplot(x = 'top',y = 'revenue',data = data, hue = 'browser', markers = '+', truncate = False,
           legend_out=False)
plt.xlabel('Time on Page (top)',fontsize=14)
plt.ylabel('Revenue',fontsize=14)
plt.legend(fontsize=14)
plt.title('Revenue vs Time on Page',fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(bottom = 0,top = 0.024)

df = data[data['browser']=='safari']
print(df.corr(numeric_only=True))
df = data[data['browser']=='chrome']
print(df.corr(numeric_only=True))
plt.savefig('brow.png',bbox_inches='tight',dpi = 1200)


# #### Visual.4 Revenue vs top (Controlled by site)

# In[ ]:


sns.lmplot(x = 'top',y = 'revenue',data = data, hue = 'Site', markers = '+', truncate = False,
           legend_out=False)
plt.xlabel('Time on Page (top)',fontsize=14)
plt.ylabel('Revenue',fontsize=14)
plt.legend(fontsize=14)
plt.title('Revenue vs Time on Page',fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(bottom = 0,top = 0.024)

df = data[data['Site']=='Site_1']
print(df.corr(numeric_only=True))
df = data[data['Site']=='Site_2']
print(df.corr(numeric_only=True))
df = data[data['Site']=='Site_3']
print(df.corr(numeric_only=True))
df = data[data['Site']=='Site_4']
print(df.corr(numeric_only=True))
plt.savefig('site.png',bbox_inches='tight',dpi = 1200)


# #### Visual.5 Revenue vs top (Controlled by platform & browser)

# In[ ]:


sns.lmplot(x = 'top', y ='revenue', hue = 'Platform & Browser', data = data, markers = '+', truncate = False, legend_out=False)
plt.xlabel('Time on Page (top)',fontsize=14)
plt.ylabel('Revenue',fontsize=14)
plt.legend(fontsize=14)
plt.title('Revenue vs Time on Page',fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(bottom = 0,top = 0.024)

df = data[(data['platform']=='desktop') & (data['browser']=='chrome')]
print(df.corr(numeric_only=True))
df = data[(data['platform']=='desktop') & (data['browser']=='safari')]
print(df.corr(numeric_only=True))
df = data[(data['platform']=='mobile') & (data['browser']=='chrome')]
print(df.corr(numeric_only=True))
df = data[(data['platform']=='mobile') & (data['browser']=='safari')]
print(df.corr(numeric_only=True))
plt.savefig('plat_brow.png',bbox_inches='tight',dpi = 1200)


# #### Visual.5 Revenue vs top (Controlled by all others)

# In[ ]:


nrows = 1
ncols = 3

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5))
data['site_plat'] = data['Site'] + data['platform']
data['site_brow'] = data['Site'] + data['browser']
data['site_plat_brow'] = data['Site'] + data['platform'] + data['browser']

sns.scatterplot(ax = axes[0],x = 'top', y ='revenue', hue = 'site_plat', data = data)
sns.scatterplot(ax = axes[1],x = 'top', y ='revenue', hue = 'site_brow', data = data)
sns.scatterplot(ax = axes[2],x = 'top', y ='revenue', hue = 'site_plat_brow', data = data)
axes[0].tick_params(axis='x', labelsize=14)
axes[0].tick_params(axis='y', labelsize=14)
axes[0].xaxis.label.set(fontsize=16)
axes[0].yaxis.label.set(fontsize=16)
axes[1].tick_params(axis='x', labelsize=14)
axes[1].tick_params(axis='y', labelsize=14)
axes[1].xaxis.label.set(fontsize=16)
axes[1].yaxis.label.set(fontsize=16)
axes[2].tick_params(axis='x', labelsize=14)
axes[2].tick_params(axis='y', labelsize=14)
axes[2].xaxis.label.set(fontsize=16)
axes[2].yaxis.label.set(fontsize=16)
fig.tight_layout()
plt.savefig('roa1.png',bbox_inches='tight',dpi = 1200)


# ### Model Fitting

# #### Model.1 Linear Regression fit

# In[ ]:


df = data[data['platform']=='mobile']
lm1 = LinearRegression()
lm1.fit(df['top'].values.reshape(-1,1),df['revenue'].values.reshape(-1,1))
y_pred1 = lm1.predict(df['top'].values.reshape(-1,1))
y_true1 =  df['revenue'].values
print(lm1.coef_, lm1.intercept_)

df = data[data['platform']=='desktop']
lm2 = LinearRegression()
lm2.fit(df['top'].values.reshape(-1,1),df['revenue'].values.reshape(-1,1))
y_pred2 = lm2.predict(df['top'].values.reshape(-1,1))
y_true2 =  df['revenue'].values
print(lm2.coef_, lm2.intercept_)

ypred = np.concatenate((y_pred1,y_pred2))
ytrue = np.concatenate((y_true1,y_true2))

print(mse(ytrue,ypred, squared = False))
print(r2_score(ytrue,ypred))


# #### Model.2 Decision Tree Fit

# In[ ]:


scaler = StandardScaler()

train['top'] = scaler.fit_transform(train['top'].values.reshape(-1,1))

y = train['revenue'].to_numpy()
X = train.drop(columns=['revenue'])

clf = tree.DecisionTreeRegressor(random_state=0,max_leaf_nodes=4)

clf.fit(X,y)

plt.figure(figsize=(10,6))  # set plot size (denoted in inches)
tree.plot_tree(clf,feature_names = X.columns.tolist(), fontsize=10,precision = 7)
plt.savefig('tree.png',bbox_inches='tight',dpi = 1200)
plt.show()


# #### Model.3 CatBoost Fit

# In[ ]:


cat_train = data.copy()
cat_y = cat_train['revenue'].values
cat_X = cat_train.drop(columns = ['revenue'])
cat_features = ['browser','platform','Site']
model = cb.CatBoostRegressor(loss_function='RMSE',silent = True, cat_features = cat_features)
model.fit(cat_X,cat_y)

sorted_feature_importance = model.feature_importances_.argsort()
plt.barh(cat_X.columns[sorted_feature_importance], 
        model.feature_importances_[sorted_feature_importance], 
        color='turquoise')
plt.xlabel("CatBoost Feature Importance")
plt.savefig('catboost.png',bbox_inches='tight',dpi = 1200)


# #### Model.4 Xgboost Fit

# In[ ]:


lgbm_X = cat_X.copy()
for c in cat_features:
    lgbm_X[c] = lgbm_X[c].astype('category')
    
xgbr = xgb.XGBRegressor(enable_categorical=True)
xgbr.fit(lgbm_X,cat_y)

sorted_feature_importance = xgbr.feature_importances_.argsort()
plt.barh(lgbm_X.columns[sorted_feature_importance], 
        xgbr.feature_importances_[sorted_feature_importance], 
        color='turquoise')
plt.xlabel("XGBoost Feature Importance")
plt.savefig('xgboost.png',bbox_inches='tight',dpi = 1200)


# In[ ]:





# ## Problem II Solution

# ### Bernoulli Distribution

# In[ ]:


Z = np.random.choice(11,10000,p = [0.07,0.09,0.11,0.13,0.11,0.08,0.07,0.08,0.10,0.09,0.07])/10
unq = np.unique(Z)
for i in range(10):
    X = len(Z[np.where(Z==unq[i])])
    X = np.random.uniform(unq[i],unq[i+1],X)
    if i==0:
        final_X = X
    else:
        final_X = np.concatenate((final_X,X))
        
X = len(Z[np.where(Z==unq[-1])])
X = np.random.uniform(0,1,X)
final_X = np.concatenate((final_X,X))
        
sns.kdeplot(final_X,cut = 0)
plt.ylim(bottom = 0)
plt.xlabel('Random Variable (Generated Data)',fontsize=14)
plt.ylabel('Probability Density',fontsize=14)
plt.title('Bernoulli Distribution based',fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('bern_dist.png',bbox_inches='tight',dpi = 1200)


# In[ ]:


df = pd.DataFrame({'Sample_data':final_X})
df = df.sample(frac=1)
df.to_csv('prob_dist.csv',index = False)


# ### Beta Distribution

# In[ ]:


X = np.random.beta(2,2,50000)*0.5-0.1
Y = np.random.beta(2,2,50000)*0.7+0.3
Z = np.concatenate((X,Y))
Z = Z[np.where(Z>=0)]
Z = Z[np.where(Z<=1)]

sns.kdeplot(Z,cut = 0)
plt.ylim(bottom = 0)
plt.xlabel('Random Variable (Generated Data)',fontsize=14)
plt.ylabel('Probability Density',fontsize=14)
plt.title('Beta Distribution based',fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('beta_dist.png',bbox_inches='tight',dpi = 1200)


# ### Truncated Normal Distribution

# In[ ]:


dist = norm(loc=0.2, scale=0.1)
a = 0 # lower cutoff
b = 0.7 # upper cutoff

X = np.random.uniform(0,1,7000) * (dist.cdf(b)-dist.cdf(a)) + dist.cdf(a)
Y1 = dist.ppf(X)

dist = norm(loc=0.7, scale=0.1)
a = 0.4 # lower cutoff
b = 1 # upper cutoff

X = np.random.uniform(0,1,3000) * (dist.cdf(b)-dist.cdf(a)) + dist.cdf(a)
Y2 = dist.ppf(X)

Y = np.concatenate((Y1,Y2))

sns.kdeplot(Y,cut = 0)
plt.ylim(bottom = 0)
plt.xlabel('Random Variable (Generated Data)',fontsize=14)
plt.ylabel('Probability Density',fontsize=14)
plt.title('Truncated Normal Distribution based',fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('truncated_normal_dist.png',bbox_inches='tight',dpi = 1200)

