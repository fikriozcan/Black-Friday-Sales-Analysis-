# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 07:51:45 2019

@author: ASUSNB
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
file ='BlackFriday.xlsx'
bf = pd.read_excel(file)
bf.info()
bf.isnull().sum()

bf.head(10)

plt.rcParams['patch.force_edgecolor'] = True
plt.rcParams['patch.facecolor'] = 'b'

# rainbow colors
rb = []
colors = plt.cm.rainbow(np.linspace(0,1,18))
for c in colors:
    rb.append(c)
rb = reversed(rb)
rb = list(rb)

# viridis colors
vd = []
colors = plt.cm.GnBu(np.linspace(0,1,6))
for c in colors:
    vd.append(c)
vd = list(vd)
bf.info()
sns.set(style="whitegrid")

##################################
bf.hist(bins=50, figsize=(20,15))
plt.show()
################################################
bf['Purchase'].sum()
df_1=bf.groupby('Age').agg({'Purchase':'sum'})
df_1.reset_index(inplace=True)
ax = sns.barplot(x="Age", y="Purchase",data=df_1)


bf['Gender'].value_counts()

ax = sns.barplot(x="Occupation", y="Purchase", data=bf, estimator=sum)

################
ax = sns.barplot(x="Age", y="Product_Category_1", data=bf, estimator=sum)
ax = sns.barplot(x="Age", y="Product_Category_2", data=bf, estimator=sum)
ax = sns.barplot(x="Age", y="Product_Category_3", data=bf, estimator=sum)


bf['combined_G_M'] = bf.apply(lambda x:'%s_%s' % (x['Gender'],x['Marital_Status']),axis=1)

print(bf['combined_G_M'].unique())


ax = sns.barplot(x="combined_G_M", y="Purchase", data=bf,  estimator=sum)
ax = sns.barplot(x="Stay_In_Current_City_Years", y="Purchase", data=bf,  estimator=sum)

#### which products are profitable 

df_3 = bf.groupby('Product_ID').agg({'Purchase':'sum', 'Product_Category_1':'sum','Product_Category_2':'sum','Product_Category_3':'sum' })

df_3['%_of_product_on_purchase'] = df_3['Purchase'] / df_3['Purchase'].sum()
df_3.reset_index(inplace=True)
df_3['product_segments'] = df_3['%_of_product_on_purchase']

## 3623 poduct var fakat ilk 25 tanesi total purchase'in %10 unu veriyor. 

df4 = df_3.sort_values('%_of_product_on_purchase',ascending=False).head(25)

df4["%_of_product_on_purchase"].sum()

for val in enumerate(df_3.loc[ : , '%_of_product_on_purchase']):
     if val[1] > 0.003 :
         df_3.loc[val[0],'product_segments'] = 1
     if val[1] < 0.003 and val[1]>0.001:
         df_3.loc[val[0],'product_segments'] = 2
     if val[1] <0.001 and val[1]>0.0001 :
         df_3.loc[val[0],'product_segments'] = 3
     if val[1] < 0.0001 :
         df_3.loc[val[0],'product_segments'] = 4


#################################################

## which customers are profitable? where they come from? 
# top 105 customer generates %10 of total purchase

df_5 = bf.groupby('User_ID').agg({'Purchase':'sum', 'Product_Category_1':'sum','Product_Category_2':'sum','Product_Category_3':'sum'})
df_5['%_of_product_on_purchase'] = df_5['Purchase'] / df_5['Purchase'].sum()

df_5.reset_index(inplace=True)
df_6 = df_5.sort_values('Purchase',ascending=False).head(105)

    
df_6["%_of_product_on_purchase"].sum()

############

# where they come from and whşch categories they prefer? 
# first which city category makes the highest purchase? 
# city B has the highest purchase but why? 
# demographic should be understood? 

df_7= bf.groupby('City_Category').agg({'Purchase':'sum', 'Product_Category_1':'sum','Product_Category_2':'sum','Product_Category_3':'sum'})
df_7.reset_index(inplace=True)
df_7['%_of_product_on_purchase'] = df_7['Purchase'] / df_7['Purchase'].sum()

ax = sns.barplot(x="City_Category", y="%_of_product_on_purchase", data=df_7)
### WHİCH PRODUCT CATEGORIES PREFERRED BY AGE GROUP AND CITY CATEGORIES? 

ax = sns.barplot(x="City_Category", y="Product_Category_1", data=df_7)
ax = sns.barplot(x="City_Category", y="Product_Category_2", data=df_7)
ax = sns.barplot(x="City_Category", y="Product_Category_3", data=df_7)

fig1, ax1 = plt.subplots(figsize=(12,7))
sns.countplot(bf['Age'],hue=bf['City_Category'])

# city b has the highest population of every age category exclude 0-17  and 55+. and also has the target customer higher than others.

bf['City_Category'].value_counts()

## city A has the 2 second highest population of 26 -25 but total profit percentage lower but could be better for investement in the future. 
# CİTY B İS HIGHHERST PURCHASE 

## kisi basına hangisi daha cok harcıyor? 
city_A = bf[bf["City_Category"] == "A" ]
city_B = bf[bf["City_Category"] == "B" ]
city_C = bf[bf["City_Category"] == "C" ]


city_A['Product_Category_2']= (city_A['Product_Category_2'].fillna(0))
city_B['Product_Category_2']= (city_B['Product_Category_2'].fillna(0))
city_C['Product_Category_2']= (city_C['Product_Category_2'].fillna(0))

city_A['Product_Category_3']= (city_A['Product_Category_3'].fillna(0))
city_B['Product_Category_3']= (city_B['Product_Category_3'].fillna(0))
city_C['Product_Category_3']= (city_C['Product_Category_3'].fillna(0))

city_A.isnull().sum()
city_B.isnull().sum()
city_C.isnull().sum()


df_8 = city_A.groupby('User_ID').agg({'Purchase':'sum'})
df_8.reset_index(inplace=True)

avg_cityA_spending_per_person = city_A['Purchase'].sum()/ city_A['User_ID'].count()
print(avg_cityA_spending_per_person)
df_9 = city_B.groupby('User_ID').agg({'Purchase':'sum'})
df_9.reset_index(inplace=True)
avg_cityB_spending_per_person = city_B['Purchase'].sum()/ city_B['User_ID'].count()
print(avg_cityB_spending_per_person)

df_10 =city_C.groupby('User_ID').agg({'Purchase':'sum'})
df_10.reset_index(inplace=True)
avg_cityC_spending_per_person = city_C['Purchase'].sum()/ city_C['User_ID'].count()
print(avg_cityC_spending_per_person)

## AVERAGE SPENDİNG SHOWS THAT C İS HIGHETS A IS LOWEST

# B HISHEST PURCHASE AND HISHESST TARGET CUSTOMER BUT CITY C HIGHEST SPENDING PER CUSTOMER

### WHİCH PRODUCT CATEGORIES PREFERRED BY AGE GROUP?  



## TOP 105 CUSTOMER NEREDE VE NE ALIYOR? 
df_11 = bf.groupby('Product_ID').agg({'Purchase':'sum', 'Product_Category_1':'sum'})

df_12 = bf.groupby('Product_ID').agg({'Purchase':'sum', 'Product_Category_2':'sum'})

df_13 = bf.groupby('Product_ID').agg({'Purchase':'sum', 'Product_Category_3':'sum'})

fig, ax = plt.subplots(2,2, figsize=(18,12))
 
labels_max = df_11.sort_values(by='Product_Category_1', ascending=False).head(3) # find label for top 7 types for attack=
labels_min = df_11.sort_values(by='Product_Category_1', ascending=True).head(3) # find label for last 3 types for attack
label_high = labels_max.index.tolist()
label_low = labels_min.index.tolist()


ax[1,0].scatter(x=df_11['Product_Category_1'], y=df_11['Purchase'],s=200,label=df_11.index, c=rb, alpha=0.7)
for label, x, y in zip(label_high, labels_max['Product_Category_1'], labels_max['Purchase']):
    ax[1,0].annotate(
        label, xy=(x, y), xytext=(-20, -5), textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='gold', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
for label, a, b in zip(label_low, labels_min['Product_Category_1'], labels_min['Purchase']):
    ax[1,0].annotate(
        label, xy=(a, b), xytext=(14, 40), textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='gold', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
ax[1,0].set_title('Price & Sales Relationship per item category')

##################################
labels_max_2 = df_12.sort_values(by='Product_Category_2', ascending=False).head(3) # find label for top 7 types for attack=
labels_min_2 = df_12.sort_values(by='Product_Category_2', ascending=True).head(3) # find label for last 3 types for attack
label_high_2 = labels_max.index.tolist()
label_low_2 = labels_min.index.tolist()


ax[0,0].scatter(x=df_12['Product_Category_2'], y=df_12['Purchase'],s=200,label=df_12.index, c=rb, alpha=0.7)
for label, x, y in zip(label_high_2, labels_max_2['Product_Category_2'], labels_max_2['Purchase']):
    ax[0,0].annotate(
        label, xy=(x, y), xytext=(-20, -5), textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='gold', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
for label, a, b in zip(label_low_2, labels_min_2['Product_Category_2'], labels_min_2['Purchase']):
    ax[0,0].annotate(
        label, xy=(a, b), xytext=(14, 40), textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='gold', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
ax[0,0].set_title('Price & Sales Relationship per item category')

########################################
labels_max_3 = df_13.sort_values(by='Product_Category_3', ascending=False).head(5) # find label for top 7 types for attack=
labels_min_3 = df_13.sort_values(by='Product_Category_3', ascending=True).head(5) # find label for last 3 types for attack
label_high_3 = labels_max.index.tolist()
label_low_3 = labels_min.index.tolist()


ax[0,1].scatter(x=df_13['Product_Category_3'], y=df_13['Purchase'],s=200,label=df_13.index, c=rb, alpha=0.7)
for label, x, y in zip(label_high_3, labels_max_3['Product_Category_3'], labels_max_3['Purchase']):
    ax[0,1].annotate(
        label, xy=(x, y), xytext=(-20, -5), textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='gold', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
for label, a, b in zip(label_low_3, labels_min_3['Product_Category_3'], labels_min_3['Purchase']):
    ax[0,1].annotate(
        label, xy=(a, b), xytext=(14, 40), textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='gold', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
ax[0,1].set_title('Price & Sales Relationship per item category')


###############################################

bf['is_duplicated'] = bf.duplicated(['User_ID'])

non_dup_df= bf[bf["is_duplicated"] == False ]
non_dup_df.info()

ax = sns.barplot(x="Stay_In_Current_City_Years", y="Purchase", data=non_dup_df, hue = 'City_Category')

ax = sns.barplot(x="combined_G_M", y="Purchase", data=non_dup_df, hue = 'City_Category')

sns.distplot( bf["Purchase"] , color="red", label="Sales")
################################################################################
corr_matrix = bf.corr()
corr_matrix["Purchase"].sort_values(ascending=False)

df_corr = bf.corr().round(2)
sns.palplot(sns.color_palette('coolwarm', 12))
fig, ax = plt.subplots(figsize=(15,15))
sns.heatmap(df_corr,
            cmap = 'coolwarm',
            square = True,
            annot = True,
            linecolor = 'black',
            linewidths = 0.5)
plt.show()

######################## 

######### REGRESSION MODELS FOR PURCHASE 

bf['Product_Category_2']= (bf['Product_Category_2'].fillna(0))
bf['Product_Category_3']= (bf['Product_Category_3'].fillna(0))
bf.isnull().sum()


bf['Age'].value_counts()
bf['Gender'].value_counts()
bf['City_Category'].value_counts()

bf.loc[:, "Age"] = pd.factorize(bf.Age)[0]
bf.loc[:, "City_Category"] = pd.factorize(bf.City_Category)[0]
bf.loc[:, "Gender"] = pd.factorize(bf.Gender)[0]# suna bak 
City_dummies = pd.get_dummies((bf['City_Category']))
Age_dummies = pd.get_dummies((bf['Age']),drop_first = True)
bf_with_dummies = pd.concat([bf.iloc[:,:],
                      City_dummies,
                      Age_dummies],
                      axis = 1)

bf_with_dummies = bf_with_dummies.drop(['City_Category','Age'],axis=1)
bf_with_dummies.loc[:, "Gender"] = pd.factorize(bf_with_dummies.Gender)[0]# suna bak 


## creating new variable, product segmentation. 

#bf['%_of_product_on_purchase'] = bf['Purchase'] / bf['Purchase'].sum()

#for val in enumerate(bf.loc[ : , '%_of_product_on_purchase']):
#     if val[1] > 0.003 :
#         bf.loc[val[0],'product_segments'] = 1
#     elif val[1] < 0.003 and val[1]>0.001:
 #        bf.loc[val[0],'product_segments'] = 2
  #   elif val[1] <0.001 and val[1]>0.0001 :
   #      bf.loc[val[0],'product_segments'] = 3
    # elif val[1] < 0.0001 :
     #    bf.loc[val[0],'product_segments'] = 4
     
y = bf_with_dummies.loc[:, "Purchase"]

X = bf_with_dummies.drop(['Purchase', 'User_ID', 'Product_ID'],axis = 1)

from sklearn.model_selection import train_test_split # train/test split


X_train, X_test, y_train, y_test = train_test_split(
                                    X,
                                    y,
                                    test_size = 0.2,
                                    random_state = 508,
                                    stratify = y)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score


predictions = lin_reg.predict(X_test)
score_1 = r2_score(y_test, predictions) 
score_2 = mean_squared_error(y_test, predictions) 
score_3 =explained_variance_score (y_test, predictions)


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)


grid_search.fit(X_train, y_train)
grid_search.best_params_
grid_search.best_estimator_
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
    
final_model = grid_search.best_estimator_
final_predictions = final_model.predict(X_test)

score_11 = r2_score(y_test, final_predictions) 
score_22= mean_squared_error(y_test, final_predictions) 
score_33 =explained_variance_score (y_test, final_predictions)














