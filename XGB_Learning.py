'''
from sklearn.datasets import load_iris
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# read in the iris data
iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=1234565)

params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class': 3,
    'gamma': 0.1,
    'max_depth': 6,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
}

plst = list(params.items())
print(plst)
dtrain = xgb.DMatrix(X_train, y_train)
num_rounds = 10
model = xgb.train(plst, dtrain, num_rounds)

# 对测试集进行预测
dtest = xgb.DMatrix(X_test)
ans = model.predict(dtest)

# 计算准确率
cnt1 = 0
cnt2 = 0
for i in range(len(y_test)):
    if ans[i] == y_test[i]:
        cnt1 += 1
    else:
        cnt2 += 1

print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))

# 显示重要特征
plot_importance(model)
plt.show()
'''

#importing standard libraries
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

#import lightgbm and xgboost
import lightgbm as lgb
import xgboost as xgb

#loading our training dataset 'adult.csv' with name 'data' using pandas
data = pd.read_csv('C:\\Users\\it\\Downloads\\adult.csv', header=None)

#Assigning names to the columns
data.columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital_Status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'Income'
]

#glimpse of the dataset
#print(data.head())

# Label Encoding our target variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#将label二值化
l = LabelEncoder()
l.fit(data.Income)

data.Income = Series(l.transform(
    data.Income))  #label encoding our target variable
data.Income.value_counts()
category_features = [
    'workclass', 'education', 'marital_Status', 'occupation', 'relationship',
    'race', 'sex', 'native_country'
]

data[category_features] = data[category_features].astype('category')
print(data.head(10))
'''
#One Hot Encoding of the Categorical features 
one_hot_workclass=pd.get_dummies(data.workclass) 
one_hot_education=pd.get_dummies(data.education) 
one_hot_marital_Status=pd.get_dummies(data.marital_Status) 
one_hot_occupation=pd.get_dummies(data.occupation)
one_hot_relationship=pd.get_dummies(data.relationship) 
one_hot_race=pd.get_dummies(data.race) 
one_hot_sex=pd.get_dummies(data.sex) 
one_hot_native_country=pd.get_dummies(data.native_country) 

#removing categorical features 
data.drop(['workclass','education','marital_Status','occupation','relationship','race','sex','native_country'],axis=1,inplace=True) 
'''

#Merging one hot encoded features with our dataset 'data'
#data=pd.concat([data,one_hot_workclass,one_hot_education,one_hot_marital_Status,one_hot_occupation,one_hot_relationship,one_hot_race,one_hot_sex,one_hot_native_country],axis=1)

#removing dulpicate columns
#_, i = np.unique(data.columns, return_index=True)
#data = data.iloc[:, i]

#Here our target variable is 'Income' with values as 1 or 0.
#Separating our data into features dataset x and our target dataset y
x = data.drop('Income', axis=1)
y = data.Income

#Imputing missing values in our target variable
y.fillna(y.mode()[0], inplace=True)

#Now splitting our dataset into test and train
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)

#The data is stored in a DMatrix object
#label is used to define our outcome variable
dtrain = xgb.DMatrix(x_train, label=y_train, enable_categorical=True)
dtest = xgb.DMatrix(x_test, enable_categorical=True)

#setting parameters for xgboost
parameters = {
    'max_depth': 7,
    'eta': 1,
    'silent': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': .05
}

#training our model
num_round = 50
from datetime import datetime

start = datetime.now()
xg = xgb.train(parameters, dtrain, num_round)
stop = datetime.now()

#Execution time of the model
execution_time_xgb = stop - start
print(execution_time_xgb)

#datetime.timedelta( , , ) representation => (days , seconds , microseconds)
#now predicting our model on test set
ypred = xg.predict(dtest)
print(ypred)

#Converting probabilities into 1 or 0
for i in range(0, 9769):
    if ypred[i] >= .5:  # setting threshold to .5
        ypred[i] = 1
    else:
        ypred[i] = 0

#calculating accuracy of our model
from sklearn.metrics import accuracy_score

accuracy_xgb = accuracy_score(y_test, ypred)
print(accuracy_xgb)
'''
train_data = lgb.Dataset(x_train, label=y_train)

#setting parameters for lightgbm
param = {
    'num_leaves': 150,
    'objective': 'binary',
    'max_depth': 7,
    'learning_rate': .05,
    'max_bin': 200
}
param['metric'] = ['auc', 'binary_logloss']

#Here we have set max_depth in xgb and LightGBM to 7 to have a fair comparison between the two.
from datetime import datetime
#training our model using light gbm
num_round = 50
start = datetime.now()
lgbm = lgb.train(param, train_data, num_round)
stop = datetime.now()

#Execution time of the model
execution_time_lgbm = stop - start
print(execution_time_lgbm)

#predicting on test set
ypred2 = lgbm.predict(x_test)
print(ypred2[0:5])  # showing first 5 predictions

#converting probabilities into 0 or 1
for i in range(0, 9769):
    if ypred2[i] >= .5:  # setting threshold to .5
        ypred2[i] = 1
    else:
        ypred2[i] = 0
from sklearn.metrics import accuracy_score
#calculating accuracy
accuracy_lgbm = accuracy_score(ypred2, y_test)
accuracy_lgbm
y_test.value_counts()

from sklearn.metrics import roc_auc_score

#calculating roc_auc_score for xgboost
auc_xgb = roc_auc_score(y_test, ypred2)
print(auc_xgb)

#calculating roc_auc_score for light gbm.
auc_lgbm = roc_auc_score(y_test, ypred2)
auc_lgbm, comparison_dict = {
    'accuracy score': (accuracy_lgbm,),
    'auc score': (auc_lgbm, auc_xgb),
    'execution time': (execution_time_lgbm,)
}

#Creating a dataframe ‘comparison_df’ for comparing the performance of Lightgbm and xgb.
comparison_df = DataFrame(comparison_dict)
comparison_df.index = ['LightGBM', 'xgboost']
print(comparison_df)
'''