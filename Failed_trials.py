import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, MaxAbsScaler, RobustScaler, StandardScaler
import seaborn as sns
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.stats import probplot
from scipy import stats
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor



df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")




# df_train['X2'].fillna(df_train['X2'].mode()[0], inplace=True)
# df_test['X2'].fillna(df_test['X2'].mode()[0], inplace=True)
#
df_train['X3'].fillna(df_train['X3'].mode()[0], inplace=True)
df_test['X3'].fillna(df_test['X3'].mode()[0], inplace=True)
#
# df_train['X4'].fillna(df_train['X4'].mode()[0], inplace=True)
# df_test['X4'].fillna(df_test['X4'].mode()[0], inplace=True)
#
df_train['X5'].fillna(df_train['X5'].mode()[0], inplace=True)
df_test['X5'].fillna(df_test['X5'].mode()[0], inplace=True)
#
# df_train['X6'].fillna(df_train['X6'].mode()[0], inplace=True)
# df_test['X6'].fillna(df_test['X6'].mode()[0], inplace=True)
#
df_train['X7'].fillna(df_train['X7'].mode()[0], inplace=True)
df_test['X7'].fillna(df_test['X7'].mode()[0], inplace=True)
#
# df_train['X8'].fillna(df_train['X8'].mode()[0], inplace=True)
# df_test['X8'].fillna(df_test['X8'].mode()[0], inplace=True)
#
df_train['X9'].fillna(df_train['X9'].mode()[0], inplace=True)
df_test['X9'].fillna(df_test['X9'].mode()[0], inplace=True)

df_train['X10'].fillna(df_train['X10'].mode()[0], inplace=True)
df_test['X10'].fillna(df_test['X10'].mode()[0], inplace=True)

df_train['X11'].fillna(df_train['X11'].mode()[0], inplace=True)
df_test['X11'].fillna(df_test['X11'].mode()[0], inplace=True)
#
# df_train['Y'].fillna(df_train['Y'].mode()[0], inplace=True)

newdf_train2 = df_train.drop_duplicates()
newdf_test2 = df_test.drop_duplicates()

newdf_train2.drop(['X1'], axis=1, inplace=True)
newdf_test2.drop(['X1'], axis=1, inplace=True)

# print(newdf_train2['X3'].unique())
# print(newdf_test2['X3'].unique())
#
# print(newdf_train2['X5'].unique())
# print(newdf_test2['X5'].unique())
#
# print(newdf_train2['X7'].unique())
# print(newdf_test2['X7'].unique())
#
# print(newdf_train2['X8'].unique())
# print(newdf_test2['X8'].unique())
#
# print(newdf_train2['X9'].unique())
# print(newdf_test2['X9'].unique())
#
# print(newdf_train2['X10'].unique())
# print(newdf_test2['X10'].unique())
#
# print(newdf_train2['X11'].unique())
# print(newdf_test2['X11'].unique())

newdf_train2['X3'].replace(['low fat', 'LF'], 'Low Fat', inplace=True)
newdf_test2['X3'].replace(['low fat', 'LF'], 'Low Fat', inplace=True)

newdf_train2['X3'].replace(['reg'], 'Regular', inplace=True)
newdf_test2['X3'].replace(['reg'], 'Regular', inplace=True)

# print(newdf_train2['X3'].unique())
# print(newdf_test2['X3'].unique())

LE = LabelEncoder()
newdf_train2['X3'] = LE.fit_transform(newdf_train2['X3'])
newdf_test2['X3'] = LE.transform(newdf_test2['X3'])
newdf_train2['X7'] = LE.fit_transform(newdf_train2['X7'])
newdf_test2['X7'] = LE.transform(newdf_test2['X7'])
newdf_train2['X9'] = LE.fit_transform(newdf_train2['X9'])
newdf_test2['X9'] = LE.transform(newdf_test2['X9'])
newdf_train2['X10'] = LE.fit_transform(newdf_train2['X10'])
newdf_test2['X10'] = LE.transform(newdf_test2['X10'])

X5_train = pd.get_dummies(newdf_train2['X5'])
X5_test = pd.get_dummies(newdf_test2['X5'])

X11_train = pd.get_dummies(newdf_train2['X11'])
X11_test = pd.get_dummies(newdf_test2['X11'])

newdf_train2.drop(['X5', 'X11'], axis=1, inplace=True)
newdf_test2.drop(['X5', 'X11'], axis=1, inplace=True)

newdf_train3 = pd.concat([newdf_train2, X5_train, X11_train], axis=1)
newdf_test3 = pd.concat([newdf_test2, X5_test, X11_test], axis=1)

newdf_train3.reset_index(inplace=True)
newdf_test3.reset_index(inplace=True)

newdf_train3.drop(['index'], axis=1, inplace=True)
newdf_test3.drop(['index'], axis=1, inplace=True)

mice_imputer = IterativeImputer(estimator=LinearRegression(), imputation_order='ascending')
df_train_mice_imputed = pd.DataFrame(mice_imputer.fit_transform(newdf_train3.drop(['Y'], axis=1)), columns=newdf_train3.drop(['Y'], axis=1).columns)
df_test_mice_imputed = pd.DataFrame(mice_imputer.transform(newdf_test3), columns=newdf_test3.columns)

df_train_mice_imputed['Y'] = newdf_train3['Y']

print(newdf_train3['Y'])
print(df_train_mice_imputed['Y'])
# numeric = ['X2', 'X4', 'X6']
# newdf_train3.boxplot(numeric)
#plt.show()

# fig, ax = plt.subplots(figsize=(18, 10))
# ax.scatter(newdf_train3['X2'], newdf_train3['Y'])
# ax.set_xlabel('X2')
# ax.set_ylabel('Y')
# plt.show()
#
# fig, ax = plt.subplots(figsize=(18, 10))
# ax.scatter(newdf_train3['X4'], newdf_train3['Y'])
# ax.set_xlabel('X4')
# ax.set_ylabel('Y')
# plt.show()
#
# fig, ax = plt.subplots(figsize=(18, 10))
# ax.scatter(newdf_train3['X6'], newdf_train3['Y'])
# ax.set_xlabel('X6')
# ax.set_ylabel('Y')
# plt.show()



# Z2 = np.abs(stats.zscore(newdf_train3['X2']))
# Z4 = np.abs(stats.zscore(newdf_train3['X4']))
# # Z6 = np.abs(stats.zscore(newdf_train3['X6']))
# # threshold = 3
# # OUT2 = np.where(Z2 > 3)
# OUT4 = np.where(Z4 > 3)
# # OUT6 = np.where(Z6 > 3)
# # print(OUT2)
# print(OUT4)
# print(OUT6)
# newdf_train3.drop( OUT4[0], inplace=True)
# fig, ax = plt.subplots(figsize=(18, 10))
# ax.scatter(newdf_train3['X4'], newdf_train3['Y'])
# ax.set_xlabel('X4_NEW')
# ax.set_ylabel('Y')
# plt.show()


# Q12 = np.percentile(newdf_train3['X2'], 25, method='midpoint')
# Q32 = np.percentile(newdf_train3['X2'], 75, method='midpoint')
# IQR2 = Q32 - Q12
# upper2 = newdf_train3['X2'] >= (Q32+1.5*IQR2)
# print("Upper bound2:", upper2)
# print(np.where(upper2))
# lower2 = newdf_train3['X2'] <= (Q12-1.5*IQR2)
# print("Lower bound2:", lower2)
# print(np.where(lower2))

# Q14 = np.percentile(newdf_train3['X4'], 25, method='midpoint')
# Q34 = np.percentile(newdf_train3['X4'], 75, method='midpoint')
# IQR4 = Q34 - Q14
# upper4 = newdf_train3['X4'] >= (Q34+1.5*IQR4)
# print("Upper bound4:", upper4)
# print(np.where(upper4))
# lower4 = newdf_train3['X4'] <= (Q14-1.5*IQR4)
# print("Lower bound4:", lower4)
# print(np.where(lower4))

# Q16 = np.percentile(newdf_train3['X6'], 25, method='midpoint')
# Q36 = np.percentile(newdf_train3['X6'], 75,method='midpoint')
# IQR6 = Q36 - Q16
# upper6 = newdf_train3['X6'] >= (Q36+1.5*IQR6)
# print("Upper bound6:", upper6)
# print(np.where(upper6))
# lower6 = newdf_train3['X6'] <= (Q16-1.5*IQR6)
# print("Lower bound6:", lower6)
# print(np.where(lower6))


train = df_train_mice_imputed.sample(frac=1, random_state=20).reset_index(drop=True)
test = df_test_mice_imputed.sample(frac=1, random_state=20).reset_index(drop=True)

# corrmat = train.corr()
# top_corr_features = corrmat.index
# plt.figure(figsize=(train.shape[1], train.shape[1]))
# g = sns.heatmap(train[top_corr_features].corr(), annot=True)
# plt.show()

# g = pd.DataFrame(train['X3']).copy()
# for i in g.columns:
#     probplot(x=g[i], dist='norm', plot=plt)
#     plt.title(i)
#     plt.show()

# model = ExtraTreesClassifier()
X_train = train.drop(['Y'], axis=1)
y_train = train['Y']
X_test = test
# model.fit(X_train, y_train)
# print(model.feature_importances_)
# feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
# feat_importances.nlargest(10).plot(kind='barh')
# plt.show()

bestfeatures = SelectKBest(score_func=f_regression, k='all')
fit = bestfeatures.fit(X_train,y_train)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X_train.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']
print(featureScores.nlargest(train.shape[1]-1,'Score'))
MinMax = MinMaxScaler().fit(X_train.drop(
    ['Breakfast', 'Canned', 'Frozen Foods', 'Baking Goods',
'Fruits and Vegetables', 'Snack Foods', 'Soft Drinks', 'Breads',
'Starchy Foods', 'Household', 'Health and Hygiene', 'Others', 'Dairy', 'Meat',
'Hard Drinks', 'Seafood', 'X3', 'X4'], axis=1))
X_train_norm = MinMax.transform(X_train.drop(
    ['Breakfast', 'Canned', 'Frozen Foods', 'Baking Goods',
'Fruits and Vegetables', 'Snack Foods', 'Soft Drinks', 'Breads',
'Starchy Foods', 'Household', 'Health and Hygiene', 'Others', 'Dairy', 'Meat',
'Hard Drinks', 'Seafood', 'X3', 'X4'], axis=1))
X_test_norm = MinMax.transform(X_test.drop(
    ['Breakfast', 'Canned', 'Frozen Foods', 'Baking Goods',
'Fruits and Vegetables', 'Snack Foods', 'Soft Drinks', 'Breads',
'Starchy Foods', 'Household', 'Health and Hygiene', 'Others', 'Dairy', 'Meat',
'Hard Drinks', 'Seafood', 'X3', 'X4'], axis=1))

MinMax = MinMaxScaler().fit(X_train)
X_train_norm = MinMax.transform(X_train)
X_test_norm = MinMax.transform(X_test)

# MaxAbs = MaxAbsScaler().fit(X_train)
# X_train_normAbs = MaxAbs.transform(X_train)
# X_test_normAbs = MaxAbs.transform(X_test)

# rob = RobustScaler().fit(X_train)
# X_train_normRob = rob.transform(X_train)
# X_test_normRob = rob.transform(X_test)

List = []
for i in range(X_test_norm.shape[0]):
    List.append(i)
CB = CatBoostRegressor().fit(X_train_norm, y_train)
y_test_CB_MinMax = pd.DataFrame(List, columns=['row_id'])
y_pred_CB_MinMax = CB.predict(X_test_norm)
y_test_CB_MinMax['Y'] = y_pred_CB_MinMax
# y_test_CB_MinMax.to_csv('CatBoost_result_MinMax_more_than_5(2).csv', index=False)


LR_MinMax = LinearRegression()
LR_MinMax.fit(X_train_norm, y_train)

y_test_LR_MinMax = pd.DataFrame(List, columns=['row_id'])
y_pred_LR_MinMax = LR_MinMax.predict(X_test_norm)
y_test_LR_MinMax['Y'] = y_pred_LR_MinMax
# y_test_LR_MinMax.to_csv('LR_result_MinMax_more_than_5wo.csv', index=False)

# LR_Rob = LinearRegression()
# LR_Rob.fit(X_train_normRob, y_train)
#
# y_test_LR_Rob = pd.DataFrame(List, columns=['row_id'])
# y_pred_LR_Rob = LR_Rob.predict(X_test_normRob)
# y_test_LR_Rob['Y'] = y_pred_LR_Rob
# y_test_LR_Rob.to_csv('LR_result_Rob_more_than_5wo.csv', index=False)

SVR_MinMax = SVR(kernel='rbf')
SVR_MinMax.fit(X_train_norm, y_train)
y_test_SVR_MinMax = pd.DataFrame(List, columns=['row_id'])
y_pred_SVR_MinMax = SVR_MinMax.predict(X_test_norm)
y_test_SVR_MinMax['Y'] = y_pred_SVR_MinMax
# y_test_SVR_MinMax.to_csv('SVR_result_MinMax_more_than_5.csv', index=False)

EN_MinMax = ElasticNet(alpha=0.0011)
EN_MinMax.fit(X_train_norm, y_train)
y_test_EN_MinMax = pd.DataFrame(List, columns=['row_id'])
y_pred_EN_MinMax = EN_MinMax.predict(X_test_norm)
y_test_EN_MinMax['Y'] = y_pred_EN_MinMax
# y_test_EN_MinMax.to_csv('EN_result_MinMax_more_than_5.csv', index=False)


# std = StandardScaler().fit(X_train.drop(
#     ['Starchy Foods', 'X3', 'Snack Foods', 'Soft Drinks', 'Frozen Foods', 'Health and Hygiene', 'Dairy', 'Canned',
#      'Household', 'Meat', 'Hard Drinks', 'Breads', 'Seafood'], axis=1))
# X_train_std = std.transform(X_train.drop(
#     ['Starchy Foods', 'X3', 'Snack Foods', 'Soft Drinks', 'Frozen Foods', 'Health and Hygiene', 'Dairy', 'Canned',
#      'Household', 'Meat', 'Hard Drinks', 'Breads', 'Seafood'], axis=1))
# X_test_std = std.transform(X_test.drop(
#     ['Starchy Foods', 'X3', 'Snack Foods', 'Soft Drinks', 'Frozen Foods', 'Health and Hygiene', 'Dairy', 'Canned',
#      'Household', 'Meat', 'Hard Drinks', 'Breads', 'Seafood'], axis=1))
# LR_std = LinearRegression()
# LR_std.fit(X_train_std, y_train)
# y_test_LR_std = pd.DataFrame(List, columns=['row_id'])
# y_pred_LR_std = LR_std.predict(X_test_std)
# y_test_LR_std['Y'] = y_pred_LR_std
# y_test_LR_std.to_csv('LR_result_std_more_than_5.csv', index=False)

Ridge_MinMax = Ridge(alpha=0.6)
Ridge_MinMax.fit(X_train_norm, y_train)
y_test_Ridge_MinMax = pd.DataFrame(List, columns=['row_id'])
y_pred_Ridge_MinMax = Ridge_MinMax.predict(X_test_norm)
y_test_Ridge_MinMax['Y'] = y_pred_Ridge_MinMax
# y_test_Ridge_MinMax.to_csv('Ridge_result_MinMax_more_than_5.csv', index=False)

Lasso_MinMax = Lasso(alpha=0.005)
Lasso_MinMax.fit(X_train_norm, y_train)
y_test_Lasso_MinMax = pd.DataFrame(List, columns=['row_id'])
y_pred_Lasso_MinMax = Lasso_MinMax.predict(X_test_norm)
y_test_Lasso_MinMax['Y'] = y_pred_Lasso_MinMax
# y_test_Lasso_MinMax.to_csv('Lasso_result_MinMax_more_than_5.csv', index=False)

ETR = ExtraTreesRegressor()
ETR.fit(X_train_norm, y_train)
y_test_ETR_MinMax = pd.DataFrame(List, columns=['row_id'])
y_pred_ETR_MinMax = ETR.predict(X_test_norm)
y_test_ETR_MinMax['Y'] = y_pred_ETR_MinMax
# y_test_ETR_MinMax.to_csv('ETR_result_MinMax_more_than_5.csv', index=False)

AdaBoost = AdaBoostRegressor()
AdaBoost.fit(X_train_norm, y_train)
y_test_Adaboost_MinMax = pd.DataFrame(List, columns=['row_id'])
y_pred_Adaboost_MinMax = AdaBoost.predict(X_test_norm)
y_test_Adaboost_MinMax['Y'] = y_pred_Adaboost_MinMax
# y_test_Adaboost_MinMax.to_csv('Adaboost_result_MinMax_more_than_5.csv', index=False)

# DT_MinMax = DecisionTreeRegressor()
# DT_MinMax.fit(X_train_norm, y_train)
# y_test_DT_MinMax = pd.DataFrame(List, columns=['row_id'])
# y_pred_DT_MinMax = DT_MinMax.predict(X_test_norm)
# y_test_DT_MinMax['Y'] = y_pred_DT_MinMax
# y_test_DT_MinMax.to_csv('DT_result_MinMax_more_than_5.csv', index=False)