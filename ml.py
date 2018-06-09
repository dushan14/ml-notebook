import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor



######################_getting_data_from_files_######################
# training data
train_file_path = 'train.csv'
train_data = pd.read_csv(train_file_path)

# test data
predicting_data_file_path='test.csv'
predicting_data = pd.read_csv(predicting_data_file_path)
######### ############################################################



######################_selecting_affecting_columns_##################
all_cols=train_data.columns 

# if selecting all columns drop predicting column
predicting_cols=all_cols.drop('SalePrice') 

# predicting_cols=['LotArea', 'YearBuilt','1stFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
#####################################################################



######################_X_y_predictingX_##############################
X=train_data[predicting_cols]
y=train_data.SalePrice

predicting_X_=predicting_data[predicting_cols]
#####################################################################



######################_handling_categorial_##########################
# dtypes of colmns
# print(training_data_selected_cols.dtypes)

# handling categorial cols using one_hot_encoding
one_hot_encoded_X = pd.get_dummies(X)
one_hot_encoded_predicting_X = pd.get_dummies(predicting_X_)
X_encoded, predicting_X_encoded = one_hot_encoded_X.align(one_hot_encoded_predicting_X, join='left', axis=1)
#####################################################################



######################_deviding_data_X_y_############################
# deviding training data for checking correctness
train_X,test_X,train_y,test_y=train_test_split(X_encoded,y,random_state=0)

#required data for prediction
predicting_X=predicting_X_encoded
#####################################################################



######################_null_col_handling_############################
# handling null columns
# checking null cols #train_data.isnull().any()

# droping
# cols_with_missing = [col for col in train_data.columns if train_data[col].isnull().any()]
# reduced_trin_data = train_data.drop(cols_with_missing, axis=1)
# reduced_test_data = test_data.drop(cols_with_missing, axis=1)

#using SimpleImputer
# for training data
imputed_X_train = train_X.copy()
imputed_X_test = test_X.copy()

# for predicting data
imputed_predicting_X=predicting_X.copy()

# cols with null in training data
cols_with_missing_training_data = (col for col in train_X.columns if train_X[col].isnull().any())

# cols with null in predicting data
cols_with_missing_predicting_data=(col for col in predicting_X.columns if predicting_X[col].isnull().any())

# set of all null cols
cols_with_missing=list(set(list(cols_with_missing_training_data)+list(cols_with_missing_predicting_data)))

for col in cols_with_missing:
    imputed_X_train[col + '_was_missing'] = imputed_X_train[col].isnull()
    imputed_X_test[col + '_was_missing'] = imputed_X_test[col].isnull()
    imputed_predicting_X[col + '_was_missing'] = imputed_predicting_X[col].isnull()

imputer = Imputer()
# imputering data in training data
imputed_X_train = imputer.fit_transform(imputed_X_train) #fit imputer and transform data
imputed_X_test = imputer.transform(imputed_X_test) #transform data

# imputering data in predicting data
imputed_predicting_X = imputer.transform(imputed_predicting_X)
#####################################################################



######################_model_selection_##############################
def get_mae_model(model_,X, y):
	# model=DecisionTreeRegressor()
    model=model_
    mae_val= -1 * cross_val_score(model, X, y, scoring = 'neg_mean_absolute_error').mean()
    print("MAE with ",type(model).__name__," \t:",mae_val)
    return mae_val

# finding max leaf nodes
def get_MAE_nodes(max_leaf_nodes, training_X, predicting_X, training_y, predicting_values_y):
    model=DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,random_state=0)
    model.fit(training_X,training_y)
    predicted_val_y=model.predict(predicting_X)
    MAE=mean_absolute_error(predicting_values_y,predicted_val_y)
    return (MAE)
# for max_leaf_nodes in [5,50,60,65,67,70,75,80,85,90]:
#     current_MAE=get_MAE_nodes(max_leaf_nodes,train_X,val_X,train_y,val_y)
#     print("max leaf nodes: %d \t\t Mean Absolute Error: %d" %(max_leaf_nodes,current_MAE))

# get_mae_model(DecisionTreeRegressor(),imputed_X_train,train_y)
# get_mae_model(RandomForestRegressor(),imputed_X_train,train_y)
# get_mae_model(XGBRegressor(),imputed_X_train,train_y)
#####################################################################



######################_trainging_the_model_##########################
# find the number of nodes for least MAE and create the model according to it
# model=DecisionTreeRegressor(max_leaf_nodes=75,random_state=0)  #model with specifying max leaf nodes
# scpecify n_jobs for XGBRegressor if dataset is too large. assign num of cores in machine to n_jobs 
# find using early_stopping_rounds and assign it to n_estimators, start with a big numner
model=XGBRegressor(n_estimators=547,learning_rate=0.05)
model.fit(imputed_X_train,train_y,early_stopping_rounds=20, eval_set=[(imputed_X_test, test_y)], verbose=True)
#####################################################################



######################_evaluating_the_model_#########################
predicted_values_for_traininig_set=model.predict(imputed_X_test)

# print correct and predicted values
# print("actual",'predicted\n')
# for idx,val in enumerate(test_y):
# 	print (val,predicted_values_for_traininig_set[idx])

# print mean absolute error
print("\nMAE:\t",mean_absolute_error(test_y,predicted_values_for_traininig_set))
#####################################################################



######################_prediction_for_required_data_#################
predicted_result=model.predict(imputed_predicting_X)
#####################################################################



######################_visualization_of_prediction_##################
# creating a csv
submission = pd.DataFrame({'Id': predicting_data.Id, 'SalePrice': predicted_result})
submission.to_csv('submission.csv', index=False)

# creating txt file
np.savetxt('output.txt',predicted_result,fmt='%.2f')
#####################################################################