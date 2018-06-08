import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer



######################_getting_data_from_files_######################
# training data
train_file_path = '../input/train.csv'
train_data = pd.read_csv(train_file_path)

# test data
predicting_data_file_path='../input/test.csv'
predicting_data = pd.read_csv(predicting_data_file_path)
#####################################################################



######################_selecting_affecting_columns_##################
# all_cols=train_data.columns
# after selecting cols for prediction 
predicting_cols=['LotArea', 'YearBuilt','1stFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
#####################################################################



######################_deviding_data_X_y_and_pred_data_##############
y=train_data.ClassOfPredicting
X=house_data[predicting_cols]
#if getting all cols drop ClassOfPredicting col
# X=train_data.drop(['ClassOfPredicting'], axis=1)

# deviding training data for checking correctness
train_X,test_X,train_y,test_y=train_test_split(X,y,random_state=0)

#required data for prediction
predicting_X=predicting_data[predicting_cols]
#####################################################################



######################_data_preprocessing_###########################
# checking null cols
# train_data.isnull().any()

# handling null columns
# droping
# cols_with_missing = [col for col in train_data.columns if train_data[col].isnull().any()]
# reduced_trin_data = train_data.drop(cols_with_missing, axis=1)
# reduced_test_data = test_data.drop(cols_with_missing, axis=1)

#using SimpleImputer
imputed_X_train = train_X.copy()
imputed_X_test = test_X.copy()

cols_with_missing = (col for col in X_train.columns if X_train[col].isnull().any())

#adding a col to track imputed cols
for col in cols_with_missing:
    imputed_X_train[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test[col + '_was_missing'] = imputed_X_test_plus[col].isnull()

# Imputation
imputer = SimpleImputer()
imputed_X_train = imputer.fit_transform(imputed_X_train) #fit imputer and transform data
imputed_X_test = imputer.transform(imputed_X_test) #transform data
#####################################################################



######################_trainging_the_model_##########################
# finding max leaf nodes
# def get_MAE(max_leaf_nodes, training_X, predicting_X, training_y, predicting_values_y):
#     model=DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,random_state=0)
#     model.fit(training_X,training_y)
#     predicted_val_y=model.predict(predicting_X)
#     MAE=mean_absolute_error(predicting_values_y,predicted_val_y)
#     return (MAE)
# for max_leaf_nodes in [5,50,60,65,67,70,75,80,85,90]:
#     current_MAE=get_MAE(max_leaf_nodes,train_X,val_X,train_y,val_y)
#     print("max leaf nodes: %d \t\t Mean Absolute Error: %d" %(max_leaf_nodes,current_MAE))
# find the number of nodes for least MAE and create the model according to it

# model=DecisionTreeRegressor(max_leaf_nodes=75,random_state=0)  #model with specifying max leaf nodes
model=DecisionTreeRegressor()
model.fit(train_X,train_y)
#####################################################################



######################_evaluating_the_model_#########################
predicted_values_for_traininig_set=model.predict(test_X)

# print correct and predicted values
# for i in range(0, len(val_y)):
#     print (val_y.index[i],val_y.get(i),predicted_values_for_traininig_set[i])

# print mean absolute error
print("MAE:",mean_absolute_error(test_y,predicted_values_for_traininig_set))
#####################################################################



######################_prediction_for_required_data_#################
predcted_result=model.predict(test_X)
#####################################################################



######################_visualization_of_prediction_#################
# creating a csv
submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predcted_result})
submission.to_csv('submission.csv', index=False)

# creating txt file
# text_file = open("Output.txt", "w")
# text_file.write("writing_text")
# text_file.close()
#####################################################################