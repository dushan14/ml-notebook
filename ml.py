import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# training data
train_file_path = '../input/train.csv'e
train_data = pd.read_csv(train_file_path)

# test data
test_file_path='../input/test.csv'
test_data = pd.read_csv(testing_file_path)

# all_cols=train_data.columns

# checking null cols
# train_data.isnull().any()

# after selecting cols for prediction 
predicting_cols=['LotArea', 'YearBuilt','1stFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']

X=house_data[predicting_cols]
y=train_data.ClassOfPredicting

test_X=test_data[predicting_cols]

# deviding training data for checking correctness
train_X,val_X,train_y,val_y=train_test_split(X,y,random_state=0)


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


# prediction for training set
predicted_values_for_traininig_set=model.predict(val_X)

# print correct and predicted values
# for i in range(0, len(val_y)):
#     print (val_y.index[i],val_y.get(i),predicted_values_for_traininig_set[i]) 

# print mean absolute error
print("MAE:",mean_absolute_error(val_y,predicted_values_for_traininig_set))


# getting prediction for test data
predcted_result=model.predict(test_X)

# creating a csv
submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predcted_result})
submission.to_csv('submission.csv', index=False)

# creating txt file
# text_file = open("Output.txt", "w")
# text_file.write("writing_text")
# text_file.close()