# import the required packages
import pandas as pd
import numpy as np

# preprocessing
from sklearn.preprocessing import Imputer
from sklearn.utils import resample

# feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

# ml libraries
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
from xgboost.sklearn import XGBRegressor

# scores
from sklearn.metrics import mean_squared_error,mean_squared_log_error,confusion_matrix,accuracy_score,roc_auc_score, balanced_accuracy_score

import pickle

from model import Model

class Preprocessing(object):
    def __init__(self, test_size=0.2, random_state=4):
        self.test_size = test_size
        self.random_state = random_state
    
    def transform(self, X):
        # get numeric and categorical columns
        categorical_columns = []
        numeric_columns = []
        for c in X.columns:
            if X[c].map(type).eq(str).any(): #check if there are any strings in column
                categorical_columns.append(c)
            else:
                numeric_columns.append(c)

        # create two DataFrames - categorical and numerical 
        data_numeric = X[numeric_columns]
        data_categorical = pd.DataFrame(X[categorical_columns])
        
        # impute missing values
        imp = Imputer(missing_values=np.nan, strategy='median', axis=0)
        data_numeric = pd.DataFrame(imp.fit_transform(data_numeric), columns = data_numeric.columns) #only apply imputer to numeric columns

        # no missing values in the categorical features as per the initial investigation 
        # join the two masked dataframes back together
        data_joined = pd.concat([data_numeric, data_categorical], axis = 1)
        
        
        data_joined.num_employees = data_joined.num_employees.replace({"None":0,"1":1,"2to5":4,"6to10":8,"11to50":32,"50plus":60})
        data_joined.num_purchases_year = data_joined.num_purchases_year.replace({'1to2':1, '25plus':32, '3to5':4, '11to25':16, 'None':0, '6to10':8})
        data_joined.cost_purchases_year = data_joined.cost_purchases_year.replace({'lessthan1':1, '25to100':64, '1to5':4, '5to25':16, 'None':0, '100plus':126})
        
        onehot_data = pd.get_dummies(data_joined, drop_first=True)
        
        X = onehot_data
        y = onehot_data.convert_30
        X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        
        # upsampling in training set
        df_majority = X_train_class[X_train_class.convert_30==0]
        df_minority = X_train_class[X_train_class.convert_30==1]
 
        df_minority_upsampled = resample(df_minority, 
                                 replace=True,    
                                 n_samples=20179, 
                                 random_state=12) 
        df_upsampled = pd.concat([df_majority, df_minority_upsampled])
        X_train_balance = df_upsampled.drop(['convert_30'], axis=1)
        y_train_balance = df_upsampled.convert_30
        
        self.y_train_revenue = X_train_balance["revenue_30"]
        self.y_test_revenue = X_test_class["revenue_30"]
        
        X_train_balance = X_train_balance.drop(['cuid','revenue_30','Unnamed: 0'], axis=1)
        X_test_class = X_test_class.drop(['cuid','revenue_30','Unnamed: 0'], axis=1)
        
        X_test_class = X_test_class.drop("convert_30",axis=1)
        
        self.X_train = X_train_balance
        self.X_test = X_test_class
        self.y_train_conv = y_train_balance
        self.y_test_conv = y_test_class
        
        return self

class FeatureSelection(object):
    def __init__(self, selection_type = "chi", chi_k=30):
        self.selection_type = selection_type
        self.chi_k = chi_k
    
    def _chi(self, X, y):
        X_norm = MinMaxScaler().fit_transform(X)
        chi_selector = SelectKBest(chi2, k=self.chi_k)
        chi_selector.fit(X_norm, y)
        chi_support = chi_selector.get_support()
        chi_feature = X.loc[:,chi_support].columns.tolist()
        return chi_feature

    def fit(self, X, y):
        if self.selection_type == "chi":
            return self._chi(X, y)

class FeatureSelection(object):
    def __init__(self, selection_type = "chi", chi_k=30):
        self.selection_type = selection_type
        self.chi_k = chi_k
    
    def _chi(self, X, y):
        X_norm = MinMaxScaler().fit_transform(X)
        chi_selector = SelectKBest(chi2, k=self.chi_k)
        chi_selector.fit(X_norm, y)
        chi_support = chi_selector.get_support()
        chi_feature = X.loc[:,chi_support].columns.tolist()
        return chi_feature
        
    def fit(self, X, y):
        if self.selection_type == "chi":
            return self._chi(X, y)

df_training = pd.read_csv("df_training_scholarjet.csv")

preprocess = Preprocessing()

data = preprocess.transform(df_training)


fs = FeatureSelection()

chi_features = fs.fit(data.X_train, data.y_train_conv)

if __name__ == "__main__":
  m = Model()

  model = m.fit(data.X_train[chi_features], data.y_train_conv, data.y_train_revenue)

  pkl_filename = "pickle_model.pkl"
  with open(pkl_filename, 'wb') as file_model:
    pickle.dump(model, file_model)

