# import the required packages
import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier
from xgboost.sklearn import XGBRegressor

# scores
from sklearn.metrics import mean_squared_error,mean_squared_log_error,confusion_matrix,accuracy_score,roc_auc_score, balanced_accuracy_score

class Model(object):
    def _classify_fit(self, X, y):
        self.classif_model = XGBClassifier(
         learning_rate =0.1,
         n_estimators=80,
         max_depth=5,
         min_child_weight=1,
         gamma=0.8,
         reg_alpha = 1.0,
         reg_lambda = 1.0,
         subsample=0.9,
         colsample_bytree=0.8,
         objective= 'binary:logistic',
         nthread=4,
         scale_pos_weight=1,
         seed=27).fit(X, y)
    
    def _regress_fit(self, X, y):
        self.regress_model = XGBRegressor(
            learning_rate =0.1,
            n_estimators=1000,
            max_depth=7,
            min_child_weight=10,
            gamma=0,
            reg_alpha = 1.0,
            reg_lambda = 0.5,
            subsample=0.9,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            nthread=4,
            scale_pos_weight=1,
            seed=27).fit(X,y,eval_metric="rmse")
    
    def fit(self, X, y_class, y_reg):
        self._classify_fit(X, y_class)   
        self._regress_fit(X, y_reg)
        return self
    
    def predict(self, X):
        X_copy = X.copy()
        
        # step 1
        y_conv = self.classif_model.predict(X_copy)
        X_copy.loc[:,"conv"] = y_conv

        # separate out the ones and zeroes
        X_copy_1 = X_copy[X_copy.conv == 1.0]
        X_copy_1 = X_copy_1.drop(["conv"], axis=1)
        X_copy_0 = X_copy[X_copy.conv == 0.0]
        if not X_copy_0.empty:
            X_copy_0.loc[:,"revenue"] = 0.0

        # step 2
        y_rev = self.regress_model.predict(X_copy_1)
        if not X_copy_1.empty:
            X_copy_1.loc[:, "revenue"] = y_rev

        
        # merge data
        X_new = pd.concat([X_copy_0, X_copy_1], axis=0)

        return y_conv, X_new.revenue
    
    def score(self, y_true_conv, y_pred_conv, y_true_revenue, y_pred_revenue):
        metrics = {}
        
        metrics["conv_balanced_accuracy"] = balanced_accuracy_score(y_true_conv, y_pred_conv)
        metrics["revenue_rmse"] = mean_squared_error(y_true_revenue, y_pred_revenue) ** 0.5
        
        response = pd.DataFrame({"y_true_conv": y_true_conv, "y_pred_conv": y_pred_conv, "y_true_revenue": y_true_revenue, "y_pred_revenue": y_pred_revenue})
        
        metrics["effort"] = ((sum(response[(response.y_true_conv == 0.0) & (response.y_pred_conv == 1.0)].y_pred_revenue))/ sum(y_true_revenue)) * 100
        
        metrics["loss"] = ((sum(response[(response.y_true_conv == 1.0) & (response.y_pred_conv == 0.0)].y_true_revenue)) / sum(y_true_revenue)) * 100
        
        metrics["total_true_revenue"] = sum(y_true_revenue)
        
        return metrics