import time
import json
import numpy as np
import pandas as pd
import sklearn
from hyperopt import hp
from collections import OrderedDict
import matplotlib.pyplot as plt
from modules import defaults
from importlib import reload 

from modules.utils import gini_score
# from modules.pipeline_experimental import PipeHPOpt

from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials
from modules.missing_values_module import teach_to_separate

import modules.feature_filters as filters
import local_modules.yield_preprocessing as prep

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from modules import pipeline as pe

seed = 42

def check_constant_column(column):
    if len(column.unique()) == 1:
        return True
    else:
        return False
    
def return_non_constant_table(table):
    constantness_of_columns = (table.apply(check_constant_column, axis = 0) == False)
    constant_columns = table.columns[constantness_of_columns]
    return table[constant_columns]

def return_constant_columns(table):
    non_constantness_of_columns = table.apply(check_constant_column, axis = 0)
    return np.array(table.columns[non_constantness_of_columns])

def find_dummies(X):
    count_values = X.apply(lambda x: len(x.unique()), axis = 0)
    dummy_mask = X.apply(lambda x: len(x.unique()), axis = 0) <= 2
    dummy_columns = count_values[dummy_mask].index.tolist()
    return dummy_columns

def new_teach_to_separate(parent_class):
    """
    Returns an object, which does exactly the same as the parent class 'parent_transformer', but before doing so separates the data into numerical and categorical columns and applies the transformation only to numerical ones. It also returns DataFrame, even when the parent class by default returns, for example, a 3-d numpy array and renames principal components, if this approach is used like "PC1", "PC2" etc.
    Example of usage: 
    >KNNImputerSeparated = teach_to_separate(KNNImputer)
    >obj = KNNImputerSeparated(categorical_variables = ["xcat"], n_neighbors = 6)
    >test_df = create_test_df()
    >obj.fit(test_df)
    >obj.transform(test_df)
    
    parent_class::class A parent class. Must have fit and transform attributes.
    
    Variables provided when instance is created:
    
    categorical_variables::list A list or an array, which contains a list of variables which must be separated and to which the transformation will not be applied
    
    functionality::str A string which defines what type of transformer class you are modifying. Currently there are option 'imputer' for all the imputers and 'PCA' for sklearn.PCA. 
    
    ignore_dummies::Bool True to treat 0,1 variables as categorical
    **kwargs Any arguments passed to parent class itself
    
    """

    
    class SeparatedDF():
        def __init__(self, X, categorical_variables = [], ignore_dummies = False):
            if (categorical_variables == [])&(ignore_dummies == True):
                categorical_variables = find_dummies(X)
            self.X_numeric = X.drop(categorical_variables, 1)
            self.X_categorical = X.copy()[categorical_variables]
            
    class ClassSeparated():

        def __init__(self, categorical_variables, functionality = "imputer", ignore_dummies = False, **kwargs):
            self.kwargs = kwargs
            self.obj = parent_class(**self.kwargs)
            self.categorical_variables = categorical_variables
            self.ignore_dummies = ignore_dummies
            if functionality not in ["imputer", "PCA"]:
                print(f"You inputed functionality {functionality}, but currently only\n 'imputer' and 'PCA' are implemented")
            self.functionality = functionality

        def fit(self, X, y = None):
    #         print("You are here 1")
#             print(list(X.columns))
            self.old_columns = X.columns
            self.categorical_variables = set(self.categorical_variables).intersection(set(X.columns))
            self.categorical_variables = list(self.categorical_variables)
            df = SeparatedDF(X, self.categorical_variables, self.ignore_dummies)
    #         print(df.X_numeric)
#             print(self.categorical_variables)
            if (len(self.categorical_variables) > 0):
                self.categorical_variables = set(self.categorical_variables + list(df.X_categorical.columns))
                self.categorical_variables = list(self.categorical_variables)
            else: 
                self.categorical_variables = list(df.X_categorical.columns)
            self.check = "I fitted the object, I swear"
#             print(df.X_numeric)
            self.obj.fit(df.X_numeric)
            return self

        def transform(self, X, y = None):
    #         print("You are here 2")
    #         print(self.check)
#             print(list(X.columns))
#             print(X.columns[~np.isin(self.old_columns, X.columns)])
            df = SeparatedDF(X, self.categorical_variables)
#             print(df.X_categorical)
#             print(df.X_numeric)
            if hasattr(self.obj, "transform"):
                fitted_df = self.obj.transform(df.X_numeric)
            elif hasattr(self.obj, "fit_transform"):
                fitted_df = self.obj.fit_transform(df.X_numeric)
    #         print(self.obj.__dict__)

            fitted_df = pd.DataFrame(fitted_df)
            if self.functionality == "imputer":
                fitted_df.columns = df.X_numeric.columns
            elif self.functionality == "PCA":
                if self.obj.n_components is None:
    #                 print(df.X_numeric.columns)
                    self.obj.n_components = len(df.X_numeric.columns)
                fitted_df.columns = [f"PC{i}" for i in range(1, self.obj.n_components +1)]
    #         print(fitted_df)
            fitted_df = pd.concat([fitted_df, df.X_categorical], axis = 1)
    #         print(df.X_categorical)
    #         print(fitted_df)
            return fitted_df
    return ClassSeparated

class SeparatedDF():
    def __init__(self, X, categorical_variables = [], ignore_dummies = False):
        if (categorical_variables == [])&(ignore_dummies == True):
            categorical_variables = find_dummies(X)
        self.X_numeric = X.drop(categorical_variables, 1)
        self.X_categorical = X.copy()[categorical_variables]

class new_OutputCorrelationFilter():


    def __init__(
                self, n_features = False, 
                share_features = False,  
                max_acceptable_correlation = False,
                corr_metrics = "pearson",
                categorical_variables = []
                ):
        
        if (np.array([n_features, share_features, max_acceptable_correlation]) != False).sum() > 1:
            raise ValueError("Please, predefine ONE OF: acceptable correlation, share of features or number of features")
        if (np.array([n_features, share_features, max_acceptable_correlation]) != False).sum() == 0:
            raise ValueError("Please, choose either acceptable correlation, share of features or number of features")
            
        self.n_features = n_features
        self.share_features = share_features
        self.corr_metrics = corr_metrics
        self.max_acceptable_correlation = max_acceptable_correlation
        self.categorical_variables = categorical_variables
    def fit(self, X, y):
#         print("I am fitting Ofilter")
        
        X_separated = SeparatedDF(X, self.categorical_variables, ignore_dummies = True)
        if self.share_features != False:
            self.n_features = round(len(X_separated.X_numeric.columns) * share_features)
        if self.n_features == 0:
            self.n_features = 1
        corrs = X_separated.X_numeric.apply( lambda column: column.corr(y, method = self.corr_metrics) )
        if self.max_acceptable_correlation != False:
            self.n_features = (corrs <= self.max_acceptable_correlation).sum()
        self.desired_names = corrs.sort_values(ascending = False).index[:self.n_features].tolist()
#         print("Desired names from filter:")
#         print(self.desired_names)
        
        return self
    
    def transform(self, X, y= None):
        X_reunited = pd.concat([X[self.desired_names], X[find_dummies(X)]], axis = 1)

        return X_reunited
# obj = new_OutputCorrelationFilter(n_features = 5, categorical_variables = ["Moodys", "ExpertRA"])
# obj.fit(X_fallen.dropna(), y_fallen)
# obj.transform(X_fallen)



filters = reload(filters)
class IndexReseter():
    
    def __init__(self):
        pass
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        return X.reset_index(drop = True)
    
IndexReset = IndexReseter()


class StandardScalerPandas():
    
    def __init__(self):
        pass
    
    def fit(self, X, y = None):
#         print("I am fitting scaler")
        self.column_names = X.columns
        fitter = StandardScaler()
        self.fitted = fitter.fit(X)
#         print("Scaler output:")
#         print(self.fitted)
        return self
    
    def transform(self, X, y = None):
        transformed = pd.DataFrame(self.fitted.transform(X))
        transformed.columns = self.column_names
        return transformed
    
class averager():
    
    def __init__(self):
        pass
    
    def fit(self, X, y):
        X["support"] = y
        self.map_table = prep.grouper(X, "support", verbose = False)["map_table"]
#         supports_train["support"] = list(m)
        X.drop("support", inplace = True, axis = 1)
        return self
        
    def transform(self, X):
        merged_table = X.merge(self.map_table, on = ["Year", "Month", "Moodys"], how = "left")
        return merged_table
    
averager_object = averager()

StScal = StandardScalerPandas()


Basic = LinearRegression(n_jobs = -1)

KNN = KNeighborsRegressor(n_neighbors=5,
                           weights='uniform',
                           algorithm='auto',
                           leaf_size=30,
                           p=2,
                           metric='minkowski',
                           metric_params=None)
SVM = SVR(kernel='rbf',
          degree=3,
          gamma='scale',
          coef0=0.0,
          C=1.0, 
          epsilon=0.1,
          shrinking=True)

RF = RandomForestRegressor(n_estimators=100, 
                           max_depth=None,
                           min_samples_split=2,
                           min_samples_leaf=1,
                           min_weight_fraction_leaf=0.0,
                           max_features='auto',
                           max_leaf_nodes=None,
                           min_impurity_decrease=0.0,
                           bootstrap=True, 
                           oob_score=False, 
                           n_jobs=-1,
                           ccp_alpha=0.0)

catboost = CatBoostRegressor(verbose=False)



PCASeparated = new_teach_to_separate(sklearn.decomposition.PCA)
PCAer = PCASeparated(categorical_variables = [], #параметр для отделения категориальных переменных, добавленный дочерним классом
                     functionality = "PCA",
                    ignore_dummies = True,
                    n_components = 2)


kPCASeparated = new_teach_to_separate(sklearn.decomposition.KernelPCA)
kPCAer = PCASeparated(categorical_variables = [], #параметр дочернего класса
                      functionality = "PCA", #параметр дочернего класса
                    ignore_dummies = True,
                      n_components = 2
                    )


lgbm = LGBMRegressor()

filters = reload(filters)
OFilter = filters.OutputCorrelationFilter(n_features = 30)
MFilter = filters.MutualCorrelationFilter(n_features = 30)
VIFFilter = filters.VIFFilter()
CFilter = filters.CliquesFilter(trsh = 0.65)


pipe_modules = defaults.get_default_modules()
pipe_modules["Basic"] = Basic
pipe_modules["KNN"] = KNN
pipe_modules["SVM"] = SVM
pipe_modules["RF"] = RF
pipe_modules["catboost"] = catboost
pipe_modules["StScal"] = StScal
pipe_modules["PCA"] = PCAer
pipe_modules["kPCA"] = kPCAer
pipe_modules["OFilter"] = OFilter
pipe_modules["MFilter"] = MFilter
pipe_modules["lgbm"] = lgbm
pipe_modules["Averagermod"] = averager_object
# pipe_modules["CFilter"] = CFilter
pipe_modules["IndexReseter"] = IndexReset


pipe_params = OrderedDict()

pipe_params['IndexReset1'] = hp.choice("IndexReset1", ["IndexReseter"])
pipe_params['Missing'] = hp.choice('Missing', ['MeanImp'])
# pipe_params['Dummies'] = hp.choice('Dummies', ['OneHot'])
# pipe_params['PrimaryFilter'] = hp.choice('PrimaryFilter', ['skip', 'VIFFilter'])
pipe_params['SecondaryFilter'] = hp.choice('SecondaryFilter', ['skip', 'MFilter'])
pipe_params['Scale'] = hp.choice("Scale", ["StScal", "skip"])
pipe_params['Transform'] = hp.choice("Transform", ["skip", "PCA", "kPCA"])
pipe_params['Model']    = hp.choice("Model", ["Basic", "KNN", "SVM", "RF", "catboost", "lgbm"])

pipe_params_averager = OrderedDict()

pipe_params_averager['IndexReset1'] = hp.choice("IndexReset1", ["IndexReseter"])
pipe_params_averager['Averager'] = hp.choice('Averager', ['Averagermod'])
pipe_params_averager['Missing'] = hp.choice('Missing', ['MeanImp'])
# pipe_params['Dummies'] = hp.choice('Dummies', ['OneHot'])
# pipe_params['PrimaryFilter'] = hp.choice('PrimaryFilter', ['skip', 'VIFFilter'])
pipe_params_averager['SecondaryFilter'] = hp.choice('SecondaryFilter', ['skip', 'MFilter'])
pipe_params_averager['Scale'] = hp.choice("Scale", ["StScal", "skip"])
pipe_params_averager['Transform'] = hp.choice("Transform", ["skip", "PCA", "kPCA"])
pipe_params_averager['Model']    = hp.choice("Model", ["Basic", "KNN", "SVM", "RF", "catboost", "lgbm"])

pipe_params

set_params  = {
#     "Restrict_Limiter__desired_columns" : hp.choice("Limiter__desired_columns", [best_linear_columns, best_knn_columns]),
#     "Limiter__number_of_columns" : hp.choice("Limiter__number_of_columns", [10, 15, 25, 30]),
    "Transform_PCA__n_components" : hp.quniform("PCA__n_components", low = 30, high = 90, q = 30),
    "Transform_PCA__whiten" : hp.choice("PCA__whiten", [True, False]),
    "Transform_PCA__categorical_variables" : hp.choice("PCA__categorical_variables", [["Year", "Month", "Moodys", "support"], ["Year", "Month", "Moodys"], ["support"], []]),
    "Transform_kPCA__categorical_variables" : hp.choice("kPCA__categorical_variables", [["Year", "Month", "Moodys", "support"], ["Year", "Month", "Moodys"], ["support"], []]),
    "Transform_kPCA__n_components" : hp.quniform("kPCA__n_components", low = 30, high = 90, q = 30),
    "Transform_kPCA__kernel" : hp.choice("kPCA__kernel", ["poly", "rbf"]),
    "Transform_kPCA__degree" : hp.quniform("kPCA__degree", low = 1, high = 3, q = 1),
    "Transform_UMAP__n_components" : hp.quniform("UMAP__n_components", low =10, high = 30, q = 20),
    "Transform_UMAP__n_neighbours" : hp.quniform("UMAP__n_neighbours",  low = 2, high = 10, q = 2),
    "Transform_UMAP__min_dist" : hp.choice("UMAP__min_dist", [0.05, 0.1,0.5, 1]),
#     "SecondaryFilter_OFilter__n_features" : hp.quniform("OFilter__n_features", low = 90, high = 130, q = 40),
#     "SecondaryFilter_OFilter__corr_metrics" : hp.choice("OFilter__corr_metrics", ["pearson", "spearman"]),
#     "SecondaryFilter_OFilter__corr_metrics" : hp.choice("MFilter__corr_metrics", ["pearson", "spearman"]),
    "SecondaryFilter_MFilter__n_features" : hp.quniform("MFilter__n_features", low = 90, high = 130, q = 40),
    "SecondaryFilter__categorical_variables" : hp.choice("SecondaryFilter__categorical_variables", [["Year", "Month", "Moodys", "support"], ["Year", "Month", "Moodys"], ["support"], []]),
    "Model_KNN__n_neighbours" : hp.choice("KNN__n_neighbours", [2, 5, 10, 50]),
    "Model_KNN__weights" : hp.choice("KNN__weights", ["uniform", "distance"]),
    "Model_KNN__leaf_size" : hp.choice("KNN__leaf_size", [10, 20, 30, 50]),
    "Model_KNN__p" : hp.choice("KNN__p", [1, 2]),
    "Model_SVM__kernel" : hp.choice("SVR__kernel", ["poly", "rbf", "sigmoid"]),
    "Model_SVM__degree" : hp.quniform("SVR__degree", low = 1, high = 3, q = 1),
    "Model_SVM__C" : hp.choice("SVR__C", [0.01, 0.1, 1, 10]),
    "Model_RF__criterion" : hp.choice("RF__criterion", ["squared_error", "poisson"]),
    "Model_RF__max_depth" : hp.choice("RF__max_depth", [None, 10, 20, 40, 60]),
    "Model_lgbm__max_depth" : hp.quniform("lgbm__max_depth", low = 2, high = 20, q = 2),
    "Model_lgbm__min_data_in_leaf" : hp.quniform("lgbm__data_in_leaf", low = 100, high = 2000, q = 500),
    "Model_lgbm__num_leaves" : hp.quniform("lgbm__num_leaves", low = 10, high = 200, q = 10)
        }


pipe_para = dict()
pipe_para['pipe_params']    = pipe_params
pipe_para['set_params']     = set_params
pipe_para['loss_func']      = lambda y, pred: mean_absolute_error(y, pred)

pipe_para_averager = dict()
pipe_para_averager['pipe_params']    = pipe_params_averager
pipe_para_averager['set_params']     = set_params
pipe_para_averager['loss_func']      = lambda y, pred: mean_absolute_error(y, pred)
