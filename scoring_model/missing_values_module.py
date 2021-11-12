# модуль с разными полезными вспомогательными функциями и самодельными классами для missing values imputation

import numpy as np
import pandas as pd

def create_test_df(nans = True):
    x1 = np.random.normal(size = 10000)
    x2 = np.random.normal(size = 10000)*x1
    x3 = np.random.normal(size = 10000) + x2
    if nans == True:
        x4 = np.random.choice(a = [1, 0], size = 10000)
        xcat = np.random.choice(a = ["a", "b", "c", "d", None], size = 10000)
    else:
        x4 = np.random.choice(a = [1], size = 10000)
        xcat = np.random.choice(a = ["a", "b", "c", "d", "e"], size = 10000)
    target = np.random.choice(a = [0,1], size = 10000)
    x5 = np.random.normal(size = 10000)

    test_df = pd.DataFrame({"x1" : x1,
                           "x2" : x2,
                           "x3" : x3,
                           "x4" : x4,
                            "xcat" : xcat,
                            "target" : target
                           } )
    
    test_df.x4 = np.where(test_df.x4 == 1, x5, None)
    return test_df

def teach_to_separate(parent_transformer):
    """
    Returns an object, which does exactly the same as the parent class 'parent_transformer', but before doing so separates the data into numerical and categorical columns and applies the transformation only to numerical ones. It also returns DataFrame, even when the parent class by default returns, for example, a 3-d numpy array and renames principal components, if this approach is used like "PC1", "PC2" etc.
    Example of usage: 
    >KNNImputerSeparated = teach_to_separate(KNNImputer)
    >obj = KNNImputerSeparated(categorical_variables = ["xcat"], n_neighbors = 6)
    >test_df = create_test_df()
    >obj.fit(test_df)
    >obj.transform(test_df)
    
    parent_transformer::class A parent class. Must have fit and transform attributes.
    
    Variables provided when instance is created:
    
    categorical_variables::list A list or an array, which contains a list of variables which must be separated and to which the transformation will not be applied
    
    functionality::str A string which defines what type of transformer class you are modifying. Currently there are option 'imputer' for all the imputers and 'PCA' for sklearn.PCA. 
    
    **kwargs Any arguments passed to parent class itself
    
    """
    
    class SeparatedDF():
        def __init__(self, X, categorical_variables = []):
            self.X_numeric = X.drop(categorical_variables, 1)
            self.X_categorical = X.copy()[categorical_variables]

    class ImputerSeparated(parent_transformer):

        def __init__(self, categorical_variables, functionality = "imputer", **kwargs):
            super().__init__(**kwargs)
            self.categorical_variables = categorical_variables
            self.obj = parent_transformer(**kwargs)
            if functionality not in ["imputer", "PCA"]:
                print(f"You inputed functionality {functionality}, but currently only\n 'imputer' and 'PCA' are implemented")
            self.functionality = functionality
            

        def fit(self, X, y = None):

            df = SeparatedDF(X, self.categorical_variables)
            self.obj.fit(df.X_numeric)

            return self

        def transform(self, X, y = None):

            df = SeparatedDF(X, self.categorical_variables)
            
            fitted_df = self.obj.transform(df.X_numeric)
            fitted_df = pd.DataFrame(fitted_df)

            if self.functionality == "imputer":
                fitted_df.columns = df.X_numeric.columns
            elif self.functionality == "PCA":
                fitted_df.columns = [f"PC{i}" for i in range(1, self.obj.n_components+1)]
            fitted_df = pd.concat([fitted_df, df.X_categorical], axis = 1)

            return fitted_df
    return ImputerSeparated


def assert_identical_results_separated(imputer_class_basic,
                                       imputer_class_modified,
                                       categorical_variables, 
                                       test_df = create_test_df()):

    obj = imputer_class_modified(categorical_variables)
    obj.fit(test_df)
    child_fitted = obj.transform(test_df)
    child_fitted = child_fitted.drop(categorical_variables, axis = 1)
    print(child_fitted)
    obj = imputer_class_basic()
    test_df_local = test_df.drop(categorical_variables,1)
    obj.fit(test_df_local)
    parent_fitted = obj.transform(test_df_local)
    parent_fitted = pd.DataFrame(parent_fitted,
                                 columns = child_fitted.columns)
    print(parent_fitted)
    assert child_fitted.equals(parent_fitted)
    
class missing_filler_category():
    
    """
    Returns a table with all the missing values filled with special category.
    params::x DataFrame with the data
    params::filling_category a value to fill the missing values
    """
    
    def __init__(self,  filling_category = "filler"):
        self.filling_category = filling_category
        
    def fit(self, x, y = None):
        self.x = x.copy()
        return self
    
    def transform(self, x, y = None):
        self.x = x.copy()
        self.x.fillna(self.filling_category, inplace = True)
        filled_table = self.x
        return filled_table
    
    
class missing_filler_mean():
    
    """
    Returns a table with all the missing values in numerical columns filled with mean and all the 
    missing values in categorical columns filled with special category. If you do not wish to fill
    categorical at all, just do not pass "categorical_variables" parameter.
    params::x DataFrame with the data
    params::categorical_variables list of categorical varaibles
    params::filling_category a value to fill the missing values in categorical variables
    """
    
    def __init__(self, categorical_variables = [], filling_category = "filler", y = None):
        self.categorical_variables = categorical_variables
        self.filling_category = filling_category
        
    def fit(self, x, y= None):
        self.x = x.copy()
        self.mean_to_fill = self.x.mean()
        return self
    
    def transform(self, x, y= None):
        self.x = x.copy()
        filled_table = self.x.copy()
        if len(self.categorical_variables) != 0:
            filled_table[self.categorical_variables] = \
            filled_table[self.categorical_variables].fillna(self.filling_category)
            
        filled_table = filled_table.fillna(self.mean_to_fill)

        return filled_table
    
    
    
class missing_filler_median():
    
    """
    Returns a table with all the missing values in numerical columns filled with median and all the 
    missing values in categorical columns filled with special category. If you do not wish to fill
    categorical at all, just do not pass "categorical_variables" parameter.
    params::x DataFrame with the data
    params::categorical_variables list of categorical varaibles
    params::filling_category a value to fill the missing values in categorical variables
    """
    
    def __init__(self, categorical_variables = [], filling_category = "filler"):

        self.categorical_variables = categorical_variables

        self.filling_category = filling_category
        
    def fit(self, x, y = None):
        self.x = x.copy()
        self.median_to_fill = self.x.median()
        return self
    
    def transform(self, x, y = None):
        self.x = x.copy()
        filled_table = self.x.copy()
        
        if len(self.categorical_variables) != 0:
            
            filled_table[self.categorical_variables] = \
            filled_table[self.categorical_variables].fillna(self.filling_category)
            
        filled_table = \
        filled_table.fillna(self.median_to_fill)
            
        return filled_table
    
    
class missing_filler_mode():
    
    """
    Returns a table with all the missing values in numerical columns filled with mode. If there are several 
    modes, the behaviour is the foolowing:
    1) For categorical variables from the input, the first element of list of modes is used
    2) For numerical variables, the mean of modes is used
    
    params::x DataFrame with the data
    params::categorical_variables list of categorical varaibles
    """
    
    def __init__(self,categorical_variables = [], filling_category = "filler"):
        self.filling_category = filling_category
        self.categorical_variables = categorical_variables

    def fit(self, x, y = None):
        return self
    
    def transform(self, x, y = None):
        self.x = x.copy()
        if (type(self.categorical_variables) == list):
            self.non_categorical_variables = \
            list(set(self.categorical_variables).symmetric_difference(list(x.columns)))
        else:
            all_variables = list(self.x.columns).copy()
            all_variables.remove(self.categorical_variables)
            self.non_categorical_variables = \
            all_variables
            
        table_to_fill = self.x.copy()
        categorical_table = table_to_fill[self.categorical_variables].copy()
        non_categorical_table = table_to_fill[self.non_categorical_variables].copy()

        if len(self.categorical_variables) == 0:
            categorical_table = pd.DataFrame(index = non_categorical_table.index())
        
        table_to_fill[self.categorical_variables] = \
        categorical_table.fillna(categorical_table.mode().loc[0])
        
        table_to_fill[self.non_categorical_variables] = \
        non_categorical_table.fillna(non_categorical_table.mode().mean())
        
        return table_to_fill