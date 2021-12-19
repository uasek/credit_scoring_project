class missing_filler_mode():
    
    """
    Returns a table with all the missing values in numerical columns filled with mode. 
    If there are several modes, the behaviour is the foolowing:
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
    
    
def teach_to_separate(parent_class):
    """
    Returns an object, which does exactly the same as the parent class 'parent_transformer', 
    but before doing so separates the data into numerical and categorical columns and applies 
    the transformation only to numerical ones. It also returns DataFrame, even when the parent 
    class by default returns, for example, a 3-d numpy array and renames principal components, 
    if this approach is used like "PC1", "PC2" etc.
    
    Example of usage: 
    >KNNImputerSeparated = teach_to_separate(KNNImputer)
    >obj = KNNImputerSeparated(categorical_variables = ["xcat"], n_neighbors = 6)
    >test_df = create_test_df()
    >obj.fit(test_df)
    >obj.transform(test_df)
    
    parent_class::class A parent class. Must have fit and transform attributes.
    
    Variables provided when instance is created:
    
    categorical_variables::list A list or an array, which contains a list of variables which must 
    be separated and to which the transformation will not be applied
    
    functionality::str A string which defines what type of transformer class you are modifying. 
    Currently there are option 'imputer' for all the imputers and 'PCA' for sklearn.PCA. 
    
    **kwargs Any arguments passed to parent class itself
    
    """
    
    class SeparatedDF():
        def __init__(self, X, categorical_variables = []):
            self.X_numeric = X.drop(categorical_variables, 1)
            self.X_categorical = X.copy()[categorical_variables]
    class ClassSeparated():

        def __init__(self, categorical_variables, functionality = "imputer", **kwargs):
            self.kwargs = kwargs
            self.obj = parent_class(**self.kwargs)
            self.categorical_variables = categorical_variables
            if functionality not in ["imputer", "PCA"]:
                print(f"You inputed functionality {functionality}, but currently only\n 'imputer' and 'PCA' are implemented")
            self.functionality = functionality

        def fit_transform(self, X, y = None):
            if hasattr(self.obj, "fit_transform"):
                df = SeparatedDF(X, self.categorical_variables)
                fitted_df = self.obj.fit_transform(df.X_numeric)
                fitted_df = pd.DataFrame(fitted_df)
                if self.functionality == "imputer":
                    fitted_df.columns = df.X_numeric.columns
                elif self.functionality == "PCA":
                    if self.obj.n_components is None:
                        self.obj.n_components = len(df.X_numeric.columns)
                    fitted_df.columns = [f"PC{i}" for i in range(1, self.obj.n_components +1)]
                fitted_df = pd.concat([fitted_df, df.X_categorical], axis = 1)
                return fitted_df
            else:
                pass
        def fit(self, X, y = None):
            df = SeparatedDF(X, self.categorical_variables)
            self.check = "I fitted the object, I swear"
            self.obj.fit(df.X_numeric)
            return self

        def transform(self, X, y = None):
            df = SeparatedDF(X, self.categorical_variables)
            if hasattr(self.obj, "transform"):
                fitted_df = self.obj.transform(df.X_numeric)
            elif hasattr(self.obj, "fit_transform"):
                fitted_df = self.obj.fit_transform(df.X_numeric)

            fitted_df = pd.DataFrame(fitted_df)
            if self.functionality == "imputer":
                fitted_df.columns = df.X_numeric.columns
            elif self.functionality == "PCA":
                if self.obj.n_components is None:
                    self.obj.n_components = len(df.X_numeric.columns)
                fitted_df.columns = [f"PC{i}" for i in range(1, self.obj.n_components +1)]
            fitted_df = pd.concat([fitted_df, df.X_categorical], axis = 1)
            return fitted_df
    return ClassSeparated

    