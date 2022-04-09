import pandas as pd
import numpy as np
import modules.missing_values_module as mvm
# import modules.cliques_filter_module as cliques 


class OutputCorrelationFilter():
    """
    Filters features based on correlation with target
    
    Attributes:
    ----------
    n_features: number of features to retain
    share_features: share of features to retain
    corr_metrics: method for correlation estimation (default: 'pearson')
    max_acceptable_correlation: maximum corr value to accept
    
    Methods:
    ----------
    fit: fits transformer on data
    transform: transforms given dataset
    """
    
    def __init__(
                self, n_features = False, 
                share_features = False,  
                max_acceptable_correlation = False,
                corr_metrics = "pearson"
                ):
        
        if (np.array([n_features, share_features, max_acceptable_correlation]) != False).sum() > 1:
            raise ValueError("Please, predefine ONE OF: acceptable correlation, share of features or number of features")
        if (np.array([n_features, share_features, max_acceptable_correlation]) != False).sum() == 0:
            raise ValueError("Please, choose either acceptable correlation, share of features or number of features")
            
        self.n_features = n_features
        self.share_features = share_features
        self.corr_metrics = corr_metrics
        self.max_acceptable_correlation = max_acceptable_correlation

        
    def fit(self, X, y):
        
        if self.share_features != False:
            self.n_features = round(len(X.columns) * share_features)
        
        if self.n_features == 0:
            self.n_features = 1
        
        corrs = X.apply( lambda column: column.corr(y, method = self.corr_metrics) )
        
        if self.max_acceptable_correlation != False:
            self.n_features = (corrs <= self.max_acceptable_correlation).sum()
            
        self.desired_names = corrs.sort_values(ascending = False).index[:self.n_features].tolist()
        
        return self
    
    def transform(self, X, y= None):
        
        return X[self.desired_names]


class MutualCorrelationFilter():
    """
    Filters features based on correlation with target
    
    Attributes:
    ----------
    n_features: number of features to retain
    share_features: share of features to retain
    corr_metrics: method for correlation estimation (default: 'pearson')
    max_acceptable_correlation: maximum corr value to accept
    categorical_variables: list of categorical features
    
    Methods:
    ----------
    fit: fits transformer on data
    transform: transforms given dataset
    """
    
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

    def fit(self, X, y = None):
        X_separated = SeparatedDF(X, self.categorical_variables, ignore_dummies = True)
        if self.share_features != False:
            self.n_features = round(len(X.columns) * self.share_features)
        
        if (self.n_features == 0)&(self.share_features != False):
            self.n_features = 1
        
        current_df = X_separated.X_numeric.copy()
        
        while len(current_df.columns) > self.n_features:
#             breakpoint()

            current_corr = current_df.corr(method = self.corr_metrics).unstack().sort_values(ascending = False)
            for i in range(len(current_corr)):
                column_names = current_corr.index[i]
                corr_value = current_corr.iloc[i]
                if column_names[0] != column_names[1]:
                    most_correlated_column = column_names[0]
                    break
                    
            if len(current_df.columns) == 1:
                break
                
            if np.abs(corr_value) <= self.max_acceptable_correlation:
                break
                
            current_df.drop(most_correlated_column, axis = 1, inplace = True)
        
        self.desired_names = current_df.columns.tolist()
        self.correlated_names = list(set(X.columns) - set(self.desired_names))
        
        return self
    
    def transform(self, X, y= None):
        X_reunited = pd.concat([X[self.desired_names], X[find_dummies(X)], X[self.categorical_variables]], axis = 1)
        X_reunited = X_reunited.loc[:,~X_reunited.columns.duplicated()]
        return X_reunited
        
        
class VIFFilter():
    """
    Filters features based on VIF
    
    Attributes:
    ----------
    acceptable_vif: max VIF acceptable
    desired_names: column names to retain
    
    Methods:
    ----------
    fit: fits transformer on data
    transform: transforms given dataset
    """
    
    def __init__(self, acceptable_vif = [5, 10]):
        
        self.acceptable_vif = acceptable_vif
        
    def fit(self, X, y = None):
        vifs = pd.Series(np.linalg.inv(X.corr().to_numpy()).diagonal(), 
                 index=X.columns, 
                 name='VIF')
        self.desired_names = vifs[(vifs <= self.acceptable_vif[0]) | (vifs >= self.acceptable_vif[1])].index.tolist()
        return self
    
    def transform(self, X, y = None):
        
        return X[self.desired_names]
    
    
# class CliquesFilter():
#     """
#     Filters features: maximum linearly independent subset of features by threshold
    
#     For additional information see [this notebook](https://notebooks.githubusercontent.com/view/ipynb?browser=chrome&color_mode=auto&commit=32ac73fafbdaf219ec8e6b5fb49a715dd54ceb1c&device=unknown&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f646d697472796f6b686f746e696b6f762f486162722f333261633733666166626461663231396563386536623566623439613731356464353463656231632f41727469636c655f6578706572696d656e74735f636f64652e6970796e62&logged_in=false&nwo=dmitryokhotnikov%2FHabr&path=Article_experiments_code.ipynb&platform=android&repository_id=428269178&repository_type=Repository&version=99}
    
#     Attributes:
#     ----------
#     acceptable_vif: max VIF acceptable
#     desired_names: column names to retain
    
#     Methods:
#     ----------
#     fit: fits transformer on data
#     transform: transforms given dataset
#     """
    
#     def __init__(self, trsh = 0.65):
#         self.trsh = 0.65
        
#     def fit(self, X, y):
#         qlq_list, G = cliques.get_noncollinear_fts(
#                                                     X, y, trsh=0.65, mode="all", verbose=False
#                                                     )
#         self.desired_names = qlq_list[list(qlq_list)[0]][0]
        
#     def transform(self, X, y = None):
        
#         return X[self.desired_names]
