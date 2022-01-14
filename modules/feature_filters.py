import pandas as pd
import numpy as np
import modules.missing_values_module as mvm
import modules.cliques_filter_module as cliques 


class OutputCorrelationFilter():
    
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
        
    def fit(self, X, y = None):
        
        if self.share_features != False:
            self.n_features = round(len(X.columns) * self.share_features)
        
        if (self.n_features == 0)&(self.share_features != False):
            self.n_features = 1
        
        current_df = X.copy()
        while len(current_df.columns) > self.n_features:
            current_corr = current_df.corr(method = self.corr_metrics).unstack().sort_values(ascending = False)
            for i in range(len(current_corr)):
                column_names = current_corr.index[i]
                corr_value = current_corr.iloc[i]
                if column_names[0] != column_names[1]:
                    most_correlated_column = column_names[0]
                    break
                    
            if len(current_df.columns) == 1:
                break
                
            if corr_value <= self.max_acceptable_correlation:
                break
                
            current_df.drop(most_correlated_column, axis = 1, inplace = True)
            
        self.desired_names = current_df.columns.tolist()
        
        return self
    
    def transform(self, X, y= None):
        
        return X[self.desired_names]
        
class VIFFilter():
    
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
    
class CliquesFilter():
    
    def __init__(self, trsh = 0.65):
        self.trsh = 0.65
        
    def fit(self, X, y):
        qlq_list, G = cliques.get_noncollinear_fts(
                                                    X, y, trsh=0.65, mode="all", verbose=False
                                                    )
        self.desired_names = qlq_list[list(qlq_list)[0]][0]
        
    def transform(self, X, y = None):
        
        return X[self.desired_names]
