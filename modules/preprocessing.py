import numpy as np
import pandas as pd
import feature_engine.transformation
    
class DimensionReducer():
    """
    Ugly wrapper fir various dimension reduction classes. Needed for 2 reasons:
    1. Features are not replaced, new ones are just added to df
    2. PCA output: np.array, should be replaced with pd.DataFrame 
    
    Comment AM: 
    1. Could be united with TransformerAdj class below, 
    2. .super() could be used
    """
    def __init__(self, gen_class, affx='feat', **kwargs):
        self.reducer = gen_class(**kwargs)
        self.affx = affx
        # self.reducer.set_params()
        
    def fit(self, X, y):
        self.reducer.fit(X, y)
        return self
    
    def transform(self, X):
        # potentially 
        Z = self.reducer.transform(X)
        return pd.concat([X, pd.DataFrame(Z, 
                                          index = X.index,
                                          columns = [f'{self.affx}_{i}' for i in range(Z.shape[1])])], axis=1)
    
    def set_params(self, **kwargs):
        self.reducer.set_params(**kwargs)
        return self     
    

class TransformerAdj():
    """
    Ugly wrapper fir various feature transformation classes. Needed for 2 reasons:
    1. Features are not replaced, new ones are just added to df
    2. PCA output: np.array, should be replaced with pd.DataFrame 
    
    Comment AM: 
    1. Could be united with DimensionReducer class above, 
    2. .super() could be used
    """
    def __init__(self, gen_class, sfx: str = '_adj', **kwargs) -> None:
        self.reducer = gen_class(**kwargs)
        self.sfx = sfx
        # self.reducer.set_params()
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.reducer.fit(X, y)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # potentially 
        return pd.concat([X, pd.DataFrame(self.reducer.transform(X), 
                                          index = X.index,
                                          columns = [i + self.sfx for i in X.columns])], axis=1)
    
    def set_params(self, **kwargs) -> None:
        self.reducer.set_params(**kwargs)
        return self   