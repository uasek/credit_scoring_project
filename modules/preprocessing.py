import numpy as np
import pandas as pd
import feature_engine.transformation
    
class DimensionReducer():
    """
    Обертка нужна:
    1. Чтобы не заменять фичи, а добавлять их к исходному df
    2. Для PCA ouput = np.array, требуется заменить на pd.DataFrame 
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
    Обертка нужна:
    1. Чтобы не заменять фичи, а добавлять их к исходному df
    2. Для PCA ouput = np.array, требуется заменить на pd.DataFrame 
    """
    def __init__(self, gen_class, sfx='_adj', **kwargs):
        self.reducer = gen_class(**kwargs)
        self.sfx = sfx
        # self.reducer.set_params()
        
    def fit(self, X, y):
        self.reducer.fit(X, y)
        return self
    
    def transform(self, X):
        # potentially 
        return pd.concat([X, pd.DataFrame(self.reducer.transform(X), 
                                          index = X.index,
                                          columns = [i + self.sfx for i in X.columns])], axis=1)
    
    def set_params(self, **kwargs):
        self.reducer.set_params(**kwargs)
        return self   