import numpy as np
import pandas as pd
import feature_engine.transformation


class Winsorizer():

    def __init__(self, **kwargs):
        self.winsorizer = feature_engine.outliers.Winsorizer(**kwargs)

    def fit(self, X, y=None):
        self.winsorizer.fit(X, y=None)
        return self

    def transform(self, X):
        return self.winsorizer.transform(X)

    def get_params(self):
        return self.__dict__['winsorizer'].get_params()

    def set_params(self, **kwargs):
        self = self.winsorizer.set_params(**kwargs)

        return self

    
class LogTransformer():

    def __init__(self, base='10'):
        self.LogTransformer = feature_engine.transformation.LogTransformer()
        self.LogTransformer.base = base

    def fit(self, X, y=None):
        self.LogTransformer.fit(X, y=None)
        return self

    def transform(self, X):
        return self.LogTransformer.transform(X)

    def inverse_transform(self, X):
        return self.LogTransformer.inverse_transform(X)

    def get_params(self):
        return self.__dict__['LogTransformer'].get_params()

    def set_params(self, **kwargs):
        self.LogTransformer.set_params(**kwargs)

        return self

    
class PowerTransformer():

    def __init__(self, exp=0.5):
        self.PowerTransformer = feature_engine.transformation.PowerTransformer()
        self.PowerTransformer.exp = exp

    def fit(self, X, y=None):
        self.PowerTransformer.fit(X, y=None)
        return self

    def transform(self, X):
        return self.PowerTransformer.transform(X)

    def inverse_transform(self, X):
        return self.PowerTransformer.inverse_transform(X)

    def get_params(self):
        return self.__dict__['PowerTransformer'].get_params()

    def set_params(self, **kwargs):
        self.PowerTransformer.set_params(**kwargs)

        return self

    
class BoxCoxTransformer():

    def __init__(self, exp=0.5):
        self.BoxCoxTransformer = feature_engine.transformation.BoxCoxTransformer()
        # self.BoxCoxTransformer.exp = exp

    def fit(self, X, y=None):
        self.BoxCoxTransformer.fit(X, y=None)
        return self

    def transform(self, X):
        return self.BoxCoxTransformer.transform(X)

    def get_params(self):
        return self.__dict__['BoxCoxTransformer'].get_params()

    def set_params(self, **kwargs):
        self.BoxCoxTransformer.set_params(**kwargs)

        return self


class YeoJohnsonTransformer():

    def __init__(self, exp=0.5):
        self.YeoJohnsonTransformer = feature_engine.transformation.YeoJohnsonTransformer()
        # self.BoxCoxTransformer.exp = exp

    def fit(self, X, y=None):
        self.YeoJohnsonTransformer.fit(X, y=None)
        return self

    def transform(self, X):
        return self.YeoJohnsonTransformer.transform(X)

    def get_params(self):
        return self.__dict__['YeoJohnsonTransformer'].get_params()

    def set_params(self, **kwargs):
        self.YeoJohnsonTransformer.set_params(**kwargs)

        return self
    
    
class DimensionReducer():
    """
    Обертка нужна:
    1. Чтобы не заменять фичи, а добавлять их к исходному df
    2. Для PCA ouput = np.array, требуется заменить на pd.DataFrame 
    """
    def __init__(self, gen_class, **kwargs):
        self.reducer = gen_class(**kwargs)
        # self.reducer.set_params()
        
    def fit(self, X, y):
        self.reducer.fit(X, y)
        return self
    
    def transform(self, X):
        # potentially 
        return pd.concat([X, pd.DataFrame(self.reducer.transform(X), index = X.index)], axis=1)
    
    def set_params(self, **kwargs):
        self.reducer.set_params(**kwargs)
        return self     