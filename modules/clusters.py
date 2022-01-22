import numpy as np
import pandas as pd

class ClusterConstr():
    """
    Обертка нужна:
    1. Чтобы не заменять фичи, а добавлять их к исходному df
    2. Для PCA ouput = np.array, требуется заменить на pd.DataFrame 
    """
    def __init__(self, gen_class, affx='clust', **kwargs):
        self.mdl = gen_class(**kwargs)
        self.affx = affx
        # self.mdl.set_params()
        
    def fit(self, X, y):
        self.mdl.fit(X, y)
        return self
    
    def transform(self, X):
        Z = self.mdl.transform(X)
        return pd.concat([
            X, 
            pd.DataFrame(Z, index = X.index, columns = [f'{self.affx}_{i}' for i in range(Z.shape[1])]),
            pd.DataFrame(self.mdl.predict(X), index = X.index, columns = ['pred_clust'])
        ], axis=1)
    
    def set_params(self, **kwargs):
        self.mdl.set_params(**kwargs)
        return self   
    
    
    
    