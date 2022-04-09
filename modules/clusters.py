import numpy as np
import pandas as pd

class ClusterConstr():
    """
    Wraps clustering techniques. Needed for 2 reasons:
    1. Output is a combination of the input df and
    output embedding (cluster probabilities, components, etc.)
    2. Output of transform is pd.DataFrame, not np.array (see PCA)
    """
    
    def __init__(self, gen_class, affx: str = 'clust', **kwargs) -> None:
        if not isinstance(affx, str):
            raise ValueError("affx takes only strings, e.g. 'clust'")
        self.mdl = gen_class(**kwargs)
        self.affx = affx
        # self.mdl.set_params()
        
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Learn the clustering / DimRed technique.
        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the categorical variables.
        y: pandas series.
            Target, not needed in most cases, but retained for consistency.
        """
        self.mdl.fit(X, y)
        return self
    
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = self.mdl.transform(X)
        return pd.concat([
            X, 
            pd.DataFrame(Z, index = X.index, columns = [f'{self.affx}_{i}' for i in range(Z.shape[1])]),
            pd.DataFrame(self.mdl.predict(X), index = X.index, columns = ['pred_clust'])
        ], axis=1)
    
    transform.__doc__ = self.mdl.transform.__doc__
    
    
    def set_params(self, **kwargs):
        self.mdl.set_params(**kwargs)
        return self   
        
    set_params.__doc__ = self.mdl.set_params.__doc__
    
    
    
    