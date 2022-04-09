from feature_engine.creation import CombineWithReferenceFeature


class CombineWithReferenceFeature_adj():
    '''
    Wrapper for CombineWithReferenceFeature()
    Let's not to set parameters:
    + variables_to_combine
    + reference_variables
    beforehand (otherwise won't work with OneHotEncoder and other transformers.
    Parameters are set at .fit()
    '''
    def __init__(self, operations):
        self.operations = operations
        
    def fit(self, X, y):
        self.combinator = CombineWithReferenceFeature(
            variables_to_combine = list(X.columns),
            reference_variables = list(X.columns),
            operations = self.operations
        )
        self.combinator.fit(X, y)
        return(self)
    
    def transform(self, X):
        return(self.combinator.transform(X))        