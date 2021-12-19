from feature_engine.creation import CombineWithReferenceFeature


class CombineWithReferenceFeature_adj():
    '''
    Обертка вокруг CombineWithReferenceFeature()
    Позволяет не устанавливать параметры
    + variables_to_combine
    + reference_variables
    заранее (иначе не будет работать с OneHotEncoder
    и прочими преобразователями данных, а делать это при .fit()
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