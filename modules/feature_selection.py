import numpy as np

from feature_engine.creation import CombineWithReferenceFeature
from feature_engine.selection import SelectByShuffling, SelectBySingleFeaturePerformance


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


class SafeSelectByShuffling(SelectByShuffling):
    """
    A wrapper for SelectByShuffling from feature_engine.selection.
    This class safely handles cases when all features are dropped by its ancestor.

    Set min_features to determine a minimum number of features to be returned. If
    there is less than min_features left after selection by shuffling, the rest is
    chosen based on performance drifts.

    Warning! This implementation assumes that greater values of the metric, which is
    set in the ancestor's parameters, means better model performance (e.g. roc auc score).

    The signature has all the arguments from SelectByShuffling to follow scikit-learn
    estimators convention to avoid varargs. This is necessary to make the class a part
    of sklearn.pipeline.Pipeline.
    """

    def __init__(
        self,
        estimator,
        min_features=1,
        scoring = "roc_auc",
        cv=3,
        threshold = None,
        variables = None,
        random_state = None,
    ):
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            cv=cv,
            threshold=threshold,
            variables=variables,
            random_state=random_state
            )
        self.min_features = min_features

    def transform(self, X):

        n_features_left  = self.n_features_in_ - len(self.features_to_drop_)

        if n_features_left >= self.min_features:
            return super().transform(X)

        else:
            m = self.min_features
            print((
                f"Less than min_features = {m} are left, "
                f"return {m} best feature{'' if m == 1 else 's'} by performance drift."
                ))

            features, values = zip(*self.performance_drifts_.items())                     # split a dictionary into keys and values
            features = np.array(features)[np.argsort(values)[::-1]]                       # sort feature names in descending order of metric drifts
            return X[features[:self.min_features]]                                        # return self.min_features features with the best values


class SafeSelectBySingleFeaturePerformance(SelectBySingleFeaturePerformance):
    """
    A wrapper for SelectBySingleFeaturePerformance from feature_engine.selection.
    This class safely handles cases when all features are dropped by its ancestor.

    Set min_features to determine a minimum number of features to be returned. If
    there is less than min_features has a univariate metric greater than the threshold,
    the rest is chosen based on that metric.

    Warning! This implementation assumes that greater values of the metric, which is
    set in the ancestor's parameters, means better model performance (e.g. roc auc score).

    The signature has all the arguments from SelectBySingleFeaturePerformance
    to follow scikit-learn estimators convention to avoid varargs. This is necessary
    to make the class a part of sklearn.pipeline.Pipeline.
    """

    def __init__(
        self,
        estimator,
        min_features=1,
        scoring="roc_auc",
        cv=3,
        threshold=None,
        variables=None,
    ):
        super().__init__(
            estimator,
            scoring=scoring,
            cv=cv,
            threshold=threshold,
            variables=variables,
        )
        self.min_features = min_features

    def transform(self, X):

        n_features_left  = self.n_features_in_ - len(self.features_to_drop_)

        if n_features_left >= self.min_features:
            return super().transform(X)

        else:
            m = self.min_features
            print((
                f"Less than min_features = {m} are left, "
                f"return {m} best feature{'' if m == 1 else 's'} by univariate performance."
                ))

                                       # feature_performance_ is used in this class
            features, values = zip(*self.feature_performance_.items())
            features = np.array(features)[np.argsort(values)[::-1]]
            return X[features[:self.min_features]]