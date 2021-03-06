from typing import List, Union

import numpy as np
import pandas as pd

from feature_engine.encoding.base_encoder import BaseCategoricalTransformer
from feature_engine.validation import _return_tags
from feature_engine.variable_manipulation import _check_input_parameter_variables

class WoEEncoder_adj(BaseCategoricalTransformer):
    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        ignore_format: bool = False,
    ) -> None:

        if not isinstance(ignore_format, bool):
            raise ValueError("ignore_format takes only booleans True and False")

        self.variables = _check_input_parameter_variables(variables)
        self.ignore_format = ignore_format

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Learn the WoE.
        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the categorical variables.
        y: pandas series.
            Target, must be binary.
        """
        
        X = self._check_fit_input_and_variables(X)

        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        # check that y is binary
        if y.nunique() != 2:
            raise ValueError(
                "This encoder is designed for binary classification. The target "
                "used has more than 2 unique values."
            )

        temp = pd.concat([X, y], axis=1)
        temp.columns = list(X.columns) + ["target"]

        # if target does not have values 0 and 1, we need to remap, to be able to
        # compute the averages.
        if any(x for x in y.unique() if x not in [0, 1]):
            temp["target"] = np.where(temp["target"] == y.unique()[0], 0, 1)

        self.encoder_dict_ = {}

        total_pos = temp["target"].sum()
        total_neg = len(temp) - total_pos
        temp["non_target"] = np.where(temp["target"] == 1, 0, 1)

        for var in self.variables_:
            pos = (temp.groupby([var])["target"].sum() + .5) / total_pos
            neg = (temp.groupby([var])["non_target"].sum() + .5) / total_neg

            t = pd.concat([pos, neg], axis=1)
            t["woe"] = np.log(t["target"] / t["non_target"])

            # we make an adjustment to override this error
            # if (
            #     not t.loc[t["target"] == 0, :].empty
            #     or not t.loc[t["non_target"] == 0, :].empty
            # ):
            #     raise ValueError(
            #         "The proportion of one of the classes for a category in "
            #         "variable {} is zero, and log of zero is not defined".format(var)
            #     )

            self.encoder_dict_[var] = t["woe"].to_dict()

        self._check_encoding_dictionary()

        self.n_features_in_ = X.shape[1]

        return self

    # Ugly work around to import the docstring for Sphinx, otherwise not necessary
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = super().transform(X)

        return X

    transform.__doc__ = BaseCategoricalTransformer.transform.__doc__

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = super().inverse_transform(X)

        return X

    inverse_transform.__doc__ = BaseCategoricalTransformer.inverse_transform.__doc__

    def _more_tags(self):
        """
        In the current format, the tests are performed using continuous np.arrays.
        This means that when we encode some of the values, the denominator is 0
        and this the transformer raises an error, and the test fails.
        For this reason, most sklearn transformers will fail. And it has nothing to
        do with the class not being compatible, it is just that the inputs passed
        are not suitable
        """
        tags_dict = _return_tags()
        tags_dict["_skip_test"] = True
        return tags_dict