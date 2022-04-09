import numpy as np
import pandas as pd


class ExampleModule:
    """
    Setting limits by quantiles + log + power

    Example of usage:
    -----------
    from sklearn.pipeline import Pipeline

    caler = ExampleModule(q_lower=0.1, q_upper=0.9)
    model = LinearRegression()

    pipeline = Pipeline(
        steps=[
            ('preprocessor', scaler),
            ('model', model)
        ]
    )
    
    pipeline.fit(x, y)
    pipeline.predict(x)
    """

    def __init__(
        self,
        q_lower: float = 0.01,
        q_upper: float = 0.99,
        log: bool = False,
        power: float = 1.0
        ):
        """
        Store params
        """
        self.q_lower = q_lower
        self.q_upper = q_upper
        self.log = log
        self.power = power

    def fit(self, x, y=None):
        """
        Calculate lower and upper limits for quantiles
        """
        lower, upper = np.quantile(x, [self.q_lower, self.q_upper])
        self.lower = lower
        self.upper = upper

        return self


    def transform(self, x, y=None):
        """
        Set limits and take logarithm and/or power
        """
        result = np.clip(a=x, a_min=self.lower, a_max=self.upper)
        result = result ** self.power

        if self.log:
            result = np.log(result)

        return result
