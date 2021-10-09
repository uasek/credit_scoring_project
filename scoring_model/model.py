import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

# Идея: написать что-то в духе
# pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
# pipe.fit(X_train, y_train)
# ...
# У всех промежуточных элементов пайплайна должны быть
# методы fit() и transform()
# Если в Pipeline что-то не понравится - допишем его
# Ну а так возможно обойдемся написанием только его элементов
