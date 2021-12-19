# import lightgbm as lgb
# import xgboost as xgb
# import catboost as ctb

import numpy as np
import pandas as pd
import warnings

from imblearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials


class PipeHPOpt(object):

    def __init__(self, X, y, modules, mode='kfold', n_folds = 5, test_size=.33, seed=42):
        if (mode != 'kfold') & (mode != 'valid'):
            raise ValueError("Choose mode 'kfold' or 'valid'")
        if (mode == 'valid') & (n_folds != 5):
            warnings.warn("Non-default n_folds won't be used since mode == valid!")
        if (mode == 'kfold') & (test_size != .33):
            warnings.warn("Non-default test_size won't be used since mode == kfold!")
            
        self.X       = X
        self.y       = y
        self.mode    = mode
        self.n_folds = n_folds
        self.seed    = seed
        self.modules = modules 
        
        if mode == 'valid':
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=seed
            )

    def process(self, space, trials, algo, max_evals):
        try:
            result = fmin(fn=self._pipe, space=space, algo=algo, max_evals=max_evals, trials=trials)
        except Exception as e:
            return {'status': STATUS_FAIL,
                    'exception': str(e)}
        self.result = result
        self.trials = trials
        return result, trials

    def _pipe(self, para):
        # print(para)
        pipe_steps = [(para['pipe_params'][i], self.modules[para['pipe_params'][i]]) for i in para['pipe_params'] if para['pipe_params'][i] != 'skip']
        reg = Pipeline(pipe_steps)
        for p in para['set_params']:
            try:
                reg.set_params(**{p: para['set_params'][p]})
            except:
                pass
        if self.mode == 'kfold':
            return self._train_reg_kfold(reg, para)
        elif self.mode == 'valid':
            return self._train_reg_valid(reg, para)

    def _train_reg_valid(self, reg, para):
        reg.fit(self.x_train, self.y_train)
        pred = reg.predict_proba(self.x_test)[:, 1]
        loss = para['loss_func'](self.y_test, pred)
        return {'loss': loss, 'model': reg, 'params': para, 'status': STATUS_OK}
    
    def _train_reg_kfold(self, reg, para):
        kf = KFold(n_splits=5, shuffle=True, random_state=self.seed)
        losses = []
        for train_index, test_index in kf.split(self.X):
            X_split_train, X_split_test = self.X.iloc[train_index, :], self.X.iloc[test_index, :]
            y_split_train, y_split_test = self.y.iloc[train_index, ],  self.y.iloc[test_index, ]
            reg.fit(X_split_train, y_split_train)
            pred = reg.predict_proba(X_split_test)[:, 1]
            loss = para['loss_func'](y_split_test, pred)
            losses.append(loss)
        return {'loss': np.mean(losses), 'params': para, 'status': STATUS_OK}
    
    def get_best_params(self):
        return self.trials.results[np.argmin([r['loss'] for r in self.trials.results])]['params']
    
    def get_best_model(self):
        para = self.get_best_params()
        pipe_steps = [(para['pipe_params'][i], self.modules[para['pipe_params'][i]]) for i in para['pipe_params'] if para['pipe_params'][i] != 'skip']
        reg = Pipeline(pipe_steps)
        for p in para['set_params']:
            try:
                reg.set_params(**{p: para['set_params'][p]})
            except:
                pass
            
        reg.fit(self.X, self.y)
        return reg
        
    