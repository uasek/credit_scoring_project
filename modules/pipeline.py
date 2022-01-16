# import lightgbm as lgb
# import xgboost as xgb
# import catboost as ctb

from functools import partial
import numpy as np
import pandas as pd
import warnings
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from imblearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials


class PipeHPOpt(object):

    def __init__(self,
                 modules, 
                 mode='kfold', 
                 n_folds = 5, 
                 test_size=.33, 
                 seed=42):
        
        if (mode != 'kfold') & (mode != 'valid'):
            raise ValueError("Choose mode 'kfold' or 'valid'")
        if (mode == 'valid') & (n_folds != 5):
            warnings.warn("Non-default n_folds won't be used since mode == valid!")
        if (mode == 'kfold') & (test_size != .33):
            warnings.warn("Non-default test_size won't be used since mode == kfold!")
    
        self.mode    = mode
        self.n_folds = n_folds
        self.seed    = seed
        self.modules = modules 

    def train(self, X, y, space, trials, algo, max_evals):
        self._last_space = space
        # We make all splits before optimization to make results more stable
        # and to improve overall performance
        if self.mode == 'kfold':
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
            _k_idx = kf.split(X)
            _pipe_partial = partial(self._pipe, X=X, y=y, _k_idx=_k_idx)
        elif self.mode == 'valid':
            x_train, x_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.seed
            )
            _pipe_partial = partial(self._pipe, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
        result = fmin(fn=_pipe_partial, space=space, algo=algo, max_evals=max_evals, trials=trials)
        self.result = result
        self.trials = trials
        self.best_params = self._get_best_params()
        self.best_model = self._get_best_model(X, y)
        return self.best_model, self.best_params, trials

    def _pipe(self, para, x_train=None, x_test=None, y_train=None, y_test=None, X=None, y=None, _k_idx=None):
        pipe_steps = self._get_ordered_steps(para)
        # print(pipe_steps)
        reg = Pipeline(pipe_steps)
        for p in para['set_params']:
            try:
                if para['set_params'][p] == int(para['set_params'][p]):
                    para['set_params'][p] = int(para['set_params'][p])
                reg.set_params(**{p: para['set_params'][p]})
            except:
                pass
        if self.mode == 'kfold':
            return self._train_reg_kfold(reg, para, X, y, _k_idx)
        elif self.mode == 'valid':
            return self._train_reg_valid(reg, para, x_train, x_test, y_train, y_test)

    def _train_reg_valid(self, reg, para, x_train, x_test, y_train, y_test):
        reg.fit(x_train, y_train)
        pred = reg.predict_proba(x_test)[:, 1]
        loss = para['loss_func'](y_test, pred)
        return {'loss': loss, 'model': reg, 'params': para, 'status': STATUS_OK}
    
    def _train_reg_kfold(self, reg, para, X, y, _k_idx):
        losses = []
        for train_index, test_index in _k_idx:
            X_split_train, X_split_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_split_train, y_split_test = y.iloc[train_index, ],  y.iloc[test_index, ]
            reg.fit(X_split_train, y_split_train)
            pred = reg.predict_proba(X_split_test)[:, 1]
            loss = para['loss_func'](y_split_test, pred)
            losses.append(loss)
        return {'loss': np.mean(losses), 'params': para, 'status': STATUS_OK}
    
    def _get_ordered_steps(self, para):
        # hp shuffles parameters, even OrderedDict(). To overcome this we
        # import order from the input OrderedDict()
        correct_order = list(self._last_space['pipe_params'].keys())
        hp_modules = para['pipe_params']
        return [(hp_modules[i], self.modules[hp_modules[i]]) for i in correct_order if hp_modules[i] != 'skip']
    
    def _get_best_params(self):
        best_params = self.trials.results[np.argmin([r['loss'] for r in self.trials.results])]['params']
        pipe_params_adj = OrderedDict()
        for i in list(self._last_space['pipe_params'].keys()):
            pipe_params_adj[i] = best_params['pipe_params'][i]
        best_params['pipe_params'] = pipe_params_adj
        return best_params
    
    def _get_best_model(self, X, y):
        para = self._get_best_params()
        pipe_steps = self._get_ordered_steps(para)
        reg = Pipeline(pipe_steps)
        for p in para['set_params']:
            try:
                # hyperopt cannot generate params as ints
                # quniform returns params as round(),
                # but most packages return error if, for example,
                # num_leaves = 1.0
                if para['set_params'][p] == int(para['set_params'][p]):
                    para['set_params'][p] = int(para['set_params'][p])
                reg.set_params(**{p: para['set_params'][p]})
            except:
                pass
        reg.fit(X, y)
        return reg
        
    def plot_convergence(self, path=None, lw=2):
        plt.plot(np.array([r['loss'] for r in self.trials.results]))
        plt.title('Hyperopt: loss function dynamics')
        plt.xlabel('Epoch')
        if self.mode == 'kfold':
            plt.ylabel('Average loss on k cross-validation samples')
        if self.mode == 'valid':
            plt.ylabel('Average loss on validation sample')
        plt.show()
        if path is not None:
            plt.savefig(path)
            
    def plot_roc(self, X_train, y_train, X_test, y_test, mdl=None, path=None, lw=2):
        # 2 modes are supported:
        # > mode="roc" — builds ROC curve
        # > mode="gain" — builds Gain curve
        if mdl is None:
            mdl = self.get_best_model()
        y_train_pred = mdl.predict_proba(X_train)[:,1]
        y_test_pred  = mdl.predict_proba(X_test)[:,1]
        
        # train
        fpr, tpr, _ = roc_curve(y_train, y_train_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color="navy", lw=lw, label="ROC curve (train, AUC = %0.3f)" % roc_auc)
        
        # test
        fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color="darkorange", lw=lw, label="ROC curve (test, AUC = %0.3f)" % roc_auc)
        
        plt.plot([0, 1], [0, 1], color="grey", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Receiver operating characteristic (ROC) curve')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.show()
        
        if path is not None:
            plt.savefig(path)
            
    
    def plot_gain(self, X_train, y_train, X_test, y_test, mdl=None, path=None, lw=2):
        pass
        
    def plot_lift(self, X_train, y_train, X_test, y_test, mdl=None, path=None, lw=2):
        pass
        
    def plot_precision_recall(self, X_train, y_train, X_test, y_test, mdl=None, path=None, lw=2):
        pass
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        