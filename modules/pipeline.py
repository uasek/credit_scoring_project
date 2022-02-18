
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

import itertools

fallen_k = 0
fallen_data = 0
fallen_pipeline = 0
fallen_test = 0
def find_dummies(X):
    count_values = X.apply(lambda x: len(x.unique()), axis = 0)
    dummy_mask = X.apply(lambda x: len(x.unique()), axis = 0) <= 2
    dummy_columns = count_values[dummy_mask].index.tolist()
    return dummy_columns


class PipeHPOpt3(object):

    def __init__(self,
                 modules,
                 folding_objects = '',
                 mode='kfold', 
                 n_folds = 5, 
                 test_size=.33, 
                 seed=42, 
                 binary = True,
                 verbose = True, 
                 minibatch_size = True
                 ):
        if (mode != 'kfold') & (mode != 'valid') & (mode != 'objects_kfolds'):
            raise ValueError("Choose mode 'kfold' or 'valid' or 'objects_kfolds'")
        if (mode == 'valid') & (n_folds != 5):
            warnings.warn("Non-default n_folds won't be used since mode == valid!")
        if (mode == 'kfold') & (test_size != .33):
            warnings.warn("Non-default test_size won't be used since mode == kfold!")
    
        self.mode    = mode
        self.folding_objects = folding_objects
        self.n_folds = n_folds
        self.test_size = test_size
        self.seed    = seed
        self.modules = modules 
        self.binary = binary
        self.verbose = verbose
        self.minibatch_size = minibatch_size
        
    def _get_minibatch_size(self, total_obs, n, frac, tp, mode, n_folds, test_size, verbose=True):
        if n > total_obs:
            n = total_obs
        if frac is not None:
            n = int(np.ceil(total_obs * frac))
        if mode == 'valid':
            # calculate n
            if tp == 'train':
                n = int(np.ceil(n * (1 - test_size)))
            if tp == 'test':
                n = int(np.ceil(n * test_size))
        if (mode == 'kfold') | (mode == "objects_kfolds"):
            if tp == 'train':
                n = int(np.ceil(n / n_folds * (n_folds - 1)))
            if tp == 'test':
                n = int(np.ceil(n / n_folds))
            
        if verbose:
            print(f"Size of {tp} minibatch is {n} obs.")
        return n
            
        
    def train(self, X, y, space, trials, algo, max_evals, minibatch=True, n=1000, frac=None, verbose=False):
        self._last_space = space
#         breakpoint()
        # minibatch size for train & test
        if self.mode != "objects_kfolds":
            total_obs = X.shape[0]
            if minibatch is False:
                n = total_obs
            n_train = self._get_minibatch_size(total_obs, n, frac, 'train', self.mode, self.n_folds, self.test_size, verbose)
            n_test  = self._get_minibatch_size(total_obs, n, frac, 'test',  self.mode, self.n_folds, self.test_size, verbose)
        else:
            total_obs = len(X[self.folding_objects].unique())
            if minibatch is False:
                n = total_obs
            n_train = self._get_minibatch_size(total_obs, n, frac, 'train', self.mode, self.n_folds, self.test_size, verbose)
            n_test  = self._get_minibatch_size(total_obs, n, frac, 'test',  self.mode, self.n_folds, self.test_size, verbose)
            print(n_train, n_test, n, total_obs)
        # We make all splits before optimization to make results more stable
        # and to improve overall performance
        if self.mode == 'kfold':
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
            _k_idx = kf.split(X)
            _k_idx = [[i,j] for [i, j] in _k_idx]
            _pipe_partial = partial(self._pipe, X=X, y=y, _k_idx=_k_idx, n_train=n_train, n_test=n_test)
        elif self.mode == 'valid':
            x_train, x_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.seed
            )
            _pipe_partial = partial(
                self._pipe, 
                x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, 
                n_train=n_train, n_test=n_test
            )
        elif self.mode == 'objects_kfolds':
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
            folding_objects_observed = pd.Series(X[self.folding_objects].unique())
            _k_idx = kf.split(folding_objects_observed)
            _k_idx = [[i,j] for [i, j] in _k_idx]
            _k_idx_new = []
            for train, test in _k_idx:
                train = folding_objects_observed.loc[train]
                test  = folding_objects_observed.loc[test]
                train_index = np.array(X[X[self.folding_objects].isin(train)].index)
                test_index = np.array(X[X[self.folding_objects].isin(test)].index)
                _k_idx_new.append([train_index, test_index])
            global fallen_k
            fallen_k = (_k_idx_new, n_test, n_train, len(folding_objects_observed))
            _pipe_partial = partial(self._pipe, X=X, y=y, _k_idx=_k_idx_new, n_train=n_train, n_test=n_test)

        result = fmin(fn=_pipe_partial, space=space, algo=algo, max_evals=max_evals, trials=trials)
        self.result = result
        self.trials = trials
        self.best_params = self._get_best_params()
        self.best_model = self._get_best_model(X, y)
        return self.best_model, self.best_params, trials

    def _pipe(self, para, x_train=None, x_test=None, y_train=None, y_test=None, X=None, y=None, _k_idx=None, n_train=None, n_test=None):
#         breakpoint()
        pipe_steps = self._get_ordered_steps(para)
        reg = Pipeline(pipe_steps)
        for p in para['set_params']:
            try:
                if para['set_params'][p] == int(para['set_params'][p]):
                    para['set_params'][p] = int(para['set_params'][p])
                reg.set_params(**{p: para['set_params'][p]})
            except:
                pass
#         breakpoint()
        if (self.mode == 'kfold') | (self.mode == 'objects_kfolds'):
            return self._train_reg_kfold(reg, para, X, y, _k_idx, n_train, n_test)
        elif self.mode == 'valid':
            return self._train_reg_valid(reg, para, x_train, x_test, y_train, y_test, n_train, n_test)
        
    def _train_reg_valid(self, reg, para, x_train, x_test, y_train, y_test, n_train=None, n_test=None):
        x_train = x_train.sample(n_train)
        x_test  = x_test.sample(n_test)
        y_train = y_train[x_train.index]
        y_test  = y_train[x_test.index]
        
        reg.fit(x_train, y_train)
        if self.binary:
            pred = reg.predict_proba(x_test)[:, 1]
        else:
            pred = reg.predict(x_test)
        loss = para['loss_func'](y_test, pred)
        return {'loss': loss, 'model': reg, 'params': para, 'status': STATUS_OK}
    
    def _train_reg_kfold(self, reg, para, X, y, _k_idx, n_train=None, n_test=None):
#         breakpoint()
        losses = []
        if self.verbose == True:
            print("Started working with:")
            print(reg)

        for train_index, test_index in _k_idx:

            X_split_train, X_split_test = X.loc[train_index, :], X.loc[test_index, :]
            y_split_train, y_split_test = y.loc[train_index, ],  y.loc[test_index, ]            
            if (self.mode == "objects_kfolds") & (self.minibatch_size is not None):
                n_train = int(np.min([np.floor(self.minibatch_size * 0.75), X_split_train.shape[0]]))
                n_test = int(np.min([self.minibatch_size - n_train, X_split_test.shape[0]]))
            X_split_train = X_split_train.sample(n_train)
            X_split_test  = X_split_test.sample(n_test)
            y_split_train = y_split_train[X_split_train.index]
            y_split_test  = y_split_test[X_split_test.index]            
            try:
                reg.fit(X_split_train, y_split_train)
                if self.binary:
                    pred = reg.predict_proba(X_split_test)[:, 1]
                else:
                    pred = reg.predict(X_split_test)

            except Exception as e:
                global fallen_data
                global fallen_pipeline
                global fallen_test
                fallen_test = X_split_test
                fallen_pipeline = reg
                fallen_data = (X_split_train, y_split_train)
                print("Was unable to fit-predict. Exception text:")
                print(e)
                print("The pipeline which broke is:")
                print(reg)
                Xt = X_split_train.copy()
                print("The data on which it broke is:")
                print(Xt)
                print("And target is:")
                print(y_split_train)
                print("Types of columns were:")
                print(Xt.dtypes)
                print(Xt.isnull().sum())
                print("I start to fit step-by-step")
                for name, step in reg.steps:
                    print("step: " + name + ", fitting")
                    step.fit(Xt, y_split_train)
                    print("step: " + name + ", transforming")
                    Xt = step.transform(Xt)
                    print("step: " + name + ", output:")
                    print(Xt)                
            loss = para['loss_func'](y_split_test, pred)
            losses.append(loss)
        if self.verbose == True:
            print(np.mean(losses))
            print("ended")
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
        
        
        
        
        
        
        
        
        