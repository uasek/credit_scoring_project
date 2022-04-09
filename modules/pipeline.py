# import lightgbm as lgb
# import xgboost as xgb
# import catboost as ctb

from functools import partial
import numpy as np
import pandas as pd
import warnings
import collections
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from imblearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials


class PipeHPOpt(object):
    """
    Pipeline with hyperparameter optimization with reliance
    on hyperopt. Provides flexibility to add custom modules
    and change pipeline structure.
    
    Attributes:
    ----------
    modules: Python dictionary of modules for the pipeline
        All modules that will be used for training in .train(space).
        Keys should be consistent with what .train(space).
        Since Pipeline() relies on sklearn.pipeline.Pipeline and
        imblearn.pipeline.Pipeline, it is required that all modules
        except the last one have .fit() and .transform() methods.
        They can also have .fit_resample() needed for imblearn.
        The last module (classifier ir regressor) should have
        .fit() and .predict() / .predict_proba() methods.
    mode: validation mode: 'kfold' (k-fold cross-validation) 
        or 'valid' (train/test division based on sklearn). 
    n_folds: number of folds in cross-validation (default: 5)
        Applies only if mode = 'kfold'
    test_size: percentage of observations in test sample 
        (default: .33). Applies only if mode = 'valid'
    seed: random seed (default: 42)
    result: result obtained from hyperopt optimization (.train())
    trials: logs obtained from hyperopt optimization (.train())
    best_params: best pipeline parameters obtained from hyperopt optimization
    best_model: best pipeline instance obtained from hyperopt optimization
    _last_space: space that was trained for the last time.
        Stored to retain pipeline order
    
    Methods:
    ----------
    train: finds optimal pipeline structure and hyperparameters,
        returns optimal trained model, its parameters + hyperopt logs
    plot_convergence: plots loss values over optimization epochs
    plot_roc: plots ROC/Gain curve for classifier
    
    (Private) methods:
    ----------
    _get_minibatch_size: estimates n of obs. is train/test minibatches
    _pipe: creates pipeline and estimates its loss on test / kfold cross-val
    _train_reg_valid: estimates model loss on test samples
    _train_reg_kfold: estimates model loss on cross-val samples
    _get_ordered_steps: since hyperopt shuffles elements of FrozenDicts,
        this method reorders them
    _get_best_params: obtains best model parameters based on hyperopt logs
    _get_best_model: constructs pipeline with best params, trains and returns it
    """
    
    def __init__(self,
                 modules: dict, 
                 mode: str = 'kfold', 
                 n_folds: int = 5, 
                 test_size: float = .33, 
                 seed: int = 42) -> None:
        """
        Pipeline parameters stored in self
        
        Parameters
        ----------
        modules: Python dictionary of modules for the pipeline
            All modules that will be used for training in .train(space).
            Keys should be consistent with what .train(space).
            Since Pipeline() relies on sklearn.pipeline.Pipeline and
            imblearn.pipeline.Pipeline, it is required that all modules
            except the last one have .fit() and .transform() methods.
            They can also have .fit_resample() needed for imblearn.
            The last module (classifier ir regressor) should have
            .fit() and .predict() / .predict_proba() methods.
        mode: validation mode: 'kfold' (k-fold cross-validation) 
            or 'valid' (train/test division based on sklearn). 
        n_folds: number of folds in cross-validation (default: 5)
            Applies only if mode = 'kfold'
        test_size: percentage of observations in test sample 
            (default: .33). Applies only if mode = 'valid'
        seed: random seed (default: 42)
        """
        
        if (mode != 'kfold') & (mode != 'valid'):
            raise ValueError("Choose mode 'kfold' or 'valid'")
        if (mode == 'valid') & (n_folds != 5):
            warnings.warn("Non-default n_folds won't be used since mode == valid!")
        if (mode == 'kfold') & (test_size != .33):
            warnings.warn("Non-default test_size won't be used since mode == kfold!")
    
        self.mode = mode
        self.n_folds = n_folds
        self.test_size = test_size
        self.seed = seed
        self.modules = modules 

        
    def _get_minibatch_size(self, 
                            total_obs: int, 
                            n: int, 
                            frac: float, 
                            tp: str, 
                            mode: str, 
                            n_folds: int, 
                            test_size: float, 
                            verbose: bool = True) -> int:
        """
        Calculates minibatch size to boost training speed
        of pipelines
        
        Parameters
        ----------
        total_obs: total number of observations in df
        n: number of observations to add to each batch
        frac: is applied if n is not None, proportion of the initial 
            dataset to add to each batch
        tp: sample type ('train' or 'test')
        mode: type of validation - k-Fold cross-validation ('kfold')
            or standard train/test validation ('valid')
        n_folds: is applied if mode = 'kfold'. Number of folds
            in cross-validation
        test_size: is applied if mode = 'valid'. Size of the test
            / validation sample
        verbose: boolean for minibatch to be printed
        """
        
        if frac is None:
            n = int(np.ceil(total_obs * frac))
        if n > total_obs:
            n = total_obs
        if mode == 'valid':
            if tp == 'train':
                n = int(np.ceil(n * (1 - test_size)))
            if tp == 'test':
                n = int(np.ceil(n * test_size))
        if mode == 'kfold':
            if tp == 'train':
                n = int(np.ceil(n / n_folds * (n_folds - 1)))
            if tp == 'test':
                n = int(np.ceil(n / n_folds))
        if verbose:
            print(f"Size of {tp} minibatch is {n} obs.")
        return n
            
        
    def train(
        self, 
        X: pd.DataFrame, 
        y: pd.DataFrame, 
        space: collections.OrderedDict, 
        trials: hyperopt.Trials, 
        algo: str, 
        max_evals: int, 
        minibatch: bool = True,
        n: int = 1000,
        frac: float = None,
        verbose: bool = False) -> (imblearn.pipeline.Pipeline, dict, hyperopt.Trials):
        """
        Runs hyperopt on the input data and returns optimal pipeline structure
        
        Parameters:
        ----------
        X: input train pandas dataframe
        y: input train pandas series with target
        space: ordered dictionary of pipeline structure, each element
            is an array of modules passed to hyperopt choice
        trials: hyperopt structure to store logs
        algo: hyperparameter optimization algorithm passed to hyperopt
        max_evals: maximum number of iterations passed to hyperopt 
        minibatch: bool if minibatch calculation to be used
        n: is applied if minibatch = True. Number of obs. in minibatch 
        frac: is applied if minibatch = True and n is None. 
            % of observation to be passed to minibatch
        verbose: bool if logs to be printed
        """
        
        self._last_space = space
        
        # minibatch size for train & test
        total_obs = X.shape[0]
        if minibatch is False:
            n = total_obs
        n_train = self._get_minibatch_size(total_obs, n, frac, 'train', self.mode, self.n_folds, self.test_size, verbose)
        n_test  = self._get_minibatch_size(total_obs, n, frac, 'test',  self.mode, self.n_folds, self.test_size, verbose)
        
        # We make all splits before optimization to make results more stable
        # and to improve overall performance
        if self.mode == 'kfold':
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
            _k_idx = kf.split(X)
            _k_idx = [[i, j] for [i, j] in _k_idx]
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
        result = fmin(fn=_pipe_partial, space=space, algo=algo, max_evals=max_evals, trials=trials)
        self.result = result
        self.trials = trials
        self.best_params = self._get_best_params()
        self.best_model = self._get_best_model(X, y)
        return self.best_model, self.best_params, trials

    
    def _pipe(self, 
              para, 
              x_train: pd.DataFrame = None, 
              x_test: pd.Series = None, 
              y_train: pd.DataFrame =None, 
              y_test: pd.Series = None, 
              X: pd.DataFrame = None, 
              y: pd.Series = None, 
              _k_idx = list, 
              n_train: int = None, 
              n_test: int = None) -> dict:
        """
        Creates and trains pipeline, returns loss + logs
        
        Parameters:
        ----------
        para: input parameters from hyperopt to construct pipeline 
        x_train: train sample, applied if self.mode = 'valid'
        y_train: test sample, applied if self.mode = 'valid'
        y_train: train series with targets, applied if self.mode = 'valid'
        y_test: test series with targets, applied if self.mode = 'valid'
        X: train + valid sample, applied if self.mode = 'kfold'
        y: train + valid targets, applied if self.mode = 'kfold'
        _k_idx: indices for cross-validation
        n_train: no. of obervations in train to sample for minibatch training
        n_test: no. of obervations in test to sample for minibatch training
        """
        
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
            return self._train_reg_kfold(reg, para, X, y, _k_idx, n_train, n_test)
        elif self.mode == 'valid':
            return self._train_reg_valid(reg, para, x_train, x_test, y_train, y_test, n_train, n_test)
        
        
    def _train_reg_valid(self, 
                         reg, 
                         para: dict, 
                         x_train: pd.DataFrame, 
                         x_test: pd.Series, 
                         y_train: pd.DataFrame, 
                         y_test: pd.Series, 
                         n_train: int = None, 
                         n_test: int = None) -> dict:
        """
        Trains model on train/valid samples
        
        Parameters:
        -----------
        reg: constructed pipeline
        para: parameters of the pipeline to log
        x_train: train sample, applied if self.mode = 'valid'
        y_train: test sample, applied if self.mode = 'valid'
        y_train: train series with targets, applied if self.mode = 'valid'
        y_test: test series with targets, applied if self.mode = 'valid'
        n_train: no. of obervations in train to sample for minibatch training
        n_test: no. of obervations in test to sample for minibatch training
        """
        
        x_train = x_train.sample(n_train)
        x_test  = x_test.sample(n_test)
        y_train = y_train[x_train.index]
        y_test  = y_train[x_test.index]
        
        reg.fit(x_train, y_train)
        pred = reg.predict_proba(x_test)[:, 1]
        loss = para['loss_func'](y_test, pred)
        return {'loss': loss, 'model': reg, 'params': para, 'status': STATUS_OK}
    
    
    def _train_reg_kfold(self, 
                         reg, 
                         para: dict, 
                         X: pd.DataFrame, 
                         y: pd.Series, 
                         _k_idx: list, 
                         n_train: int = None, 
                         n_test: int = None) -> dict:
        """
        Trains model on k-fold cross-validation samples
        
        Parameters:
        -----------
        reg: constructed pipeline
        para: parameters of the pipeline to log
        X: train + valid sample, applied if self.mode = 'kfold'
        y: train + valid targets, applied if self.mode = 'kfold'
        _k_idx: indices for cross-validation
        n_train: no. of obervations in train to sample for minibatch training
        n_test: no. of obervations in test to sample for minibatch training
        """
        
        losses = []
        for train_index, test_index in _k_idx:
            X_split_train, X_split_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_split_train, y_split_test = y.iloc[train_index, ],  y.iloc[test_index, ]            
            
            X_split_train = X_split_train.sample(n_train)
            X_split_test  = X_split_test.sample(n_test)
            y_split_train = y_split_train[X_split_train.index]
            y_split_test  = y_split_test[X_split_test.index]            
            
            reg.fit(X_split_train, y_split_train)
            pred = reg.predict_proba(X_split_test)[:, 1]
            loss = para['loss_func'](y_split_test, pred)
            losses.append(loss)
        return {'loss': np.mean(losses), 'params': para, 'status': STATUS_OK}
    
    
    def _get_ordered_steps(self, para: dict) -> list:
        """
        HyperOpt shuffles parameters, even OrderedDict(). To overcome this,
        we import order from the input OrderedDict()
        
        Parameters:
        ----------
        para: unordered list of parameters from hyperopt to order
        
        (implied):
        self._last_space: OrderedDict of parameters to draw correct order from
        """
        correct_order = list(self._last_space['pipe_params'].keys())
        hp_modules = para['pipe_params']
        return [(hp_modules[i], self.modules[hp_modules[i]]) for i in correct_order if hp_modules[i] != 'skip']
    
    
    def _get_best_params(self) -> dict:
        """
        hyperopt returns trials object of optimization. Hereby
        we find the optimal parameters
        
        Parameters:
        ----------
        
        (implied):
        self.trials: logged results from hyperopt fmin optimization
        self._last_space: OrderedDict of parameters to draw correct order from
        """
        best_params = self.trials.results[np.argmin([r['loss'] for r in self.trials.results])]['params']
        pipe_params_adj = OrderedDict()
        for i in list(self._last_space['pipe_params'].keys()):
            pipe_params_adj[i] = best_params['pipe_params'][i]
        best_params['pipe_params'] = pipe_params_adj
        return best_params
    
    
    def _get_best_model(self, X: pd.DataFrame, y: pd.Series) -> sklearn.pipeline.Pipeline:
        """
        hyperopt returns trials object of optimization. Hereby
        we train and return the model with the optimal parameters
        
        Parameters:
        -----------
        X: training dataset
        y: training targets
        """
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
        
        
    def plot_convergence(self, path: str = None, lw: float = 2) -> None:
        """
        Plots loss values dynamics by epoch from hyperopt.trials
        
        Parameters:
        ----------
        path: path to save the figure
        lw: linewidth passed to matplotlib
        """
        plt.plot(np.array([r['loss'] for r in self.trials.results]), lw=lw)
        plt.title('Hyperopt: loss function dynamics')
        plt.xlabel('Epoch')
        if self.mode == 'kfold':
            plt.ylabel('Average loss on k cross-validation samples')
        if self.mode == 'valid':
            plt.ylabel('Average loss on validation sample')
        plt.show()
        if path is not None:
            plt.savefig(path)
            
            
    def plot_roc(self, 
                 X_train: pd.DataFrame,
                 y_train: pd.Series,
                 X_test: pd.DataFrame,
                 y_test: pd.Series,
                 mdl = None, 
                 path: str = None, 
                 lw: float = 2) -> None:
        """
        Plots ROC/Gain curves for ranking quality assessment
        
        2 modes are supported:
        * mode="roc" — builds ROC curve
        * mode="gain" — builds Gain curve
        
        Parameters:
        ----------
        X_train: features for train
        y_train: target for train
        X_test: features for test
        y_test: target for test
        mdl: classifier class instance 
        path: path to save the figure
        lw: linewidth passed to matplotlib
        """
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        