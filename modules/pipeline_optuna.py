# imports
import numpy as np
from sklearn.pipeline import Pipeline
# import optuna
from collections import OrderedDict

from modules.defaults_optuna import modules_dict


def get_params(trial, modules):
    """
    Analogue for get_set_params from hyperopt pipeline

    Parameters
    ----------
    trial : ...
        Technical argument used by optuna
    modules : array-like
        Array with modules names used in pipeline.

    Returns
    -------
    params : dict
        Dict with params of the pipeline.
    """

    params = {}

    if "PCA" in modules:
        params.update({
            'feat_eng_PCA__n_components'    : trial.suggest_int('feat_eng_PCA__n_components', 2, 11),
            'feat_eng_PCA__whiten'          : trial.suggest_categorical('feat_eng_PCA__whiten', [True, False]),
            'feat_eng_PCA__svd_solver'      : trial.suggest_categorical('feat_eng_PCA__svd_solver', ['full', 'arpack', 'auto', 'randomized'])
        })

    if "kPCA" in modules:
        params.update({
            'feat_eng_kPCA__n_components'   :  trial.suggest_int('feat_eng_kPCA__n_components', 5, 11),
            'feat_eng_kPCA__kernel'         :  trial.suggest_categorical('feat_eng_kPCA__kernel', ['linear', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed'])
        })

    if "Isomap" in modules:
        params.update({
            'feat_eng_Isomap__n_neighbors'  :   trial.suggest_int('feat_eng_Isomap__n_neighbors', 2, 5),
            'feat_eng_Isomap__n_components' :   trial.suggest_int('feat_eng_Isomap__n_components', 2, 11),
            'feat_eng_Isomap__path_method'  :   trial.choice('feat_eng_Isomap__path_method',    ['auto', 'FW', 'D']),
        })

    if "UMAP" in modules:
        params.update({
        'feat_eng_UMAP__n_neighbors'        :   trial.suggest_int('feat_eng_UMAP__n_neighbors', 2, 11),
        'feat_eng_UMAP__n_components'       :   trial.suggest_int('feat_eng_UMAP__n_components', 2, 11),
        'feat_eng_UMAP__min_dist'           :   trial.suggest_uniform('feat_eng_UMAP__min_dist', .05, 1),
        })

    # if "lgbm" in modules:
    #     params.update({
    #     'boosting_learning_rate':            trial.suggest_uniform('boosting_learning_rate', .05, .31),
    #     'boosting_num_leaves':               trial.suggest_int('boosting_num_leaves', 5, 32),
    #     'boosting_reg_alpha':                trial.suggest_uniform('boosting_reg_alpha', 0, 16),
    #     'boosting_reg_lambda':               trial.suggest_uniform('boosting_reg_lambda', 0, 16),
    #     'boosting_n_estimators':             100
    #     })

    if "xgb" in modules:
        params.update({
            "boosting_xgb__n_estimators"     : trial.suggest_int("boosting_xgb__n_estimators", 100, 1000),
            "boosting_xgb__max_depth"        : 2 ** trial.suggest_int("boosting_xgb__max_depth", 1, 4),
            "boosting_xgb__learning_rate"    : trial.suggest_uniform("boosting_xgb__learning_rate", .05, .31),
            "boosting_xgb__reg_alpha"        : trial.suggest_uniform("boosting_xgb__reg_alpha", 0, 16),
            "boosting_xgb__reg_lambda"       : trial.suggest_uniform("boosting_xgb__reg_lambda", 0, 16)
        })

    return params

    # if "DimRed__PCA" in modules:
    #     params.update({
            
    #     })


def get_fast_pipe(trial):

    pipe_params = OrderedDict()
    pipe_params['cat_encoding'] = trial.suggest_categorical('cat_encoding', ['OneHot', 'WoE'])
    pipe_params['missing_vals'] = trial.suggest_categorical('missing_vals', ['passthrough', 'MeanImp', 'MedImp']) 
    # pipe_params['imbalance']    = trial.suggest_categorical('imbalance',    ['passthrough', 'RUS', 'ROS'])
    pipe_params['feat_eng']     = trial.suggest_categorical('feat_eng',     ['passthrough', 'PCA']) # , 'kPCA'
    pipe_params['feat_sel']     = trial.suggest_categorical('feat_sel',     ['passthrough', 'SmartSel']) 
    pipe_params['boosting']     = 'xgb'

    return pipe_params
    
def get_standard_pipe(trial):
    """
    Аналог get_standard_pipe из пайплайна для hyperopt.

    Parameters
    ----------
    trial : ...
        Технический аргумент, используемый оптуной.

    Returns
    -------
    pipe_params : OrderedDict
        Словарь с элементами пайплайна
    """

    pipe_params = OrderedDict()
    pipe_params['cat_encoding'] = trial.suggest_categorical('cat_encoding', ['OneHot', 'WoE'])
    pipe_params['missing_vals'] = trial.suggest_categorical('missing_vals', ['passthrough', 'MeanImp', 'MedImp']) 
    # pipe_params['imbalance']    = trial.suggest_categorical('imbalance',    ['passthrough', 'RUS', 'ROS', 'SMOTE', 'ADASYN'])
    pipe_params['scaler']       = trial.suggest_categorical('scaler',       ['passthrough', 'StandSc', 'MinMax', 'StandSc', 'WinsTrans', 'LogTrans', 'PwrTrans',  'YeoJTrans']) # 'BxCxTrans',
    pipe_params['feat_eng']     = trial.suggest_categorical('feat_eng',     ['passthrough', 'PCA', 'Isomap', 'UMAP']) # , 'kPCA' 
    pipe_params['clusters']     = trial.suggest_categorical('clusters',     ['passthrough', 'kmeans', 'mbatch_kmeans']) 
    pipe_params['feat_sel']     = trial.suggest_categorical('feat_sel',     ['passthrough', 'SeqFearSel', 'RecFeatAdd'])  # , 'SmartSel'
    pipe_params['boosting']     = 'xgb'

    return pipe_params


def get_greedy_pipe(trial):

    pipe_params = OrderedDict()
    pipe_params['cat_encoding'] = trial.suggest_categorical('cat_encoding', ['OneHot', 'WoE'])
    pipe_params['missing_vals'] = trial.suggest_categorical('missing_vals', ['passthrough', 'MeanImp', 'MedImp', 'ModeImp', 'RandomImp', 'KNNImp', 'IterImp']) 
    # pipe_params['imbalance']    = trial.suggest_categorical('imbalance',    ['passthrough', 'RUS', 'ROS', 'SMOTE', 'ADASYN'])
    pipe_params['scaler']       = trial.suggest_categorical('scaler',       ['passthrough', 'StandSc', 'MinMax','StandSc' ,'WinsTrans', 'LogTrans', 'PwrTrans',  'YeoJTrans']) # 'BxCxTrans'
    pipe_params['feat_eng']     = trial.suggest_categorical('feat_eng',     ['passthrough', 'PCA', 'Isomap', 'UMAP', 'CombWRef']) # , 'kPCA' 
    pipe_params['clusters']     = trial.suggest_categorical('clusters',     ['passthrough', 'kmeans', 'mbatch_kmeans', 'birch']) 
    pipe_params['feat_sel']     = trial.suggest_categorical('feat_sel',     ['passthrough', 'SeqFearSel', 'RecFeatAdd']) # 'SelShuffl', 'SmartSel'
    pipe_params['boosting']     = 'xgb'

    return pipe_params


def get_model(trial, mode='fast'):
    """
    Аналог get_space из пайплайна для hyperopt.

    Parameters
    ----------
    trial : ...
        Technical argument used by optuna.
    mode : {'fast', 'standard', 'greedy'}
        Type of pipeline. Affects the modules available for choice.

    Returns
    -------
    model : sklearn.pipeline.Pipeline
        Model to be fitted.
    """
    
    if mode == 'fast':
        pipe_params = get_fast_pipe(trial)
    elif mode == 'standard':
        pipe_params = get_standard_pipe(trial)
    elif mode == 'greedy':
        pipe_params = get_greedy_pipe(trial)

    set_params = get_params(trial, pipe_params.values())

    # "feat_eng" : "PCA" -> "feat_eng_PCA", PCA-object
    model = Pipeline(
        [(f"{key}_{val}", modules_dict[val]) for key, val in pipe_params.items()]
    )

    # print(pipe_params)
    # print(model)
    # print(f"{model.get_params()=}")
    # print(set_params) 
    model.set_params(**set_params)

    return model


def optimized_function(trial, mode, loss, strategy, *, X_train=None, X_val=None, y_train=None, y_val=None, X=None, y=None, kf=None):
    """
    Base function for optuna optimization.

    Notes
    -----
    Add partial?
    """

    model = get_model(trial, mode=mode)

    if strategy == "valid":
        assert X_train is not None and y_train is not None\
            and X_val is not None and y_val is not None\
                , "Not enough arguments for strategy='valid'"

        model.fit(X_train, y_train)
        pred = model.predict_proba(X_val)[:, 1]
        loss_val = loss(y_val, pred)

    elif strategy == "kfold":
        assert X is not None and y is not None\
             and kf is not None, "Not enough arguments for strategy='kfold'"

        losses = []
        for train_index, test_index in kf:
            X_split_train, X_split_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_split_train, y_split_test = y.iloc[train_index],  y.iloc[test_index]

            model.fit(X_split_train, y_split_train)
            pred = model.predict_proba(X_split_test)[:, 1]
            losses.append(loss(y_split_test, pred))
            loss_val = np.mean(losses)

    else:
        raise ValueError(f"Invalid argument {strategy=}. It should in {'valid', 'kfold'} ")

    return loss_val