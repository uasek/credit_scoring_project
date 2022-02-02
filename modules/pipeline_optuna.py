# imports
import numpy as np
from sklearn.pipeline import Pipeline

# import optuna
from collections import OrderedDict
from modules.defaults_optuna import modules_dict, get_params


def get_fast_pipe(trial):
    """Works."""

    pipe_params = OrderedDict()
    pipe_params['cat_encoding'] = trial.suggest_categorical('cat_encoding', ['OneHot', 'WoE'])
    pipe_params['missing_vals'] = trial.suggest_categorical('missing_vals', ['passthrough', 'MeanImp', 'MedImp']) 
    # pipe_params['imbalance']    = trial.suggest_categorical('imbalance',    ['passthrough', 'RUS', 'ROS'])
    pipe_params['feat_eng']     = trial.suggest_categorical('feat_eng',     ['passthrough', 'PCA']) # , 'kPCA'
    # pipe_params['feat_sel']     = trial.suggest_categorical('feat_sel',     ['passthrough', 'SmartSel', 'SelShuffl'])
    # pipe_params['feat_sel']     = trial.suggest_categorical('feat_sel',     ['passthrough', 'SmartSel', 'SelShuffl', 'RecFeatAdd', 'SinglePerf'])  # 'SeqFeatSel'
    pipe_params['feat_sel']     = trial.suggest_categorical('feat_sel',     ['passthrough', 'SeqFeatSel', 'RecFeatAdd', 'SelShuffl', 'SmartSel'])
    pipe_params['boosting']     = 'xgb'

    return pipe_params
    
def get_standard_pipe(trial):
    """Thows error."""

    pipe_params = OrderedDict()
    pipe_params['cat_encoding'] = trial.suggest_categorical('cat_encoding', ['OneHot', 'WoE'])
    pipe_params['missing_vals'] = trial.suggest_categorical('missing_vals', ['passthrough', 'MeanImp', 'MedImp']) 
    # pipe_params['imbalance']    = trial.suggest_categorical('imbalance',    ['passthrough', 'RUS', 'ROS', 'SMOTE', 'ADASYN'])
    pipe_params['scaler']       = trial.suggest_categorical('scaler',       ['passthrough', 'StandSc', 'MinMax', 'StandSc', 'WinsTrans', 'LogTrans', 'PwrTrans',  'YeoJTrans']) # 'BxCxTrans',
    pipe_params['feat_eng']     = trial.suggest_categorical('feat_eng',     ['passthrough', 'PCA', 'Isomap',]) # , 'kPCA', 'UMAP'
    pipe_params['clusters']     = trial.suggest_categorical('clusters',     ['passthrough', 'kmeans', 'mbatch_kmeans']) 
    pipe_params['feat_sel']     = trial.suggest_categorical('feat_sel',     ['passthrough', 'SmartSel', 'SelShuffl', 'RecFeatAdd', 'SinglePerf'])  # 'SeqFeatSel'
    pipe_params['boosting']     = 'xgb'

    return pipe_params


def get_greedy_pipe(trial):
    """Not tested yet."""

    pipe_params = OrderedDict()
    pipe_params['cat_encoding'] = trial.suggest_categorical('cat_encoding', ['OneHot', 'WoE'])
    pipe_params['missing_vals'] = trial.suggest_categorical('missing_vals', ['passthrough', 'MeanImp', 'MedImp', 'ModeImp', 'RandomImp', 'KNNImp', 'IterImp']) 
    # pipe_params['imbalance']    = trial.suggest_categorical('imbalance',    ['passthrough', 'RUS', 'ROS', 'SMOTE', 'ADASYN'])
    pipe_params['scaler']       = trial.suggest_categorical('scaler',       ['passthrough', 'StandSc', 'MinMax','StandSc' ,'WinsTrans', 'LogTrans', 'PwrTrans',  'YeoJTrans']) # 'BxCxTrans'
    pipe_params['feat_eng']     = trial.suggest_categorical('feat_eng',     ['passthrough', 'PCA', 'Isomap', 'CombWRef']) # , 'kPCA', 'UMAP'
    pipe_params['clusters']     = trial.suggest_categorical('clusters',     ['passthrough', 'kmeans', 'mbatch_kmeans', 'birch']) 
    pipe_params['feat_sel']     = trial.suggest_categorical('feat_sel',     ['passthrough', 'SeqFeatSel', 'RecFeatAdd', 'SelShuffl', 'SmartSel'])
    pipe_params['boosting']     = 'xgb'

    return pipe_params


def get_model(trial, mode='fast'):
    """
    Analogue for get_space from hyperopt pipeline.

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