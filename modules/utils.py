import sklearn
import imblearn


def filter_params(params: dict, pipe: imbearn.pipeline.Pipeline):
    '''
    From all input parameters filter only
    those that are relevant for the current
    pipeline
    
    Parameters:
    ----------
    params: dict of parameters with values obtained from 
        hyperparameter optimization (hyperopt or optuna)
    pipe: sklearn.pipeline.Pipeline or imblearn.pipeline.Pipeline
        current Pipeline model based on sklearn implementation
    '''
    pipe_steps = list(pipe.named_steps.keys())
    params_keys = list(params.keys())
    
    return {
        key: params[key]
        for key in params_keys
        if key.split('__')[0] in pipe_steps
    }

## test
# pipe = Pipeline([
#     # ('woe',       self.mudules['woe']),
#     ('onehot',      modules['onehot']), # must-have
#     ('kPCA',        modules['kPCA']),
#     # ('UMAPer',    self.mudules['UMAPer']),
#     ('feat_eng',    modules['feat_eng']),
#     # ('feat_select', self.mudules['feat_sel']),
#     # ('lgbm',        modules['lgbm']) # must-have
# ])

# params = {
#     'kPCA__n_components':     8,
#     'lgbm__learning_rate':    .1,
#     'lgbm__num_leaves':       10,
#     'lgbm__reg_alpha':        8,
#     'lgbm__reg_lambda':       8,
#     'lgbm__n_estimators':     100
# }

# filter_params(params, pipe)


def construct_pipe(steps_dict: dict, 
                   modules: dict) -> list:
    '''
    Construct a pipeline given structure
    
    Parameters:
    ----------
    steps_dict: dictionary  like {'feature_transform_stage': 'WoE'}
    modules: dictionary like {'module_name': class}
    '''
    return [(steps_dict[s], modules[steps_dict[s]]) for s in steps_dict if steps_dict[s] != 'skip']

## test
# construct_pipe({'cat_encoding': 'onehot', 'feat_eng': 'kPCA'}, modules)


def gini_score(y: pd.Series, y_pred: pd.Series) -> float:
    """
    Calculates Gini index based on ROC AUC:
    Gini = AUC * 2 - 1
    
    Parameters:
    ----------
    y: true values of target (0 or 1)
    y: predicted vales of target (from .predict_proba)
    """
    res = sklearn.metrics.roc_auc_score(y, y_pred) * 2 - 1
    print(f"Gini: {res}")
    return(res)