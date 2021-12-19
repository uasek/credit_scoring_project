from sklearn.metrics import roc_auc_score


def filter_params(params, pipe):
    '''
    From all input parameters filter only
    those that are relevant for the current
    pipeline
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


def construct_pipe(steps_dict, modules):
    '''
    Construct a pipeline given structure
    '''
    return [(steps_dict[s], modules[steps_dict[s]]) for s in steps_dict if steps_dict[s] != 'skip']

## test
# construct_pipe({'cat_encoding': 'onehot', 'feat_eng': 'kPCA'}, modules)


def gini_score(y, y_pred):
    res = roc_auc_score(y, y_pred) * 2 - 1
    print(f"Gini: {res}")
    return(res)