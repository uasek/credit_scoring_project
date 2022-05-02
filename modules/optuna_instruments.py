# imports
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from tqdm.notebook import tqdm
from sklearn.base import clone

def optimized_function(
    trial, *,
    stages_options: dict,
    hparams_options: dict,
    loss: callable,
    X: pd.DataFrame,
    y: pd.Series,
    kf: "kfold instance")->float:
    """
    Base function for optuna optimization. Takes options for pipeline stages (stages_options)
    and options for hyperparameters (hparams_options), loss function, and dataset (X, y).

    With the params given in trial (argument from optuna) constructs a model, performs
    cross validation (defined in kf argument), and returns cross val score.
    """

    # construct_model
    model = []
    hparams = {}

    ## choose stages

    # example items: ("cat_feat", [woe_encoder]), ("feat_sel", [sel_shuffl]), ...
    for stage, options in stages_options.items():
        if len(options) == 0: continue

        options_idx = list(range(len(options)))  # choose from 0, 1, ... len(options) - 1
        choice_idx = trial.suggest_categorical(stage, options_idx)

        choice = options[choice_idx]
        if choice == "skip": continue

        pipeline_nm = choice.pipeline_nm
        model.append((pipeline_nm, choice))

    model = Pipeline(model)

    ## add hyperparameters (see hparams_options_* dictionaries in notes)
    for pipeline_nm in model.named_steps.keys():
        if pipeline_nm in hparams_options:
            give_hparams_func = hparams_options[pipeline_nm]
            hparams_choice = give_hparams_func(trial)
            hparams.update(hparams_choice)

    model.set_params(**hparams)

    # assess model quality
    losses = []

    for train_index, test_index in kf.split(X, y):
        X_split_train, X_split_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_split_train, y_split_test = y.iloc[train_index],  y.iloc[test_index]

        model = clone(model)  # create a new unfitted instance of the model
        model.fit(X_split_train, y_split_train)
        pred = model.predict_proba(X_split_test)[:, 1]
        losses.append(loss(y_split_test, pred))
    
    loss_val = np.mean(losses)

    return loss_val

def get_top_k_models(study_results, top_k, stages_options):
    """
    Technical function for quality assessment in my diploma.
    """

    list_of_trials = study_results.copy()
    # get only unique parameter sets
    list_of_trials.sort(key=lambda Trial: Trial["value"], reverse=True)  # inplace sort by key
    list_of_trials = list_of_trials[:top_k]                           # get the best ones

    list_of_models = []

    for Trial in list_of_trials:
        model = []

        ## choose stages
        for stage, options in stages_options.items():
            if len(options) == 0: continue

            choice_idx = Trial["params"][stage]
            choice = options[choice_idx]
            if choice == "skip": continue
            pipeline_nm = choice.pipeline_nm
            model.append((pipeline_nm, choice))

        ## choose hparams
        model = Pipeline(model)
        hparams = {key : val for (key, val) in Trial["params"].items() if key not in stages_options.keys()}  # hparams are in trial params
        model.set_params(**hparams)
        model = clone(model)  # create a new unfitted instance of the model

        list_of_models.append(model)

    return list_of_models

def get_top_test_scores(study_results, top_k, stages_options, *, X_train, y_train, X_test, y_test, loss):
    """
    Technical function for quality assessment in my diploma.
    """

    list_of_models = get_top_k_models(study_results=study_results, top_k=top_k, stages_options=stages_options)

    test_scores = []
    for model in tqdm(list_of_models):
        model.fit(X_train, y_train)
        pred = model.predict_proba(X_test)[:, 1]
        test_score = loss(y_test, pred)
        test_scores.append(test_score)

    return np.array(test_scores)