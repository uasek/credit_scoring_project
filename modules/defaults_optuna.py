## IMPORTS
import numpy as np
import sklearn
from collections import OrderedDict
## from hyperopt import hp

# import umap
## import umap.umap_ as umap

# encoders
from modules.encoders import WoEEncoder_adj
from feature_engine.encoding import OneHotEncoder

# missings
from feature_engine.imputation import MeanMedianImputer
from feature_engine.imputation import CategoricalImputer
from feature_engine.imputation import RandomSampleImputer
from modules.missings import teach_to_separate
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from modules.missings import missing_filler_mode

# Scalers
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

# preprocessing
from modules.preprocessing import DimensionReducer
from modules.preprocessing import TransformerAdj

from feature_engine.outliers import Winsorizer
from feature_engine.transformation import LogTransformer
from feature_engine.transformation import PowerTransformer
from feature_engine.transformation import BoxCoxTransformer
from feature_engine.transformation import YeoJohnsonTransformer


# imbalanced
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler

# feature selection
from modules.feature_selection import CombineWithReferenceFeature_adj, SafeSelectByShuffling
from mlxtend.feature_selection import SequentialFeatureSelector
from feature_engine.selection  import SelectByShuffling, SelectBySingleFeaturePerformance
from feature_engine.selection  import RecursiveFeatureAddition, SmartCorrelatedSelection

# clustering as feature engineering method
from modules.clusters import ClusterConstr

# classifiers
# from catboost import CatBoostClassifier
# from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
seed = 42


## CLASS DECLARATIONS
# Classifiers

# пока не встает на мак :)))
# lgbm_mdl = LGBMClassifier(
#     num_leaves = 10,
#     learning_rate = .1,
#     reg_alpha = 8,
#     reg_lambda = 8,
#     random_state = seed
# )

xgb_mdl = XGBClassifier(
    eval_metric="logloss",  # set expicitly to avoid warnings
    learning_rate = .1,
    reg_alpha = 8,
    reg_lambda = 8,
    random_state = seed
)

logreg_mdl = LogisticRegression(penalty="none", max_iter=1000)

# Encoders
WoE_module = WoEEncoder_adj()
OneHot_module = OneHotEncoder()


# Missings
MedImp_module = MeanMedianImputer(imputation_method='median')
MeanImp_module = MeanMedianImputer(imputation_method='mean')
ModeImp_module = missing_filler_mode()
RandomImp_module = RandomSampleImputer()
KNNImp_module = teach_to_separate(KNNImputer)
IterImp_module = teach_to_separate(IterativeImputer)


# Scalers
StandSc_module  = TransformerAdj(sklearn.preprocessing.StandardScaler, '_scl')
MinMaxSc_module = TransformerAdj(sklearn.preprocessing.MinMaxScaler,   '_scl')
StandSc_module  = TransformerAdj(sklearn.preprocessing.RobustScaler,   '_scl') #!

WinsTrans_module = Winsorizer()
LogTrans_module  = LogTransformer(base = '10')
PwrTrans_module  = PowerTransformer(exp = 0.5)
BxCxTrans_module = BoxCoxTransformer()
YeoJTrans_module = YeoJohnsonTransformer()


# Clustering Models
kmeans_module = ClusterConstr(
    sklearn.cluster.KMeans, 
    affx = 'clust',
    n_clusters = 2,     # Число кластеров
    init = 'k-means++', # Метод установки первых точек
    algorithm = 'auto'  # Какой алгоритм юзать
)

mbatch_kmeans_module = ClusterConstr(
    sklearn.cluster.MiniBatchKMeans, 
    affx = 'clust',
    n_clusters = 2,     # Число кластеров
    init = 'k-means++', #Метод установки первых точек
    batch_size = 1024, #Размер Batch
    reassignment_ratio = 0.01, #Параметр регуляризации
    random_state = 42
)

birch_module = ClusterConstr(
    sklearn.cluster.Birch, 
    affx = 'clust',
    n_clusters = 2,
    branching_factor = 50,
    threshold = 0.52 #Параметр регуляризации
) 


# Dimension Reducers
PCA_module = DimensionReducer(
    gen_class = sklearn.decomposition.PCA,
    n_components = 2,    # сколько оставить компонентов; по дефолту - все
    whiten = False,      # отключаем whitening - декорреляцию фичей
    svd_solver = "full", # детали SVD преобразования, за подробностями см. доки
)

kPCA_module = DimensionReducer(
    gen_class = sklearn.decomposition.KernelPCA,
    n_components = 8,  # сколько оставить компонентов; по дефолту - все
    kernel = "linear", # ядро. По дфеолту линейное. Можно сделать своё, но тогда его нужно предварительно вычислить отдельно,
                       # поставить kernel = "precomputed" и передать уже вычисленное ядро в качестве X
    degree = 3,        # степень полинома для некоторых типов ядер. Важный параметр для тьюнинга, но сильно напрягает процессор
    n_jobs = -1        # объект умеет быть многопоточным! -1 займет все ядра
)
# вот тут проблема

Isomap_module = DimensionReducer(
    gen_class = sklearn.manifold.Isomap,
    n_neighbors = 5, #количество соседей при вычислении KNN. Основной гиперпараметр, кстати (!!!)
    n_components = 2,  #сколько оставить компонент; по дефолту - 2
    path_method = "auto", #алгоритм, который вычисляет кратчайший путь. Варианты см. на странице функции. Этот подбирает сам.
    neighbors_algorithm = "auto", #алгоритм, который ищет соседей. Инстанс класса NearestNeighbours
    n_jobs = -1 #объект умеет быть многопоточным! -1 займет все ядра
)

# UMAP_module = DimensionReducer(
#     gen_class = umap.UMAP,
#     n_neighbors = 5,  # количество соседей при вычислении KNN. Основной гиперпараметр, кстати (!!!)
#     n_components = 2, # сколько оставить компонентов; по дефолту - 2
#     min_dist = 0.1    # минимальная дистанция, которую можно сохранять между точками в получающемся пространстве. 
#     # Гиперпараметр. При увеличении начинает лучше улавливать общую структуру, но хуже - локальную
# )

CombWRef_module = CombineWithReferenceFeature_adj(
    operations = ['mul']
)


# Imbalances in target
RUS_module    = RandomUnderSampler(random_state = seed)
ROS_module    = RandomOverSampler(random_state = seed)
SMOTE_module  = SMOTE(random_state = seed)
ADASYN_module = ADASYN(random_state = seed)


# Feature Engineering
CombWRef_module = CombineWithReferenceFeature_adj(operations = ['mul'])


# Feature selection
SeqFeatSel_module = SequentialFeatureSelector(  # very slow but seems to work
    estimator  = xgb_mdl,  # base model
    k_features = 5,      # number of features to select                                            
    forward    = True,     # start from 0 features (True) or from all (False)               
    floating   = True,     # whether to perform a backward step
    verbose    = 0,
    cv         = 5
)

RecFeatAdd_module = RecursiveFeatureAddition(  # rather slow
    xgb_mdl,
    variables=None,
    threshold = 0.005,
    cv=5
)

SmartSel_module = SmartCorrelatedSelection(
    variables=None,                  # If None, the transformer will evaluate all numerical variables in the dataset.  -- нужен класс.
    method="pearson",                # can be replaced by a custom function
    threshold=0.3,                   # correlation threshold
    selection_method="variance",     # select feature with greatest variance from a correlated group
    estimator=None,                  # for selection_method="model_performance"        
    cv=5
)

# params from SelectByShuffling + min_features
SelShuffl_module = SafeSelectByShuffling(
    min_features=1,
    variables=None,                   # If None, the transformer will shuffle all numerical variables in the dataset.
    estimator=logreg_mdl,
    scoring='roc_auc',
    threshold=0.01,
    cv=5
)

SinglePerf_module = SelectBySingleFeaturePerformance(  # rather slow
    estimator=logreg_mdl,
    scoring="roc_auc",
    threshold=None,               # will be automatically set to the mean performance value of all features
)

# ---
modules_dict =  {
    "passthrough" : "passthrough",

    # encoders
    'WoE':         WoE_module,
    'OneHot':      OneHot_module,
    
    # missings
    'MedImp':      MedImp_module,
    'MeanImp':     MeanImp_module, 
    'RandomImp':   RandomImp_module,
    'KNNImp':      KNNImp_module, 
    'IterImp':     IterImp_module, 
        
    # reducers
    'PCA':         PCA_module,
    'kPCA':        kPCA_module,
    'Isomap':      Isomap_module,
    # 'UMAP':        UMAP_module,
    
    # clustering models
    'kmeans':      kmeans_module,
    'mbatch_kmeans': mbatch_kmeans_module,
    'birch':       birch_module,
    
    # scalers
    'StandSc':     StandSc_module,
    'MinMax':      MinMaxSc_module,
    'StandSc':     StandSc_module, #!
    'WinsTrans':   WinsTrans_module,
    'LogTrans':    LogTrans_module ,
    'PwrTrans':    PwrTrans_module ,
    'BxCxTrans':   BxCxTrans_module,
    'YeoJTrans':   YeoJTrans_module,
    
    # feature engineering
    'CombWRef':    CombWRef_module,
    # 'RecFeatAdd':  RecFeatAdd_module,
    
    # data imbalances
    'RUS':         RUS_module,      
    'ROS':         ROS_module,      
    'SMOTE':       SMOTE_module,  
    'ADASYN':      ADASYN_module,
    
    # feature selection
    'SeqFeatSel':  SeqFeatSel_module,
    'RecFeatAdd':  RecFeatAdd_module,
    'SmartSel':    SmartSel_module,
    'SelShuffl' :   SelShuffl_module,
    'SinglePerf' : SinglePerf_module,
    
    # classifiers
    'xgb':        xgb_mdl
}


def get_params(trial, modules):
    """
    Analogue for get_set_params from hyperopt pipeline.

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
            'feat_eng_Isomap__path_method'  :   trial.suggest_categorical('feat_eng_Isomap__path_method',    ['auto', 'FW', 'D']),
        })

    if "UMAP" in modules:
        params.update({
        'feat_eng_UMAP__n_neighbors'        :   trial.suggest_int('feat_eng_UMAP__n_neighbors', 2, 11),
        'feat_eng_UMAP__n_components'       :   trial.suggest_int('feat_eng_UMAP__n_components', 2, 11),
        'feat_eng_UMAP__min_dist'           :   trial.suggest_uniform('feat_eng_UMAP__min_dist', .05, 1),
        })

    if "SeqFeatSel" in modules:
        params.update({
          "feat_sel_SeqFeatSel__estimator"    : trial.suggest_categorical("feat_sel_SeqFeatSel__estimator", [xgb_mdl, logreg_mdl]),
          "feat_sel_SeqFeatSel__k_features"   : trial.suggest_int("feat_sel_SeqFeatSel__k_features", 3, 10),  # нужна классовая архитектура
        #   "feat_sel_SeqFeatSel__forward"      : trial.suggest_categorical("feat_sel_SeqFeatSel__forward", [True, False]),
          "feat_sel_SeqFeatSel__floating"     : trial.suggest_categorical("feat_sel_SeqFeatSel__floating",[True, False])
        }) 

    if "SmartSel" in modules:
        params.update({
          "feat_sel_SmartSel__method"           : trial.suggest_categorical("feat_sel_SmartSel__method", ["pearson", "spearman"]),
          "feat_sel_SmartSel__threshold"        : trial.suggest_float("feat_sel_SmartSel__threshold", 0, 1),
          "feat_sel_SmartSel__selection_method" : trial.suggest_categorical("feat_sel_SmartSel__selection_method", ["missing_values", "cardinality", "variance"])  # что значит cardinality
        }) 

    if "SelShuffl" in modules:
        params.update({
            "feat_sel_SelShuffl__estimator"      : trial.suggest_categorical("feat_sel_SelShuffl__estimator", [logreg_mdl, xgb_mdl]),
            "feat_sel_SelShuffl__threshold"      : trial.suggest_float("feat_sel_SelShuffl__threshold", 0, 0.1)
        }) 

    if "RecFeatAdd" in modules:
        params.update({
            "feat_sel_RecFeatAdd__estimator"      : trial.suggest_categorical("feat_sel_RecFeatAdd__estimator", [logreg_mdl, xgb_mdl]),
            "feat_sel_RecFeatAdd__threshold"      : trial.suggest_float("feat_sel_RecFeatAdd__threshold", 0, 1)
        })

    if "SinglePerf" in modules:
        params.update({
            "feat_sel_SinglePerf__estimator"      : trial.suggest_categorical("feat_sel_SinglePerf__estimator", [logreg_mdl, xgb_mdl]),
            # "feat_sel_SinglePerf__threshold"      : trial.suggest_float("feat_sel_SinglePerf__threshold", 0.5, 1)  # uncomment after writing a wrapper
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