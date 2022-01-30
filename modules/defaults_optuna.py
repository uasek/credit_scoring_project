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
from modules.feature_selection import CombineWithReferenceFeature_adj
from mlxtend.feature_selection import SequentialFeatureSelector
from feature_engine.selection  import SelectByShuffling
from feature_engine.selection  import RecursiveFeatureAddition
from feature_engine.selection  import SmartCorrelatedSelection

# clustering as feature engineering method
from modules.clusters import ClusterConstr

# classifiers
# from catboost import CatBoostClassifier
# from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

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
SeqFearSel_module = SequentialFeatureSelector(
    estimator  = xgb_mdl,  
    # k_features = 5,                                                  
    forward    = True,                                                  
    floating   = True,                                                
    verbose    = 0,
    cv         = 5
)

RecFeatAdd_module = RecursiveFeatureAddition(
    xgb_mdl,
    threshold = 0.005
)

SmartSel_module = SmartCorrelatedSelection(
    # variables=X.columns.to_list(),
    method="pearson",                # можно взять свою функцию
    threshold=0.3,                   # порог корреляции
    selection_method="variance",     # из коррелирующих групп выбираем признак с наиб дисперсией
    estimator=None,                  # понадобится для selection_method="model_performance"        
    cv=5
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
    'RecFeatAdd':  RecFeatAdd_module,
    
    # data imbalances
    'RUS':         RUS_module,      
    'ROS':         ROS_module,      
    'SMOTE':       SMOTE_module,  
    'ADASYN':      ADASYN_module,
    
    # feature selection
    'SeqFearSel':  SeqFearSel_module,
    'RecFeatAdd':  RecFeatAdd_module,
    'SmartSel':    SmartSel_module,
    # 'SelShuffl':   SelShuffl_module, 
    
    # classifiers
    'xgb':        xgb_mdl
    }