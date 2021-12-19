## IMPORTS
import sklearn
# import umap
import umap.umap_ as umap

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

# preprocessing
from modules.preprocessing import DimensionReducer

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

# classifiers
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

seed = 42


## CLASS DECLARATIONS
# Classifiers
lgbm_mdl = LGBMClassifier(
    num_leaves = 10,
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

Isomap_module = DimensionReducer(
    gen_class = sklearn.manifold.Isomap,
    n_neighbors = 5, #количество соседей при вычислении KNN. Основной гиперпараметр, кстати (!!!)
    n_components = 2,  #сколько оставить компонент; по дефолту - 2
    path_method = "auto", #алгоритм, который вычисляет кратчайший путь. Варианты см. на странице функции. Этот подбирает сам.
    neighbors_algorithm = "auto", #алгоритм, который ищет соседей. Инстанс класса NearestNeighbours
    n_jobs = -1 #объект умеет быть многопоточным! -1 займет все ядра
)

UMAP_module = DimensionReducer(
    gen_class = umap.UMAP,
    n_neighbors = 5,  # количество соседей при вычислении KNN. Основной гиперпараметр, кстати (!!!)
    n_components = 2, # сколько оставить компонентов; по дефолту - 2
    min_dist = 0.1    # минимальная дистанция, которую можно сохранять между точками в получающемся пространстве. 
    # Гиперпараметр. При увеличении начинает лучше улавливать общую структуру, но хуже - локальную
)

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
    estimator  = lgbm_mdl,  
    # k_features = 5,                                                  
    forward    = True,                                                  
    floating   = True,                                                
    verbose    = 0,
    cv         = 5
)

RecFeatAdd_module = RecursiveFeatureAddition(
    lgbm_mdl,
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


## GET DEFAULT MODULES
def get_default_modules():
    # eval
    return {
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
        'UMAP':        UMAP_module,
        
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
        'lgbm':        lgbm_mdl
    }