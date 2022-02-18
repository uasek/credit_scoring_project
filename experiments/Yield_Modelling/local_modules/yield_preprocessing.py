import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn import model_selection

def check_constant_column(column):
    if len(column.unique()) == 1:
        return True
    else:
        return False
    
def return_non_constant_table(table):
    constantness_of_columns = (table.apply(check_constant_column, axis = 0) == False)
    constant_columns = table.columns[constantness_of_columns]
    return table[constant_columns]

def return_constant_columns(table):
    non_constantness_of_columns = table.apply(check_constant_column, axis = 0)
    return np.array(table.columns[non_constantness_of_columns])

def outliers_filter(X, y, y_names = [], condition = ""):
    
    if isinstance(y, dict):
        y_names = list(y.keys())
        y = list(y.values())
    if (type(y[0]) == "float64") | (type(y[0]) == "int8"):
        y = [y]
        y_names = [y_names]
    
        
    for i in range(len(y)):
        X[y_names[i]] = y[i]
        
    desired_index = X[eval(condition)].index
#     print(X[eval(condition)])

    X = X.drop(y_names, axis = 1)

#     print(desired_index)
#     print(X)
#     print(X.iloc[desired_index])
    y_filtered = []
    for i in y:
        y_filtered.append(i[desired_index])
    
    return X.loc[desired_index], dict(zip(y_names, y_filtered))

def group_applier(df, supplementary_table, groups, suffixes):
    df_no_outliers = df.merge(supplementary_table, on = groups, how = "left", suffixes = ("", "".join(suffixes)))
    return df_no_outliers

def grouper(df_no_outliers, dependent_variable, groups = ["Year", "Month", "Moodys"], suffixes = ["_year", "_month", "_moo"], verbose = True):

    supplementary_table = df_no_outliers[[dependent_variable] + groups].groupby(groups).mean()
    
    df_no_outliers = group_applier(df_no_outliers, supplementary_table, groups, suffixes)
    
    if verbose == True:
        print((np.abs(df_no_outliers[dependent_variable] - df_no_outliers[dependent_variable + "".join(suffixes)])).mean())
    
#     print("".join(suffixes))
#     print(df_no_outliers[dependent_variable])
#     print(df_no_outliers[dependent_variable + "".join(suffixes)])
    return {"predictions" : (df_no_outliers[dependent_variable + "".join(suffixes)]), "map_table" : supplementary_table}