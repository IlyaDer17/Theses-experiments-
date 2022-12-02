import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
import re

"""method - one from two methods for inputing missing values
we inputing onli dinamic variable, becouse in static variable we haven't missing (?)
"""
def missing_value_imputations(data, method):
    imp=None
    dinamic_feats=[f for f in data.columns if "_dinam_fact" in str(f)]
    stat_feats = [f for f in data.columns if "_stat_fact" in str(f)]

    if method == "MICE":
        lr = LinearRegression()
        imp = IterativeImputer(estimator=lr, missing_values=np.nan,
                               max_iter=20, verbose=2, imputation_order='roman', random_state=0)

    elif method == "KNN":
        imp = KNNImputer(n_neighbors=5, add_indicator=False)

    new_dinamic_feats = imp.fit_transform(data[dinamic_feats+stat_feats])
    data[dinamic_feats+stat_feats]=new_dinamic_feats

    return data

def clearing_features_name(data:pd.DataFrame)->pd.DataFrame:
    return data.rename(columns = lambda x:re.sub('[^A-Za-zA-я0-9#%_]+', '', x))
