import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import pickle
from matplotlib import pyplot as plt
import os

if __name__ == "__main__":
    from paths import *
else:
    from functions_new_appr.paths import *


def one_hot_encoding(
        data,
        categor_feat,
        drop_old_feat=True,
        append_feat_name=True):
    categor_feat_frozen = categor_feat.copy()
    new_cat_feats_names = []

    for f in categor_feat_frozen:
        for v in data[f].unique():
            if append_feat_name:
                new_feat_name = f"{f} _ {v}"
            else:
                new_feat_name = str(v)
            new_cat_feats_names.append(new_feat_name)
            data[new_feat_name] = (data[f] == v).astype(float)  # !

        if drop_old_feat:
            data.drop(f, axis=1, inplace=True)

    return data, new_cat_feats_names


def vars_is_categorial(vars: list) -> bool:
    return len(set(vars)) <= 10 and not all(list(map(lambda x: str(x).strip().replace(".","").replace(",","").isdigit(), vars))) and len(set(vars)) != 2
    # we append last condition becouse we don't need encoding binary features


def max_min_scale(data, feats, mm_scaler=None):
    if mm_scaler == None:
        mm_scaler = MinMaxScaler(feature_range=(0, 1))
        mm_scaler.fit(data[feats].values)

    data[feats] = pd.DataFrame(
        mm_scaler.transform(data[feats].values),
        index=data.index)

    return data, mm_scaler


"""
Function for define optimal numbers of clusters using elbow method 
"""


def kmeans_elbow_viz(data):
    sum_of_squared_distance = []
    n_cluster = range(1, 25)

    for k in n_cluster:
        kmean_model = KMeans(n_clusters=k)
        kmean_model.fit(data)
        sum_of_squared_distance.append(kmean_model.inertia_)

    plt.plot(n_cluster, sum_of_squared_distance, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum of squared distances')
    plt.title('Elbow method for optimal K')
    plt.show()


"""
Function for clustering dinamic conditions plus age
Input parameters
- method of clustering (Kmeans or DBSCAN)
- parameters for methods - numbers of clustering for Kmeans and distance for DBSCAN
Output 
- model of clustering
"""


def training_clustering_models(data: pd.DataFrame,
                               method: str,
                               n_clusters: int or None,  # !
                               append_static_features: bool,
                               dir_temporary_files=dir_temporary_files) -> None:

    # check is there trained clustering models
    models_files = os.listdir(os.path.join(dir_temporary_files, "trained_clust_models"))
    clust_models_files = ["mm_scaler.pkl",
                          "model_clust.pkl",
                          "features_with_right_order.pkl",
                          "new_cat_feats.pkl",
                          "count_feats.pkl",
                          "cat_feats.pkl"]

    # if all(map(lambda f: f in models_files, clust_models_files)):
    #     retrain = input("Trained models for clustering are found. Do you want retrain it? yes/no : ") #for fast testing code
    # else:
    #     retrain="yes"
    retrain = "yes"

    if retrain=="yes":
        cols = data.columns.to_list()
        din_feats = np.array(cols)[list(map(lambda col_name: '_dinam_fact' in col_name, cols))].tolist()

        if append_static_features:
            din_feats += np.array(cols)[list(map(lambda col_name: '_stat_fact' in col_name, cols))].tolist()

        # dividing on categorical and continuous variables
        cat_feats = [f for f in din_feats if vars_is_categorial(data[f])]
        count_feats = [f for f in din_feats if not vars_is_categorial(data[f])]

        # One hot encoding
        data, new_cat_feats = one_hot_encoding(
            data,
            categor_feat=cat_feats,
            drop_old_feat=False,
            append_feat_name=True)

        # Scaling
        data, mm_scaler = max_min_scale(data, new_cat_feats + count_feats)

        # Clustering
        if method == "Kmeans":
            if n_clusters is None:
                kmeans_elbow_viz(data[new_cat_feats + count_feats])
                # n_clusters = int(input("Input the number of clusters as unique process states: "))
                n_clusters=5

            model_clust = KMeans(n_clusters=n_clusters, random_state=0).fit(data[new_cat_feats + count_feats])

        # Saving trained models and features for clustering with rihgt order
        with open(os.path.join(dir_temporary_files, "trained_clust_models/mm_scaler.pkl"), "wb") as f:
            pickle.dump(mm_scaler, f)

        with open(os.path.join(dir_temporary_files, "trained_clust_models/model_clust.pkl"), "wb") as f:
            pickle.dump(model_clust, f)

        with open(os.path.join(dir_temporary_files, "trained_clust_models/features_with_right_order.pkl"), "wb") as f:
            pickle.dump(cat_feats + count_feats, f)

        with open(os.path.join(dir_temporary_files, "trained_clust_models/new_cat_feats.pkl"), "wb") as f:
            pickle.dump(new_cat_feats, f)

        with open(os.path.join(dir_temporary_files, "trained_clust_models/count_feats.pkl"), "wb") as f:
            pickle.dump(count_feats, f)

        with open(os.path.join(dir_temporary_files, "trained_clust_models/cat_feats.pkl"), "wb") as f:
            pickle.dump(cat_feats, f)


"""
calculating centroides of all clusters
Input - data - dataframe with sequences, feat_clust - признаки для кластеризации, cluster_marker - features with clusters
"""


def get_clusters_centroides(data, feats_clust, cluster_marker="cell"):
    centroid = {}
    for clust in data[cluster_marker].unique():
        centroid[clust] = data[data[cluster_marker] == clust][feats_clust].mean().to_dict()
    return centroid


"""
creating sequences of clusters
"""

def creating_sequences_of_clusters(data: pd.DataFrame, iter: int):
    # Extract models
    with open(os.path.join(dir_temporary_files, "trained_clust_models/mm_scaler.pkl"), "rb") as f:
        mm_scaler = pickle.load(f)

    with open(os.path.join(dir_temporary_files, "trained_clust_models/model_clust.pkl"), "rb") as f:
        model_clust = pickle.load(f)

    with open(os.path.join(dir_temporary_files, "trained_clust_models/features_with_right_order.pkl"), "rb") as f:
        feats_right_order = pickle.load(f)

    with open(os.path.join(dir_temporary_files, "trained_clust_models/new_cat_feats.pkl"), "rb") as f:
        used_new_cat_feats = pickle.load(f)

    with open(os.path.join(dir_temporary_files, "trained_clust_models/count_feats.pkl"), "rb") as f:
        count_feats=pickle.load(f)

    with open(os.path.join(dir_temporary_files, "trained_clust_models/cat_feats.pkl"), "rb") as f:
        cat_feats=pickle.load(f)

    # One hot encoding
    data, new_cat_feats = one_hot_encoding(
        data,
        categor_feat=cat_feats,
        drop_old_feat=False,
        append_feat_name=True)

    # Append categorial features used for clustering if it is not existed
    for f in used_new_cat_feats:
        if f not in data.columns.tolist():
            data[f] = 0

    # Scaling
    data = max_min_scale(data,
                         used_new_cat_feats + count_feats,
                         mm_scaler)[0]

    # Clustering (cells becouse for other part of code it's it is suitable
    data["cells"] = pd.Series(model_clust.predict(data[used_new_cat_feats + count_feats]), index=data.index)

    # Calculating centroids of clusters (for scaled features)
    centroids = get_clusters_centroides(data, used_new_cat_feats + count_feats, cluster_marker="cells")

    with open(os.path.join(dir_temporary_files, f"data_for_steps/centroids_{iter}.pkl"), "wb") as f:
        pickle.dump(centroids, f)
    return data


"""
Task of this function - creating and saving datasets for flow samples with cells (clusters) sequenses
data - dataset, index - number of samples
date_marker - feature with dates for range samples for dividing datasets on cells
step - number of old samples for removing and new samples for adding on each step
batch_size - size of each batch
Output - list of datasets with sequenses
"""


def creating_sequences_for_flow_samples(data, time_intervals, date_marker="admission_date"):
    time_intervals.sort(key=lambda x: x[0])  # sort time intervals using start dates

    number_of_inters = len(time_intervals)  # append final part
    datasets_flow = [
        (data.loc[(data[date_marker] >= time_intervals[inter][0]) & (data[date_marker] < time_intervals[inter][1])],
         inter,
         time_intervals[inter]) for inter in range(number_of_inters)]

    # save batches
    for dataset in datasets_flow:
        data, inter = dataset[0], dataset[1]

        data.to_pickle(os.path.join(dir_temporary_files, f"data_for_steps/batch_{inter}.pkl"))

    new_datasets_flow = [
        (creating_sequences_of_clusters(dataset[0], dataset[1]), dataset[2]) for dataset in
        datasets_flow]  # dataset[1] - step for dataset
    return new_datasets_flow
