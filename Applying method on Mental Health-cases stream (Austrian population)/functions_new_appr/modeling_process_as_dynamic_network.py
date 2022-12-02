import pandas as pd
import os
import plotly.graph_objects as go
import numpy as np
import pickle
from itertools import combinations_with_replacement
import json
import plotly.express as px
import math
from tqdm import tqdm_notebook

if __name__ == "__main__":
    from paths import *
    from other_functions import corr_matrix_plotly, confidence_interval
else:
    from functions_new_appr.paths import *
    from functions_new_appr.other_functions import confidence_interval


def show_violin(data: list, title: str) -> None:
    fig = go.Figure(data=go.Violin(y=data, box_visible=True, line_color='black',
                                   meanline_visible=True, fillcolor='lightseagreen',
                                   opacity=0.6, x0=title))

    fig.update_layout(yaxis_zeroline=False)
    fig.show()


"""
Function shows plot with distance between network matrices and hidden factors (clinics, new strains)
that can influence on the process
Input - time_intervals - list of time intervals for painting
Output - graph distances, time intervals dates (end of intervals dates) and list of events
"""


def show_distance_betw_networks_plot(time_intervals: list, distances: list, events=None):
    # Выгружаем основной словарь и преобразуем его в нужный вид
    fig = go.Figure()
    timeline = [end_date for start_date, end_date in time_intervals]
    # fig.add_trace(go.Scatter(x=timeline, y=[0] + distances, name='Поступление', marker_line_color='rgb(255,99,20)',
    #                          marker_color='rgb(255,99,20)'
    #                          ))
    print(len([0] + distances), len(time_intervals))
    for index_inter in range(len(time_intervals)):
        start_date, end_date = time_intervals[index_inter]
        all_distances = [0] + distances
        dist = all_distances[index_inter]

        fig.add_trace(go.Scatter(x=[start_date, end_date], y=[dist] * 2, name=f'Time interval# {index_inter}',
                                 marker_line_color='green',
                                 marker_color='green'
                                 ))
        fig.update_layout(yaxis_title=f"Distance between neighbors' intervals",
                          xaxis_title="Date")

    # Name event - date event - line color - line_dash="dash"
    if events is not None:
        for event in events:
            name_event, date_event, line_color, dash = event
            fig.add_vrect(x0=pd.to_datetime(date_event, format="%d.%m.%Y"),
                          x1=pd.to_datetime(date_event, format="%d.%m.%Y"),
                          annotation_text=name_event, annotation_position="top left",
                          line_width=2, line_dash=dash, line_color=line_color,
                          annotation_textangle=90)  # !dash None

    fig.show()


def get_all_clust_sequences():
    file_sequences = [f for f in os.listdir(dir_with_sequences) if "edges_sequences" in f]
    all_seq = []
    for f in file_sequences:
        with open(os.path.join(dir_with_sequences, f), "rb") as file:
            seq = pickle.load(file)
        all_seq += seq
    return all_seq

"""
Function for calculating confidence intervals for distance between networks
Input - directories with saved adjacency matrices
Output - matrix of distance between matrices
"""

def CI_distance_between_matrices(matrices: list, type_dist: str) -> pd.DataFrame:
    steps = list(range(len(matrices)))
    distance_matrix = pd.DataFrame(columns=steps, index=steps)
    all_seq = get_all_clust_sequences()
    for step1, m1 in enumerate(matrices):
        for step2, m2 in enumerate(matrices):
            if type_dist == "euclid":
                dist = (np.linalg.norm((m1 - m2).abs()))
                distance_matrix.loc[step1, step2] = dist
            elif type_dist == "kullback_leibler":
                distance_matrix.loc[step1, step2] = kullback_leibler_dist(m1,
                                                                          m2,
                                                                          all_seq=all_seq)
    # show heatmap for steps
    matrix_round = lambda v: 0 if v == 0 else round(v, 4)
    fig = px.imshow(distance_matrix.applymap(matrix_round), text_auto=True)
    fig.show()

    # save matrix of distance between networks
    distance_matrix.to_pickle(os.path.join(dir_with_samples_for_steps,
                                           f"{type_dist}_distances_between_networks_matrix.pkl"))

    return distance_matrix

"""
We can't calculate kullback leibler distance with zero values in matrix
Each transation haven't zero probability in real process
This function append to each zero value in matrix small values, and change other weights 
for sum in each rows and each columns will be one
"""

def softmax(List,num_to_smooth=10**(-5)):
    new_list=np.array(list(map(lambda v: v + num_to_smooth, List)))
    return new_list/sum(new_list)

def smooth_zero_values(m: pd.DataFrame, num_to_smooth=10**(-5)) -> pd.DataFrame:
    for i in m.index:
        m.loc[i]=softmax(m.loc[i].to_list(),num_to_smooth)
    return m

"""
Calculating kullback leibler distances
We use sequences from first interval for 
calculating probabilities for each sequency using two compared 
probablistic graphs and calculating metric.
"""

def kullback_leibler_dist(m1, m2, all_seq):
    m1 = smooth_zero_values(m1) #!
    m2 = smooth_zero_values(m2)

    dist = 0
    considered_seqs = []
    for seq in all_seq:
        if seq not in considered_seqs:
            prob1 = math.prod([m1.loc[e] for e in seq]) #
            prob2 = math.prod([m2.loc[e] for e in seq])
            dist += prob1 * np.log(prob1 / prob2)
            considered_seqs.append(seq)
    return dist

"""
Function for creating heatmap of distances between network matrixes
Input - list of matrices for steps
Output - matrix of distance between matrices
"""

def distance_between_matrices(matrices: list, type_dist: str,moment=None) -> pd.DataFrame:
    steps = list(range(len(matrices)))
    if moment is None:
        moment=len(steps)

    distance_matrix = pd.DataFrame(columns=steps, index=steps)
    all_seq = get_all_clust_sequences()
    for step1, m1 in enumerate(matrices):
        for step2, m2 in enumerate(matrices):
            if type_dist == "euclid":
                dist = (np.linalg.norm((m1 - m2).abs()))
                distance_matrix.loc[step1, step2] = dist
            elif type_dist == "kullback_leibler":
                distance_matrix.loc[step1, step2] = kullback_leibler_dist(m1,
                                                                          m2,
                                                                          all_seq=all_seq)
    # show heatmap for steps
    matrix_round = lambda v: 0 if v == 0 else round(v, 4)
    fig = px.imshow(distance_matrix.applymap(matrix_round), text_auto=True)
    fig.show()

    # save matrix of distance between networks
    distance_matrix.to_pickle(os.path.join(dir_with_samples_for_steps,
                                           f"{type_dist}_{moment}_distances_between_networks_matrix.pkl"))

    return distance_matrix


"""
Сalculating variance of edges weights for dimamical network 
Input
- directory with dynamical network, set of networks for different steps
- time_intervals - list of tuples with start-end time intervals data
Output 
- plotly violinplot graph of weight network distributions, title mean and 95%IC
- graph of distances between matrices
- matrix of distance
"""


def variance_edges_weights(time_intervals: list):
    n_steps = len([f for f in os.listdir(dinamic_netwokr_dir) if "pkl" in f])
    matrices = []
    for step in range(n_steps):
        step_matrix = pd.read_pickle(os.path.join(dinamic_netwokr_dir, f"adjacency_matrix_{step}.pkl"))
        matrices.append(step_matrix.drop(['Recovered', 'Death'], axis=0).drop('start',
                                                                              axis=1))  # Drop lines in adgency matric with zero

    variances_matrix = [matrices[step + 1] - matrices[step] for step in range(n_steps - 1)]
    all_variances = [v for v in sum(sum([m.abs().values.tolist() for m in variances_matrix], []), []) if v != 0]
    matrices_variances = [[v for v in sum(m.abs().values.tolist(), []) if v != 0] for m in variances_matrix]
    means_values = [np.mean(m_vars) for m_vars in matrices_variances]

    euclidian_dists = [np.linalg.norm(m) for m in variances_matrix]  # distance between matrix

    mean_var = np.mean(all_variances)
    CI95 = confidence_interval(all_variances)
    # show_violin(all_variances, title="All Variances of condition transition probabilities for different time points")
    # print(f"mean_var {mean_var} CI95 {CI95}")

    # # show euclidian_dist plot for intervals with
    # show_distance_betw_networks_plot(time_intervals,
    #                                  euclidian_dists)
    #
    #
    # # show and save matrix of distances between features
    distance_between_matrices(matrices=matrices, type_dist="euclid")
    # distance_between_matrices(matrices=matrices, type_dist="kullback_leibler")
    border_dist_net = float(input("Input border of distance between networks using created graphs: "))
    return border_dist_net

"""
Сalculating variance of edges weights for dimamical network 
Input
- directory with dynamical network, set of networks for different steps
- time_intervals - list of tuples with start-end time intervals data
Output 
- plotly violinplot graph of weight network distributions, title mean and 95%IC
- graph of distances between matrices
- matrix of distance
"""


def new_variance_edges_weights(time_intervals: list):
    n_steps = len([f for f in os.listdir(dinamic_netwokr_dir) if "pkl" in f])
    matrices = []
    for step in range(n_steps):
        step_matrix = pd.read_pickle(os.path.join(dinamic_netwokr_dir, f"adjacency_matrix_{step}.pkl"))
        matrices.append(step_matrix.drop(['Recovered', 'Death'], axis=0).drop('start',
                                                                              axis=1))  # Drop lines in adgency matric with zero

    variances_matrix = [matrices[step + 1] - matrices[step] for step in range(n_steps - 1)]
    all_variances = [v for v in sum(sum([m.abs().values.tolist() for m in variances_matrix], []), []) if v != 0]
    matrices_variances = [[v for v in sum(m.abs().values.tolist(), []) if v != 0] for m in variances_matrix]
    means_values = [np.mean(m_vars) for m_vars in matrices_variances]

    euclidian_dists = [np.linalg.norm(m) for m in variances_matrix]  # distance between matrix

    mean_var = np.mean(all_variances)
    CI95 = confidence_interval(all_variances)
    # show_violin(all_variances, title="All Variances of condition transition probabilities for different time points")
    # print(f"mean_var {mean_var} CI95 {CI95}")

    # # show euclidian_dist plot for intervals with
    # show_distance_betw_networks_plot(time_intervals,
    #                                  euclidian_dists)
    #
    #
    # # show and save matrix of distances between features
    distance_between_matrices(matrices=matrices, type_dist="euclid")
    # distance_between_matrices(matrices=matrices, type_dist="kullback_leibler")
    border_dist_net = float(input("Input border of distance between networks using created graphs: "))
    return border_dist_net


"""
Append features from start of observation - medians and means of each dimanic and static features 
and append features about control process - frequences of including drugs in treatment process
"""


def append_start_observ_feats(step: int, edges_inform: dict,
                              full_dataset_path=str,
                              drug_frequency_border=0.2) -> None:
    batch_path = f"{dir_steps_inform}/batch_{step}.pkl"
    batch = pd.read_pickle(batch_path)

    full_data = pd.read_pickle(full_dataset_path)

    # select only t_0 points
    full_data = full_data.query("t_point == 't_0'")

    # select frequent drugs
    drugs_feats = [f for f in full_data.columns if 'drug' in f]
    frequency_drugs = {"start_observ_" + drug: full_data[drug].value_counts(normalize=True)[True] for drug in
                       drugs_feats if
                       (full_data[drug].value_counts(normalize=True)[True] >= drug_frequency_border)}

    # calculating means of object features for t_0 points
    object_features = [f for f in full_data.columns if ('_dinam_fact' in f) | ("_stat_fact" in f)]
    cat_feats = [f for f in object_features if full_data[f].unique().shape[0] <= 2]
    statistic_f = lambda f: full_data[f].mean() if f in cat_feats else full_data[
        f].median()  # calculating statistical informations about feats, median for count feats, mean for boolean feats
    start_observ_inform = {"start_observ_" + f: statistic_f(f) for f in object_features}

    edge_inform_items = edges_inform.items()
    for edge, inform in edge_inform_items:
        inform.update(frequency_drugs)
        inform.update(start_observ_inform)
        edges_inform.update({edge: inform})


"""
This function append centroide features for edges informations
Also this function append distance between centroids
"""


def append_centroides_feats(step: int, edges_inform: dict) -> None:
    # get centroid for step
    with open(f"{dir_steps_inform}/centroids_{step}.pkl", "rb") as f:
        centroid = pickle.load(f)

    centroid_feats = list(centroid[0])

    # replace centroid for nodes without it
    for node in ["Death", "Recovered", "start"]:
        centroid[node] = {f: -1000 for f in centroid_feats}  # !

    # append centroid features - centroids and distance between two centroids
    edge_inform_items = edges_inform.items()
    for edge, inform in edge_inform_items:
        source, target = edge

        # # append centroides informations
        # inform.update({"s_" + str(f): val for f, val in centroid[source].items()})
        # inform.update({"t_" + str(f): val for f, val in centroid[target].items()})

        # append euclidian distances between centroides
        s_centr, t_centr = np.array(list(centroid[source].values())), np.array(list(centroid[target].values()))

        inform["euclidian_dict"] = np.sqrt(
            np.sum(np.square(s_centr - t_centr)))  # Calculating euclidian distance between centroides
        edges_inform.update({edge: inform})


"""This function used for append feats for edges weights predicting
Input - conditions matrix
Output - edges_inform with appended nodes features

1. Observation probability of two nodes for one sequences
2. <-> only in the right order (source-target) (random walk imitation)
3.  <-> only in the not right order (target-source)
4. Averedge distance (time points) between source and targets (for each nodes to nearest nodes)
5. Frequency of edge, and of each nodes
5. RW imitation (optionaly) - frequency of edge, and of each nodes,
 number of edges with this two nodes (and in comparison from orders) 
"""


def min_dist_to_other_node(nodes, cell_sequences):
    n1, n2 = nodes

    min_dists_to_neigh = []
    for seq in cell_sequences:
        if (n1 in seq) & (n2 in seq):
            for n in seq:
                if n in {n1, n2}:
                    ind = seq.index(n)
                    if n == n1:
                        other_n = n2
                    else:
                        other_n = n1
                    min_dist = min([ind - other_ind for other_ind, node in enumerate(seq) if node == other_n])
                    min_dists_to_neigh.append(min_dist)
    if len(min_dists_to_neigh) > 0:
        return np.mean(min_dists_to_neigh)
    else:
        return 100


"""
function of calculating node features
"""


def append_node_features(step: int,
                         edges_inform: dict) -> None:
    # Extract cell sequences
    with open(f"{dir_steps_inform}/edges_sequences_{step}.pkl", "rb") as file:
        edges_sequences = pickle.load(file)

    cell_sequences = [[edge[0] for edge in seq] for seq in edges_sequences]
    n_sequence = len(cell_sequences)
    edge_inform_items = edges_inform.items()

    for edge, inform in edge_inform_items:
        s, t = edge

        # # Probability of two nodes for one sequences
        # inform['both_nodes_in_seq'] = sum([1 for seq in cell_sequences if (s in seq) & (t in seq)]) / n_sequence
        # # Probability of two nodes for one sequences right direction. P(t|s)
        # sequences_with_s = [seq for seq in cell_sequences if (s in seq)]
        # inform['both_nodes_in_seq_right_order'] = sum(
        #     [1 for seq in sequences_with_s if (t in seq[seq.index(s):])]) / n_sequence
        # # Probability of each of two nodes
        # inform['prob_s_in_seq'] = sum([1 for seq in cell_sequences if s in seq]) / n_sequence
        # inform['prob_t_in_seq'] = sum([1 for seq in cell_sequences if t in seq]) / n_sequence
        # Average distance from each node to nearest neighbor
        inform['mean_min_dist'] = min_dist_to_other_node(edge, cell_sequences)
        # Probability of edges in sequences (number of sequences with transitions/number of transitions)
        # inform['freq_of_edge'] = sum([1 for seq in edges_sequences if edge in seq]) / n_sequence
        # # Probability of edges in sequences (number of transitions/all_transitions)
        inform['freq_of_edge_trans'] = sum([1 for e in sum(edges_sequences, []) if e == edge]) / n_sequence
        edges_inform.update({edge: inform})


"""
In this function we creating datasets for predicting network dinamics
Input 
- directory with set of networks
- n_steps_for_predict - number of the past steps that used for predicting current step 
Output - created datasets of dicts with features that extracted from network, 
and aggregated dataset for train model for currently steps
"""


def extracted_node_features():
    n_steps = len(os.listdir(dinamic_netwokr_dir))
    for step in range(n_steps):
        m = pd.read_pickle(os.path.join(dinamic_netwokr_dir, f"adjacency_matrix_{step}.pkl"))

        nodes_for_edges_static_weights = {"Death",
                                          "Recovered",
                                          "start"}  # if edges include both nodes from this set - weight static=0

        # append features for prediction weights of transitions
        edges_inform = {(i, j): {"edge_weight": m.loc[i, j]} for i, j in
                        combinations_with_replacement(m.index.to_list(), r=2) if
                        (i not in ["Death", "Recovered"]) & (j != 'start') & (
                                (i, j) not in [("start", "Recovered"), ("start", "Death")])}

        append_centroides_feats(step, edges_inform)
        append_node_features(step, edges_inform)
        # append_start_observ_feats(step, edges_inform)

        # save dataset for predicting edges weights
        dataset = pd.DataFrame.from_dict(edges_inform, "index")
        dataset.index = dataset.index.to_flat_index()
        dataset.to_pickle(f"{dir_steps_inform}/data_weights_predict_step_{step}.pkl")


"""
Functions for trainded models for dinamic networks predictig and calculating qualities of this predicting
Input - used_n_past_nets - used number past networks for predictive
Output - set of quality graphs (for each step) - X number of steps for prediction Y - quality predictions
"""


def creating_train_test_datasets(used_n_past_nets=2):
    n_steps = len(os.listdir(os.path.join(dir_temporary_files, dinamic_netwokr_dir)))

    # creating datasets for training models
    # aggregate datasets for first {used_n_past_nets} networks and append one networks iterativle, with markers of network
    # to each dataset append Y vectors
    # creating y matrix

    # creating datasets for train pred model
    for y_step_test in range(used_n_past_nets + 1, n_steps):
        # save test vectors for each step
        step_df = pd.read_pickle(f"{dir_steps_inform}/data_weights_predict_step_{y_step_test}.pkl")
        prev_step_df = pd.read_pickle(f"{dir_steps_inform}/data_weights_predict_step_{y_step_test - 1}.pkl")
        step_df['edge_weight'].to_pickle(f"{dir_steps_inform}/edges_weights_step_{y_step_test}.pkl")

        # creating cols for this dataset
        cols = sum([list(map(lambda col: f"{col}_{lag}", step_df.columns.tolist())) for lag in
                    range(1, used_n_past_nets + 1)], []) + ["step_network_for_train"] + ['y']

        # func for drop lag for start_observ functions because this functions we get in current step without lags
        start_observ_funcs = [f for f in step_df.columns if "start_observ_" in f]
        start_observ_drop_lag = lambda col_name: "_".join(
            col_name.split("_")[:-1]) if "start_observ_" in col_name else col_name
        cols = list(set(map(start_observ_drop_lag, cols)))  # del lag for start observe funcs

        step_train_df = pd.DataFrame(columns=cols)

        # one iter  - one dataset with one vector of y for train
        prev_steps = list(range(y_step_test))

        # (used_n_past_nets+1  - batch size), number of iterations - the number of moving window, +(1) - plus first batch
        number_iter = len(prev_steps) - (used_n_past_nets + 1) + (1)

        # y_step_test
        for iter in range(number_iter):
            y_step_train = iter + used_n_past_nets
            y_step_train_df = pd.read_pickle(f"{dir_steps_inform}/data_weights_predict_step_{y_step_train}.pkl")
            prev_y_step_df = pd.read_pickle(f"{dir_steps_inform}/data_weights_predict_step_{y_step_train - 1}.pkl")
            edges_dict = y_step_train_df['edge_weight'].to_dict()
            prev_step_edges_dict = prev_y_step_df['edge_weight'].to_dict()
            prev_step_edges_dict.update({e: 0 for e in edges_dict if e not in prev_step_edges_dict})

            # create dataset
            y_step_dataset = y_step_train_df[start_observ_funcs].copy()  # use start_observ_ feats for predict
            y_step_dataset['y'] = y_step_dataset.index.map(
                lambda edge: prev_step_edges_dict[edge] - edges_dict[edge])  # !
            y_step_dataset["step_network_for_train"] = int(y_step_train)

            for x_step_train in range(y_step_train - used_n_past_nets, y_step_train):
                lag = (y_step_train - x_step_train)
                x_step_train_df = pd.read_pickle(f"{dir_steps_inform}/data_weights_predict_step_{x_step_train}.pkl")
                x_step_train_df.drop(start_observ_funcs, axis=1,
                                     inplace=True)  # don't use start observe from previous steps for predicting
                x_step_train_df.rename(
                    {f: f"{f}_{lag}" for f in x_step_train_df.columns},
                    axis=1, inplace=True)
                y_step_dataset = y_step_dataset.join(x_step_train_df)

            step_train_df = pd.concat([step_train_df, y_step_dataset])

        # Save datasets for train models
        right_features_order = step_train_df.drop("step_network_for_train",
                                                  axis=1).columns.to_list()  # for equal order for train and test samples
        step_train_df["step_network_for_train"] = step_train_df["step_network_for_train"].astype(int)
        step_train_df.to_pickle(f"train_test_datasets/data_train_step_{y_step_test}.pkl")

        # CREATING DATA FOR TESTING TRAINED MODELS
        # creating cols for this dataset
        step_test_df = step_df[start_observ_funcs].copy()
        edge_weights = step_df['edge_weight'].to_dict()

        # creating edges dictionary for edges weights predicting
        prev_step_edge_weights = prev_step_df['edge_weight'].to_dict()
        prev_step_edge_weights.update({e: 0 for e in edge_weights if e not in prev_step_edge_weights})

        step_test_df['y'] = step_test_df.index.map(lambda edge: prev_step_edge_weights[edge] - edge_weights[edge])

        # Create dataset for test quality of models (last models) for y_step_test
        test_batch = list(range(y_step_test - used_n_past_nets, y_step_test))
        for x_step_test in test_batch:
            lag = (y_step_test - x_step_test)
            x_step_test_df = pd.read_pickle(f"{dir_steps_inform}/data_weights_predict_step_{x_step_test}.pkl")
            x_step_test_df.drop(start_observ_funcs, axis=1,
                                inplace=True)  # don't use start observe from previous steps for predicting
            x_step_test_df.rename({f: f"{f}_{lag}" for f in x_step_test_df.columns if f not in start_observ_funcs},
                                  axis=1,
                                  inplace=True)
            step_test_df = step_test_df.join(x_step_test_df, how="left")

        # Save datasets for train models
        step_test_df[right_features_order].to_pickle(os.path.join(dir_temporary_files,
                                                                  f"train_test_datasets/data_test_step_{y_step_test}.pkl"))


"""
Function for show corr all corr matrix for all steps 
"""


def show_all_corr_matrixes(y_features, x_border):
    train_datasets = [f for f in os.listdir(os.path.join(dir_temporary_files, "train_test_datasets")) if "train" in f]
    datasets = [pd.read_pickle(os.path.join(dir_temporary_files,
                                            f"train_test_datasets/{f}")) for f in train_datasets]

    for ind, d in enumerate(datasets):
        try:
            corr_matrix_plotly(d, title=f"corr_matrix_{ind}", y_features=y_features, x_border=x_border)
        except:
            print(f"Для интервала {d} матрицу корреляций построить не удалось")
