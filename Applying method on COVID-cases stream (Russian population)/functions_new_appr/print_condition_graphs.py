import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from itertools import product
import json
import pyvis
from collections import Counter
from pyvis.network import Network
import time
import pickle
import os
from tqdm import tqdm_notebook


if __name__ == "__main__":
    from paths import *
else:
    from functions_new_appr.paths import *

border_frequency_cell = 0.025  # we drop rare cell with freq less than border

"""Create adjency matrix.
Input - cells_sequences
Output - dict , keys - uniq_cells values - observed cells frequencies from data
"""


def calculating_frequency_cells(cells_sequences):
    number_sequences = len(cells_sequences)

    """
    transform cells_sequences - drop duplicates conditions for one samples all tuple to lists
    """

    cells_sequences = [list(set(sum(map(list, seq), []))) for seq in cells_sequences]
    numbers_cells_observation = Counter(sum(cells_sequences, []))

    del numbers_cells_observation['start']
    # all_cell_number = sum(list(dict(numbers_cells_observation).values())) use number_sequences instead of it now
    cell_frequencies = {cell: cell_number / number_sequences for cell, cell_number in
                        dict(numbers_cells_observation).items()}
    return cell_frequencies


"""Create adjency matrix.
Input - cells_sequences
Output - matrix , shape=number_feat*number_feat, each matrix cells - weights
"""


def create_adjacency_matrix(cells_sequences, border_frequency_cell=border_frequency_cell):
    uniq_cells = list(set(sum(map(list, sum(cells_sequences, [])), [])))
    adjacency_matrix = pd.DataFrame(columns=uniq_cells, index=uniq_cells)

    numbers_edges = Counter()
    for sequence in cells_sequences:
        numbers_edges += Counter(sequence)

    numbers_edges = dict(numbers_edges)

    # transform numbers to frequences
    for cell in uniq_cells:
        cell_source_edges = {pair_cells: numbers for pair_cells, numbers in numbers_edges.items() if
                             pair_cells[0] == cell}
        if len(cell_source_edges) != 0:
            numbers_edges_cell_source = sum(cell_source_edges.values())
            for pair_cells, numbers in cell_source_edges.items():
                numbers_edges[pair_cells] = numbers / numbers_edges_cell_source

    # creating adjacency matrix
    for s in adjacency_matrix.columns:
        for t in adjacency_matrix.index:
            try:
                adjacency_matrix.loc[s, t] = numbers_edges[(s, t)]
            except:
                adjacency_matrix.loc[s, t] = 0  # probability is empty

    # drop rare cells from matrix
    rare_cells = [c for c, f in calculating_frequency_cells(cells_sequences).items() if f < border_frequency_cell]
    adjacency_matrix.drop(rare_cells, axis=1, inplace=True)
    adjacency_matrix.drop(rare_cells, axis=0, inplace=True)

    return adjacency_matrix


def round_label(label):
    if not str(label).isalpha():
        label = round(label, 3)
    return label


def show_html_pyvis_graph(adjacency_matrix, step):
    g = Network(directed=True, height='750px', width='1250px', heading=f"Network for time interval #{step}")
    nodes_attributes = {"start": {"color": "blue", "size": 150,'value':100},
                        "Death": {"color": "red", "size": 100,'value':100},
                        "Recovered": {"color": "green", "size": 100,'value':100},
                        }

    for s in adjacency_matrix.columns:
        for t in adjacency_matrix.index:

            for node in [s, t]:
                if node not in nodes_attributes:
                    nodes_attributes.update({node: {"color": "grey", "size": 50,'value':50}})

            if adjacency_matrix.loc[s, t] != 0:
                g.add_node(str(s), value=nodes_attributes[s]['value'],
                           size=nodes_attributes[s]["size"], label=str(s),
                           color=nodes_attributes[node]['color'])
                g.add_node(str(t), value=nodes_attributes[t]['value'],
                           size=nodes_attributes[t]["size"], label=str(t),
                           color=nodes_attributes[node]['color'])
                g.add_edge(str(s), str(t), value=5, size=5, label=str(round_label(adjacency_matrix.loc[s, t])))

    g.show_buttons() #filter_=['physics']
    g.barnes_hut(gravity=-20000)
    g.show(f'graphs/html files/step_{step}.html')


"""
Function for transform sequence include three steps 
1. tratsform number to [number]
2. append outcome observation
3. append start-end conditions
  
"""


def transform_sequence(sequence, data, sampl, sample_outcome_feat, outcome_translater):
    if_num_to_list = lambda val: [val] if type(val) != type(pd.Series()) else val.to_list()
    get_sampl_outcome = lambda sampl: int(if_num_to_list(data.loc[sampl][sample_outcome_feat])[0])
    sampl_outcome = outcome_translater[get_sampl_outcome(sampl)]

    # sampl to [sampl], append sampl outcome and start-end points
    return ['start'] + if_num_to_list(sequence) + [sampl_outcome]


"""
Print condition graph
Input - data, cells, reference_intervals
time_point_marker - marker of time of each point epizode
Output - pyvis graph of conditions and table of conditions interpretations (using decision trees)
"""


def print_cond_graph(data,
                     step,
                     sample_outcome_feat="outcome_tar",
                     outcome_translater=('Recovered', 'Death'),
                     time_point_marker="t_point"):

    adjacency_matrix_path=os.path.join(dir_temporary_files,"graphs/adjacency matrices/")
    sequences_path = os.path.join(dir_temporary_files,f"data_for_steps")
    #If files for step is exist we don't recreating it
    if os.path.exists(os.path.join(adjacency_matrix_path,f"adjacency_matrix_{step}.pkl")) and os.path.exists(os.path.join(sequences_path,f"edges_sequences_{step}.pkl")):
        pass
    else:
        data[time_point_marker]=data[time_point_marker].apply(lambda point:point.lstrip("t_")).astype(int)
        data.sort_values(["admission_date",time_point_marker], inplace=True)

        path_sequences=os.path.join(dir_temporary_files,f"data_for_steps/edges_sequences_{step}.pkl")
        if os.path.exists(path_sequences):
            cells_sequences=pd.read_pickle(path_sequences)
        else:
            # tratsform number to [number]
            cells_sequences = [
                list(zip(transform_sequence(data.loc[sampl]['cells'], data, sampl, sample_outcome_feat, outcome_translater),
                         transform_sequence(data.loc[sampl]['cells'], data, sampl, sample_outcome_feat, outcome_translater)[
                         1:])) for sampl
                in tqdm_notebook(data.index)]

            # Save cell sequences
            with open(path_sequences, "wb") as file:
                pickle.dump(cells_sequences, file)

        # creating adjacency matrix
        adjacency_matrix = create_adjacency_matrix(cells_sequences)

        # save adjacency matrix
        adjacency_matrix.to_pickle(os.path.join(dir_temporary_files,f"graphs/adjacency matrices/adjacency_matrix_{step}.pkl"))
        adjacency_matrix.to_csv(os.path.join(dir_temporary_files,f"graphs/adjacency matrices/adjacency_matrix_{step}.csv"))

        # show graph
        show_html_pyvis_graph(adjacency_matrix, step)

