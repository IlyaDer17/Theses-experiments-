import pandas as pd
import os
from tqdm import tqdm_notebook

dir_metrics_files = "metrics"

"""
Function for creating quality tables with best border for model-target combinations.
Also we include best classic models for creating this experience
"""


def find_best_border_dists():
    # mearge all qualities to one file
    all_qialities = pd.DataFrame()
    get_border = lambda qual_file_name: int(qual_file_name.split('_')[1].rstrip(".pkl"))

    for qual_file_name in os.listdir(dir_metrics_files):
        border = get_border(qual_file_name)
        qual_table = pd.read_pickle(os.path.join(dir_metrics_files, qual_file_name))
        qual_table['border'] = border
        all_qialities = pd.concat([all_qialities, qual_table])

    #have all qialities, for all target and all model find border with best approach result
    #best result == maximum of point with the first place
    for tar in tqdm_notebook(all_qialities.target.unique()):
        for model_name in tqdm_notebook(all_qialities.model_name.unique()):
            # Check type of task - classif or regression
            if "Regressor" in model_name:
                type_task='r'
                func_for_get_best_metric=min
            else:
                type_task = 'c'
                func_for_get_best_metric = max

            tabl=all_qialities.query(f"target == {tar}").query(f"model == {model_name}")
            number_win_for_borders={b:0 for b in tabl.border.unique()}
            for border in tabl.border.unique():
                tabl_border=tabl.query(f"border == {border}")
                if type_task=='r':
                    best_appr=tabl_border.loc[tabl_border.idmin("metric")]["approach"]
                if type_task=='c':
                    best_appr = tabl_border.loc[tabl_border.idmax("metric")]["approach"]
                if best_appr=='proposed_appr_data':
                    number_win_for_borders[b]+=1

            best_proposed_model=tabl.query("approach == ")['metric']

