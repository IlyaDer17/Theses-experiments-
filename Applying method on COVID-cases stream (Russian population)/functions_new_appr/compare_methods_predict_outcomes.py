import pandas as pd
from random import seed
import numpy as np
from tqdm import tqdm_notebook
import plotly.express as px

if __name__ == "__main__":
    from paths import *
    from other_functions import confidence_interval
else:
    from functions_new_appr.paths import *
    from functions_new_appr.other_functions import confidence_interval

from xgboost import XGBClassifier, XGBRegressor
# from catboost import CatBoostClassifier, CatBoostRegressor
# from lightgbm import LGBMClassifier, LGBMRegressor
# from sklearn.metrics import mean_squared_error, f1_score

def drug_feats_to_int(data):
    drugs_feats = [c for c in data.columns if "_drug" in c]
    data[drugs_feats] = data[drugs_feats].applymap(int)
    return data

"""
Function for build dataset for train predictive models for interval "inter" using only other_inter
for that dist(inter_network,other_inter_network)<=border

Input dist_border of distances between networks of intervals for add interval to dataset
Output - set of pickle files that to data_for_steps
"""

# change similar steps (in terms of distance between process networks)
def creating_data_for_learn(dist_border: float,
                            number_inters: int,
                            type_dist="euclid",
                            ) -> None:

    networks_dists_matrix = pd.read_pickle(os.path.join(dir_with_samples_for_steps,
                                                        f"{type_dist}_distances_between_networks_matrix.pkl")
                                           )

    for inter in range(number_inters):
        # Creating datasets for learn using my approach
        similar_inters = networks_dists_matrix[(networks_dists_matrix[inter] <= dist_border
                                                ) & (networks_dists_matrix.index.map(int) <= inter)].index.tolist()  # ! index int?

        proposed_appr_data = pd.DataFrame()
        for inter in similar_inters:
            proposed_appr_data = pd.concat([proposed_appr_data,
                                            drug_feats_to_int(pd.read_pickle(os.path.join(dir_with_samples_for_steps,
                                                                                          f"batch_{inter}.pkl")))])

        proposed_appr_data.drop_duplicates(inplace=True)
        proposed_appr_data.to_pickle(os.path.join(dir_with_samples_for_steps, f"proposed_appr_data_{inter}.pkl"))

        # Creating datasets for learn using all past data
        interns_for_learn = [i for i in range(number_inters) if i <= inter]
        all_past_inters_data = pd.DataFrame()
        for inter in interns_for_learn:
            all_past_inters_data = pd.concat([all_past_inters_data,
                                              drug_feats_to_int(pd.read_pickle(os.path.join(dir_with_samples_for_steps,
                                                                                            f"batch_{inter}.pkl")))])
        all_past_inters_data.drop_duplicates(inplace=True)
        all_past_inters_data.to_pickle(os.path.join(dir_with_samples_for_steps, f"all_past_inters_data_{inter}.pkl"))


"""
Function for incremental training model
Input - data and model for learning and testing
"""

"""
function for calculating metric of model usint inctemental or statical learning methods
Input 
- model - model for learn
- x_train,y_train,x_test,y_test, - data for learn 
- numbers_batches - number of batches for learning
- metric 
Output
- tuples with averadge metrics and CI95% for this metrics for two intervals
"""


def calc_metric_inremental_learning(x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame,
                                    y_test: pd.DataFrame, model_name: str,
                                    model, metric, numbers_batches=1) -> dict:
    # creating batches for incremental learning
    batch_size = x_train.shape[0] // numbers_batches
    batches = []
    for i_batch in range(numbers_batches):
        first_batch_ind = i_batch * batch_size
        last_batch_ind = first_batch_ind + batch_size

        if i_batch + 1 == numbers_batches:
            last_batch_ind = x_train.shape[0]  # for last batch we get all remaining samples

        batches.append((
            x_train[first_batch_ind:last_batch_ind],
            y_train[first_batch_ind:last_batch_ind]))

    # incremental learning
    for ind, batch in enumerate(batches):
        batch_x, batch_y = batch
        if ind == 0:
            current_model = None
        else:
            current_model = model

        if "XGB" in model_name:
            model = model.fit(batch_x,
                              batch_y,
                              xgb_model=current_model)
        else:
            model = model.fit(batch_x,
                              batch_y,
                              init_model=current_model)

    # calculating metric
    y_pred = model.predict(x_test)
    incr_mert = metric(y_test, y_pred)
    return incr_mert


"""Mini functions for extracting models using models name from tatgets dict"""


def get_model(model_name, targets):
    all_models_dict = {}  # dict with all names of models
    for tar in targets:
        all_models_dict.update(targets[tar]['models'])

    return all_models_dict[model_name]


def calculate_metrics(
        data: pd.DataFrame,
        current_time_inter: tuple,
        model_name: str,
        target: str,
        metric,
        targets: dict,
        n_folds=100,  # 100
        test_size=0.3  # 33% of current patients to test
) -> dict:
    data = data.query("t_point == 't_0'")  # use only t_0
    data.admission_date = pd.to_datetime(data.admission_date, format='%Y-%m-%d')  # !

    data = drug_feats_to_int(data)
    current_data = data[
        (data.admission_date >= current_time_inter[0]) & (data.admission_date <= current_time_inter[1])
        ]

    n_current_patients = current_data.shape[0]
    n_cur_pat_test = int(n_current_patients * test_size)
    seed(0)  # ! may be problem in row 136 with that sample will not shown different tests

    # cross validation for calculating means of metrics
    incr_metrics = []
    static_metrics = []

    # drop features that we can't use for train
    feats_for_drop = ["t_point", "end_epizode", "waves", "admission_date"]

    for i in tqdm_notebook(range(n_folds)):
        test_data = current_data.sample(n_cur_pat_test)  # ! I could add random random_state
        train_data = data[~data.index.isin(test_data.index.to_list())]

        x_test, y_test = test_data.drop(list(targets.keys()) + feats_for_drop, axis=1), test_data[target]
        x_train, y_train = train_data.drop(list(targets.keys()) + feats_for_drop, axis=1), train_data[target]

        # incremental learning
        incr_metr = calc_metric_inremental_learning(x_train, y_train, x_test, y_test, model_name,
                                                    get_model(model_name, targets),
                                                    metric)

        # static learning
        train_data = train_data.sample(frac=1)  # we use .sample(frac=1) for shuffle datafraime

        x_test, y_test = test_data.drop(list(targets.keys()) + feats_for_drop, axis=1), test_data[target]
        x_train, y_train = train_data.drop(list(targets.keys()) + feats_for_drop, axis=1), train_data[target]

        model_stat = get_model(model_name, targets)
        y_pred = model_stat.fit(x_train, y_train).predict(
            x_test)

        static_metr = metric(y_test, y_pred)

        static_metrics.append(static_metr)
        incr_metrics.append(incr_metr)

    return {
        "incr_mean": np.mean(incr_metrics),
        "incr_CI95%": confidence_interval(incr_metrics),
        "static_mean": np.mean(static_metrics),
        "static_CI95%": confidence_interval(incr_metrics)
    }


"""
Function for compare approaches for predict treatment-goal indicators
First approach (proposed) - we use only intervals that similar with test interval
Second approach (analog) - we use all past intervals 
Third approach (analog) - we use only interval for predicting
We use three Gradient Boosting approaches for make comparison between 
Input - targets - dict targets-metrics, n_intervs - number of intervals,time_intervals - tuple of intervals for steps
Output - graphs for all targets (two for COVID dataset) with metrics of models 
"""


# # targets example
# targets={"duration_treatment_tar": {
#                                      "metric": mean_squared_error,
#                                      "type_task": "regression",
#                                      "models": [
#                                           ("XGBRegressor", XGBRegressor()),
#                                           ("CatBoostRegressor", CatBoostRegressor()),
#                                           ("LGBMRegressor", LGBMRegressor())]
#                                       },
#            "outcome_tar": {"metric": f1_score,
#                            "type_task": "classification",
#                            "models": [
#                                ("XGBClassifier", XGBClassifier()),
#                                ("CatBoostClassifier", CatBoostClassifier()),
#                                ("LGBMClassifier", LGBMClassifier())]
#                            }}


def compare_approaches_predict_outcomes(targets: dict,
                                        n_intervs: int,
                                        time_intervals: list,
                                        dist_border: float):
    files_name_for_apprs = ["proposed_appr_data", "all_past_inters_data", "batch"]

    qualities = pd.DataFrame(columns=["target", "model_name", "approach", "interval",
                                      "type_learning", "metric", "CI95%_left", "CI95%_right"])

    # target - model - appr - steps - value of metric
    metrics = {}
    for tar in tqdm_notebook(targets):
        tar_dict = {}

        models = targets[tar]['models']
        metric = targets[tar]['metric']

        for model_name, model in tqdm_notebook(models.items()):
            model_dict = {}

            for approach in tqdm_notebook(files_name_for_apprs):
                approach_dict = {}

                for inter in tqdm_notebook(range(n_intervs)):
                    train_data_path = os.path.join(dir_with_samples_for_steps,
                                                   f"{approach}_{inter}.pkl")
                    print(f"Exp for appr {approach} and interval {inter}")
                    data = pd.read_pickle(train_data_path)

                    # sort by data admission
                    data.admission_date = pd.to_datetime(data.admission_date, format='%Y-%m-%d')
                    data.sort_values("admission_date", inplace=True)

                    current_time_inter = time_intervals[inter]

                    # cacculatins two metric (incremental and classic) for combination target-approach-model for three models
                    inct_stat_metrics = calculate_metrics(data,
                                                          current_time_inter,
                                                          model_name,
                                                          tar,
                                                          metric,
                                                          targets)

                    # append result to qualities
                    for type_learning in ["incr", "static"]:
                        next_ind = qualities.shape[0] + 1
                        qualities.loc[next_ind] = [tar, model_name, approach, inter,
                                                   f"{type_learning}_learning",
                                                   inct_stat_metrics[f"{type_learning}_mean"],
                                                   inct_stat_metrics[f"{type_learning}_CI95%"][0],
                                                   inct_stat_metrics[f"{type_learning}_CI95%"][1]]

    # Metric dict is not need now
    #                 approach_dict[inter] = inct_stat_metrics
    #             model_dict[approach] = approach_dict
    #         tar_dict[model_name] = model_dict
    #     metrics[tar] = tar_dict
    #
    # with open(os.path.join(dir_metrics_files,
    #                        "compare_methods_predict_outcomes_metrics.pkl", "wb")) as file:
    #     pickle.dump(metrics, file)

    # creating aggregate name for methods
    approach_features_cols = ["model_name", "approach", "type_learning"]
    qualities['method_name'] = pd.Series(
        zip(*[qualities[f] for f in approach_features_cols]), index=qualities.index).apply(
        lambda Tupl_feats: "_".join(Tupl_feats))
    qualities.to_pickle(os.path.join(dir_metrics_files, f"qualities_{dist_border}.pkl"))


"""
Function for train models using datasets created using knowledges extracted from flow
"""




"""
Function for show plots of mertics for combination of models,
approaches of using historical data, incremental or non incremental models

Input 
- qualities file with metrics
qualities.columns=["target", "model_name", "approach", "interval","type_learning", "metric", "IC95%_left", "IC95%_right"]
Output - plot for each target and each model that shows different approaches for target predicting
For each targets we will have 3 model * 3 approaches * 2 types of learning = 18 lines
We could devide lines for different models, if for one models we don't have good result we can delite it.
"""


def show_plots_metrics_expirements():
    qualities_path = os.path.join(dir_metrics_files, "qualities.pkl")
    qualities = pd.read_pickle(qualities_path)

    # creating graphs for all methods for all targets
    for tar in qualities.target.unique():
        tar_qualities = qualities.query(f"target == '{tar}'")
        fig = px.line(tar_qualities,
                      x="interval",
                      y="metric",
                      color='method_name',
                      title=f"Qialities for target {tar}",
                      markers=True)
        fig.show()

    # creating particular graph for each model
    for tar in qualities.target.unique():
        tar_qualities = qualities.query(f"target == '{tar}'")
        for mod_name in tar_qualities.model_name.unique():
            tar_model_qualities = qualities.query(f"model_name == '{mod_name}'")
            fig = px.line(tar_model_qualities,
                          x="interval",
                          y="metric",
                          color='method_name',
                          title=f"Qialities for model {mod_name} and target {tar}",
                          markers=True)
            fig.show()


"""function for geting all dist borders"""


def get_all_possible_dist_borders():
    networks_dists_matrix = pd.read_pickle(os.path.join(dir_with_samples_for_steps,
                                                        "distances_between_networks_matrix.pkl"))

    return list(set(sum(networks_dists_matrix.values.tolist(), [])))
