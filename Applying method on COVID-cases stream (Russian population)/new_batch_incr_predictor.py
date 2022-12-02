import pandas as pd
from functions_new_appr.paths import *
from functions_new_appr.other_functions import *
from functions_new_appr.dataset_preprocessing import *
from functions_new_appr.clustering_states_batches_time_intervals import *
from functions_new_appr.print_condition_graphs import *
from functions_new_appr.modeling_process_as_dynamic_network import *
from functions_new_appr.compare_methods_predict_outcomes import *
import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier
import os
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error,mean_absolute_error

"""
Class for create batch incremental model for stream of batches
"""


class new_bath_incr_predictor:
    def __init__(self,
                 data: pd.DataFrame,
                 target: str,
                 features: list,
                 time_intervals: list,
                 marker_date_start: str = "admission_date",
                 # marker_date_end: str = "discharge_date",
                 marker_end_epizode='end_epizode',
                 marker_t_point: str = "t_point",
                 marker_conds_dates: str = "date_cond_observ",
                 task_name: str = "task"
                 ):

        self.marker_conds_dates = marker_conds_dates
        self.time_intervals = time_intervals
        self.full_data = data
        self.features = features
        self.target = target
        self.number_intervals = len(time_intervals)
        self.metrics = {"stat": [], "incr": [], "proposed": [], "kl": [], "accur": []}
        self.other_inform = [target, marker_date_start, marker_conds_dates, marker_t_point,marker_end_epizode]
        self.task_name = task_name

        self.control_features = []
        self.static_features = []
        self.dinamic_features = []
        self.regression_targets = []
        self.classification_targets = []

        for col in self.full_data.columns:
            if col.endswith('stat_control'):
                self.control_features.append(col)
            elif col.endswith('dinam_fact'):
                self.dinamic_features.append(col)
            elif col.endswith('stat_fact'):
                self.static_features.append(col)
            elif col.endswith('tar'):
                if self.full_data[col].nunique() < 10:
                    self.classification_targets.append(col)
                else:
                    self.regression_targets.append(col)

        if self.target in self.regression_targets:
            self.model = XGBRegressor
            self.metric = mean_absolute_error
            self.type_task = "reg"
            self.name_metric="MAE score"
        else:
            self.model = XGBClassifier
            self.metric = f1_score
            self.type_task = "classif"
            self.name_metric = "F1 score"

    """
    Functions for creating batches for intervals in dynamics, when interval changes, new information 
    appending to old batch 
    """

    def create_batches_for_intervals(self) -> dict:
        batches = {}
        num_intervals = len(self.time_intervals)
        for inter in range(num_intervals):
            inter_borders = self.time_intervals[inter]

            id_processes_in_batch = self.full_data[(self.full_data[self.marker_conds_dates] >= inter_borders[0]) & (
                    self.full_data[self.marker_conds_dates] <= inter_borders[
                1])].query("t_point == 't_0'").index.unique()  # Calculating all cases begin in this interval

            for current_inter in range(inter, num_intervals):
                current_inter_borders = self.time_intervals[current_inter]
                batch = self.full_data[(self.full_data[self.marker_conds_dates] >= inter_borders[0]) & (
                        self.full_data[self.marker_conds_dates] <= current_inter_borders[1])][
                    self.features + self.other_inform].loc[id_processes_in_batch]
                batch.drop_duplicates(keep='first', inplace=True)
                batches[(inter, current_inter)] = batch
        return batches

    """Сreating all all known information at particular time point"""

    def all_known_inform_batches(self):
        first_batch = self.full_data[(self.full_data[self.marker_conds_dates] >= self.time_intervals[0][0]) & (
                self.full_data[self.marker_conds_dates] <= self.time_intervals[0][1])]
        known_batches = {0: first_batch}
        num_intervals = len(self.time_intervals)

        for inter in range(1, num_intervals):
            inter_borders = self.time_intervals[inter]
            batch = self.full_data[(
                                           self.full_data[self.marker_conds_dates] >= inter_borders[0]) & (
                                           self.full_data[self.marker_conds_dates] <= inter_borders[
                                       1])]  # calculating all indexes
            known_batches[inter] = pd.concat([first_batch, batch]).drop_duplicates()
        return known_batches

    def preprocessing(self, batch):
        return missing_value_imputations(batch,
                                         "MICE")  # !for each batch we use particular imputer, but we use information from other batches

    """
    Function for calculating weights of base learners for weighted ensemble of models
    adj_marts_moment_i - dict with all matrixes in moment i and past moments
    """

    def calculating_distance_matrix(self, adj_marts_moment_i, i):
        num_steps = len(adj_marts_moment_i)
        matrs_for_calc = [(inter, m.drop(['Recovered', 'Death'], axis=0).drop('start', axis=1)) for inter, m in
                          adj_marts_moment_i.items()]
        matrs_for_calc.sort(key=lambda T: T[0][0])
        matrs_for_calc = list(map(lambda T: T[1], matrs_for_calc))

        eucl_dist_m = distance_between_matrices(matrices=matrs_for_calc, type_dist="euclid", moment=i)
        # kl_dist_m = distance_between_matrices(matrices=matrs_for_calc, type_dist="kullback_leibler", moment=i)
        kl_dist_m=eucl_dist_m.copy() #patch

        return eucl_dist_m , kl_dist_m

    def weight_using_accuracy(self, i) -> pd.Series:
        X_test, y_test = self.train_batches[(i, i)][self.features], \
                         self.train_batches[(i, i)][self.target]

        accuracies = pd.Series()
        batches_moment_i = {inter: batch for inter, batch in self.train_batches.items() if inter[1] == i}

        for inter, batch in batches_moment_i.items():
            inter_0 = inter[0]
            X_train, y_train = batch[self.features], \
                               batch[self.target]

            y_pred = self.model().fit(X_train, y_train).predict(X_test)
            accuracies.loc[inter_0] = self.metric(y_test, y_pred)

        #if type_task reg we use RMSE, and need calc 1/RMSE for weighted
        if self.type_task=="reg":
            accuracies=1/accuracies

        # if metric is negative metric==zero
        accuracies = accuracies.apply(lambda metric: 0 if metric < 0 else metric)
        weights_accur = accuracies / accuracies.sum()
        return weights_accur

    """
    Function for calculating weights using distances between processes
    """

    def weights_of_BL_proposed(self, path_to_dist_matr=r"temporary_files_new_appr/graphs/adjacency matrices/"):
        weights = {}

        for i in range(self.number_intervals):
            # For each inter we calculating weights for each i
            # Calculating adjacency matrixes

            adj_marts_moment_i = {(past_i, i): pd.read_pickle(
                os.path.join(path_to_dist_matr, f"adjacency_matrix_{(past_i, i)}.pkl")) for
                past_i in range(i + 1)}

            eucl_dist_m, kl_dist_m = self.calculating_distance_matrix(adj_marts_moment_i, i)

            weights_by_eucl = (1 / eucl_dist_m.loc[i].drop(i)) / (1 / eucl_dist_m.loc[i].drop(i)).sum()
            weights_by_kl = (1 / kl_dist_m.loc[i].drop(i)) / (1 / kl_dist_m.loc[i].drop(i)).sum()

            # append weight if last interval and repeat the procedure
            weights_by_eucl.loc[i] = 1
            weights_by_kl.loc[i] = 1

            weights_by_eucl = weights_by_eucl / (weights_by_eucl).sum()
            weights_by_kl = weights_by_kl / (weights_by_kl).sum()

            # calculating weigths using accuracy of predicting incremental XGB on each batch
            weight_accur = self.weight_using_accuracy(i)  #
            weights[i] = {"proposed": weights_by_eucl, "kl": weights_by_kl, "accur": weight_accur}

        return weights

    """
    Function realize bootstrap empirical CI
    """

    def empirical_CI(self, values, alpha=0.2) -> list:
        values.sort()
        p = ((1.0 - alpha) / 2.0) * 100
        lower = max(0.0, np.percentile(values, p))
        p = (alpha + ((1.0 - alpha) / 2.0)) * 100
        if self.type_task =="classif":
            upper = min(1.0, np.percentile(values, p))
        else:
            upper = np.percentile(values, p)
        return [lower, upper]

    """
    Function for calculating quality of each type weighting for ensemble voicing for each moment of time with CI
    Input 0 weigths dict, Output-dict with lists of quality and two CI of it.
    """

    def ensambles_qialities(self, weights, n_iters=100, fraq_bootstrap=1) -> dict:

        for i in range(self.number_intervals - 1):  # -1 because we can't test quality for last moment
            # train ensemble
            batches_moment_i = {inter: batch for inter, batch in self.train_batches.items() if inter[1] == i}
            ens_i = [(inter[0], self.model().fit(batch[self.features], batch[self.target])) for inter, batch in
                     batches_moment_i.items()]  # self.model() !
            ens_i.sort(key=lambda t: t[0])

            test_data = self.train_batches[(i + 1, i + 1)]
            X_test, y_test = test_data[self.features], test_data[self.target]

            # y_pred ->np.array
            if self.type_task == "reg":
                y_pred = [m.predict(X_test) for i, m in ens_i]
            if self.type_task == "classif":
                y_pred = [m.predict_proba(X_test)[:, 1] for i, m in ens_i]

            # calculating qualities
            qual_i = {"proposed": [], "kl": [], "accur": []}
            for type_weighting in weights[i]:
                weights_ = weights[i][type_weighting]
                y_pred_weighted = [pred * weights_[i] for i, pred in enumerate(y_pred)]
                y_pred_aggregated = sum(np.array(y_pred_weighted))  # final predictions of all ansamble

                if self.type_task == "classif":
                    # probably to value
                    y_pred_aggregated = list(map(lambda val: 1 if val >= 0.5 else 0, y_pred_aggregated))

                # qual_i[type_weighting] += [self.metric(y_test, y_pred_aggregated)]  # add empirical metric

            # Calculate CI of qualities using bootstrap
            bs_qialities = {"proposed": [], "kl": [], "accur": []}
            for seed in tqdm_notebook(range(n_iters)):
                X_test, y_test = test_data.sample(frac=fraq_bootstrap,replace=True, random_state=seed)[self.features], \
                                 test_data.sample(frac=fraq_bootstrap,replace=True, random_state=seed)[self.target]

                # y_pred ->np.array
                if self.type_task == "reg":
                    y_pred = [m.predict(X_test) for i, m in ens_i]
                if self.type_task == "classif":
                    y_pred = [m.predict_proba(X_test)[:, 1] for i, m in ens_i]

                # calculating qualities
                for type_weighting in weights[i]:
                    weights_ = weights[i][type_weighting]
                    y_pred_weighted = [pred * weights_[i] for i, pred in enumerate(y_pred)]
                    y_pred_aggregated = sum(np.array(y_pred_weighted))  # final predictions of all ansamble

                    if self.type_task == "classif":
                        # probably to value
                        y_pred_aggregated = list(map(lambda val: 1 if val >= 0.5 else 0, y_pred_aggregated))

                    bs_qialities[type_weighting] += [self.metric(y_test, y_pred_aggregated)]

            # Append CI to qualities
            for type_weighting in weights[i]:
                qual_i[type_weighting] += [np.mean(bs_qialities[type_weighting])]
                qual_i[type_weighting] += self.empirical_CI(bs_qialities[type_weighting])
                self.metrics[type_weighting].append(qual_i[type_weighting])
                # !

    """
    Function for calculating qualities for all moments for static and incremental XGB, with cofidence intervals
    frac_bootstrap - percentage of observations for bootstrap
    """

    def one_learner_quality(self, shuffle=True, n_iters=100, fraq_bootstrap=1):
        # creating qualities
        for i in range(self.number_intervals - 1):  # -1 because we can't test quality for last moment
            batches_moment_i = {inter: batch for inter, batch in self.train_batches.items() if inter[1] == i}
            test_data = self.train_batches[(i + 1, i + 1)]

            final_quals = {'stat':[],'incr':[]}

            # calculate empirical metrics
            X_test, y_test = test_data[self.features], \
                             test_data[self.target]

            # test batch is next batch for i
            # for static XGB
            for_learn = pd.concat([*[b for inter, b in batches_moment_i.items()]])

            if shuffle:
                for_learn = for_learn.sample(frac=1, random_state=0)  # !

            X_train, y_train = for_learn[self.features], for_learn[self.target]
            self.stat_model = self.model().fit(X_train, y_train)

            y_pred_stat = self.stat_model.predict(X_test)
            # final_quals['stat'] = [self.metric(y_test, y_pred_stat)]

            # for dynamic XGB
            batches = [(inter, batch) for inter, batch in batches_moment_i.items()]
            batches.sort(key=lambda t: t[0][0])

            for inter, batch in batches:
                step = inter[0]
                if step == 0:
                    self.dinam_model = self.model().fit(batch[self.features],
                                                        batch[self.target])
                else:
                    self.dinam_model = self.dinam_model.fit(batch[self.features],
                                                            batch[self.target],
                                                            xgb_model=self.dinam_model)  # !

            y_pred_incr = self.dinam_model.predict(X_test)
            # final_quals['incr'] = [self.metric(y_test, y_pred_incr)]

            # calculate CI of metrics using bootstrap
            samples_quals = {"static": [], "incr": []}
            for seed in tqdm_notebook(range(n_iters)):
                X_test, y_test = test_data.sample(frac=fraq_bootstrap, replace=True, random_state=seed)[self.features], \
                                 test_data.sample(frac=fraq_bootstrap, replace=True, random_state=seed)[self.target]

                # test batch is next batch for i
                # for static XGB
                y_pred_stat = self.stat_model.predict(X_test)
                samples_quals['static'] += [self.metric(y_test, y_pred_stat)]

                # for dynamic XGB
                y_pred_incr = self.dinam_model.predict(X_test)
                samples_quals['incr'] += [self.metric(y_test, y_pred_incr)]

            # append metrics
            final_quals['stat'] += [np.mean(samples_quals["static"])]  # metric is mean of bootstrap metrics
            final_quals['incr'] += [np.mean(samples_quals["incr"])]

            final_quals['stat'] += self.empirical_CI(samples_quals["static"])  # !
            final_quals['incr'] += self.empirical_CI(samples_quals["incr"])

            self.metrics['stat'].append(final_quals['stat'])
            self.metrics['incr'].append(final_quals['incr'])

    """
    Shown quality graphs
    """

    def plot_metrics(self):
        x = list(map(lambda x:f"batch #{x} ({self.num_observ_batches[x]})",np.arange(self.number_intervals - 1)))  # -1 because we cannot calculate quality for the last interval
        fig = go.Figure()
        colors = ['pink', 'red', 'green', 'blue', 'orange', 'aquamarine', 'darkturquoise', "yellow"]
        for i, method in enumerate(self.metrics):
            if method!="kl":
                fig.add_trace(go.Scatter(x=x, y=[vals[0] for vals in self.metrics[method]],
                                         mode='lines+markers',
                                         name=f'{method}',
                                         line=dict(color=colors[i], width=2)
                                         ))

                fig.add_trace(go.Scatter(x=x, y=[vals[1] for vals in self.metrics[method]],
                                         mode='markers',
                                         name=f'CI {method}',
                                         marker=dict(symbol=141),
                                         line=dict(color=colors[i], width=0.5)
                                         ))

                fig.add_trace(go.Scatter(x=x, y=[vals[2] for vals in self.metrics[method]],
                                         mode='markers',
                                         name=f'CI {method}',
                                         marker=dict(symbol=141),
                                         line=dict(color=colors[i], width=0.5)
                                         ))
                fig.update_layout(
                    title=f"Метрики качества различных методов обучения адаптивных ML алгоритмов",
                    xaxis_title = "Stream of batches",
                    yaxis_title = f"{self.name_metric}",
                    legend_title = "Learning methods",
                )

        fig.show()
        fig.to_html(f"results/{self.task_name}_{self.target}_metrics.html")

    """
    Functions for calculating qualities of each models
    type_clust - if static using statistical model, in incremental - using incremental models
    """

    def make_expirement(self, type_clust="static") -> list:
        batches = self.create_batches_for_intervals()
        self.batches = {inters: self.preprocessing(batch) for inters, batch in batches.items()}
        if type_clust == "static":
            training_clustering_models(data=self.batches[(0, 0)],
                                       method="Kmeans",
                                       n_clusters=None,
                                       append_static_features=False)  # Тут можно бы попробовать сохранять всю выборку в памяти и использовать кластеризацию, но тогда придется все в памяти хранить

            # creating cells for all batches
            self.clusted_batches = {inter: creating_sequences_of_clusters(batch, inter) for inter, batch in
                                    self.batches.items()}

            # creating batches that we can use for predicting (that have end conditions)
            self.train_batches = {
                inter: batch.loc[batch.query("end_epizode == 1").index.unique()].query("t_point == 't_0'") for
                inter, batch in
                self.batches.items()}

            #calculating number of observations of each batches
            self.num_observ_batches = {inters[0]: batch.query("t_point == 't_0'").index.nunique() for inters, batch in batches.items() if
                                        inters[0] == inters[1]}

            # shown and save process graphs for all batches
            for inter, batch in self.clusted_batches.items():
                print_cond_graph(batch,  # !
                                 step=inter,
                                 time_point_marker="t_point")

        weights = self.weights_of_BL_proposed()

        # cacluclating metrics
        self.ensambles_qialities(weights)
        self.one_learner_quality()

        # plot metrics
        self.plot_metrics()
