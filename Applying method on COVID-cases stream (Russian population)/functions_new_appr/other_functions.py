import os
import plotly.figure_factory as ff
import plotly.express as px
import numpy as np
import scipy.stats as st
import shutil


if __name__ == "__main__":
    from paths import *
else:
    from functions_new_appr.paths import *

"""
Functions for clearing all directories with temporal files
"""
dir_with_temp_files = ['train_test_datasets',
                       r"trained_models/edges_weight_predict",
                       "data_for_steps",r"graphs/adjacency matrices",
                       r"graphs/html files"]

def clearing_directories(dir_with_temp_files):
    for folder in os.listdir(dir_with_temp_files):
        folder_path=os.path.join(dir_with_temp_files,folder)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(*[dir_with_temp_files,folder,filename])
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)

                if os.path.isdir(file_path):
                    folder_path = os.path.join(*[dir_with_temp_files,folder,filename])
                    shutil.rmtree(folder_path)
                    os.makedirs(folder_path)


            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

"""
Function for creating correlating matrix for train and test datasets
"""


def corr_matrix_plotly(data, title='Heatmap',
                       x_features=None,
                       y_features=None,
                       file_name="corr_matrix",
                       width=1500,
                       height=750,
                       x_border=None,
                       y_border=None,
                       ):
    corr = data.corr().round(2)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    df_mask = corr.mask(mask)

    # Выбираем какие колонки показывать по x какие по y
    if x_features is not None:
        df_mask = df_mask[x_features]

    if y_features is not None:
        df_mask = df_mask.loc[y_features]

    # Применяем границы x_border и y_border - удаляем из матрицы корреляции все строки (столбцы) в которых связей > border
    if x_border is not None:
        # Фильтруем столбцы
        df_mask = df_mask.loc[:,
                  list(filter(lambda f: df_mask[f].apply(abs).max() >= x_border, df_mask.columns.tolist()))]

    if y_border is not None:
        # Фильтруем строки
        df_mask = df_mask.loc[
            list(filter(lambda f: df_mask.loc[f].apply(abs).max() >= x_border, df_mask.index.tolist()))]

    fig = ff.create_annotated_heatmap(z=df_mask.to_numpy(),
                                      x=df_mask.columns.to_list(),
                                      y=df_mask.index.to_list(),
                                      colorscale=px.colors.sequential.RdBu,  # Turbo Blackbody Jet(лучшее)
                                      # Настройка палитры тут https://plotly.com/python/builtin-colorscales/
                                      hoverinfo="none",  # Shows hoverinfo for null values
                                      showscale=True, ygap=1, xgap=1
                                      )

    fig.update_xaxes(side="bottom")

    fig.update_layout(
        title_text=title,
        title_x=0.5,
        width=width,
        height=height,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_zeroline=False,
        yaxis_zeroline=False,
        yaxis_autorange='reversed',
        template='plotly_white'
    )

    # NaN values are not handled automatically and are displayed in the figure
    # So we need to get rid of the text manually
    for i in range(len(fig.layout.annotations)):
        if fig.layout.annotations[i].text == 'nan':
            fig.layout.annotations[i].text = ""

    fig.show()

    # html file

    # plotly.offline.plot(fig, filename=f'corr_matrix/{file_name}.html')

def confidence_interval(data: list) -> tuple:
    return st.t.interval(alpha=0.95, df=len(data) - 1, loc=np.mean(data), scale=st.sem(data))


