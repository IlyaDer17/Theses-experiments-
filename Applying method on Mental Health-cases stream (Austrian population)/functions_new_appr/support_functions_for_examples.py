import pandas as pd
from tqdm import tqdm_notebook
import plotly.graph_objects as go
from datetime import timedelta

# Name event - date event - line color - line_dash="dash"
covide_events = [
    ("ClinRec #6", "24.04.2020", "#511CFB", None),
    ("ClinRec #7", "03.06.2020", "#511CFB", None),
    ("ClinRec #8", "03.09.2020", "#511CFB", None),
    ("ClinRec #9", "26.10.2020", "#511CFB", None),
    ("ClinRec #10", "08.02.2021", "#511CFB", None),
    ("Alpha strain", "01.12.2020", "#19D3F3", "dash")]


# show flow cases
def graph_flow_cases(data:pd.DataFrame,
                     title:str,
                     events:list
                     )->None:
    admission_to_hospital = data.admission_date.tolist()

    timeline = []
    admission = []

    date_to_datetime = lambda date: pd.to_datetime(date, format='%Y-%m-%d')
    for year in tqdm_notebook(range(2011, 2022)):
        for month in range(1, 13):
            for day in range(1, 32):
                number_admission = sum([1 for date in admission_to_hospital
                                        if (date_to_datetime(date).year == year) & (
                                                date_to_datetime(date).month == month)& (
                                                date_to_datetime(date).day == day)
                                        ])

                if number_admission > 0:
                    timeline.append(pd.to_datetime(f"{year}.{month}.{day}"))
                    admission.append(number_admission)

    fig = go.Figure()

    fig.add_trace(go.Bar(x=timeline, y=admission, name='title',
                         marker_line_color='rgb(255,99,20)',
                         marker_color='rgb(255,99,20)'))
    fig.update_layout(
        title={
            'text': title,
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'})

    if events is not None:
        for event in events:
            name_event, date_event, line_color, dash = event
            fig.add_vrect(x0=pd.to_datetime(date_event, format="%d.%m.%Y"),
                          x1=pd.to_datetime(date_event, format="%d.%m.%Y"),
                          annotation_text=name_event, annotation_position="top left",
                          line_width=2, line_dash=dash, line_color=line_color,
                          annotation_textangle=90)  # !dash None

    fig.show()


def create_intervals(long_inter=30, long_step=15, min_long_interv=25):
    date_to_format = lambda date: pd.to_datetime(date, format="%d.%m.%Y")
    wave_1, wave_2 = (date_to_format("13.05.2020"), date_to_format("21.07.2020")), (
    date_to_format("05.12.2020"), date_to_format("04.03.2021"))

    inters = []

    for wave in [wave_1, wave_2]:
        start_wave, end_wave = wave[0], wave[1]
        start_interv = start_wave
        while (start_interv >= start_wave) & (start_interv <= end_wave) & (
                (end_wave - start_interv).days >= min_long_interv):
            end_interv = start_interv + timedelta(days=long_inter)
            inters.append((start_interv, end_interv))
            start_interv = start_interv + timedelta(days=long_step)
    return inters