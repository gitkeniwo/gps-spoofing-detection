import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_trace(df: pd.DataFrame, mode: str="position-only"):
    """
    plot_trace is a function that plots the trace of the vehicle.

    Parameters
    ----------
    df : _type_
        df is a pandas DataFrame that contains the trace data.
    mode : str
        mode is a string that specifies the type of visualization to be done.
    """
    
    if mode == "position-only":
        fig = make_subplots(rows=1, cols=1)
    elif mode == "velocity":
        fig = make_subplots(rows=3, cols=2)

    fig.add_trace(
        go.Scatter(x=df['GPS_long'], y=df['GPS_lat']),
        row=1, col=1, 
    )

    if mode == "velocity":
        fig.add_trace(
            go.Scatter(x=df['Time'], y=df['vx']),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['Time'], y=df['vy']),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=df['Time'], y=df['ax']),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df['Time'], y=df['ay']),
            row=3, col=2
        )

    if mode == "position-only":
        fig.update_layout(height=400, width=400, title_text="Trace")
    else:
        fig.update_layout(height=900, width=800, title_text="Trace")
        
    fig.show()
    
    
    
def plot_pca(pca_dfs: pd.DataFrame, n_components: int=2):
    """
    plot_pca # use plotly to visualize the pca result
    """
    
    if n_components == 2:
        fig = px.scatter(pca_dfs, x='pca-one', y='pca-two', color='trace')

        # adjust dot size
        fig.update_traces(marker=dict(size=3))

        # adjust the figure width and height
        fig.update_layout(
            autosize=False,
            width=600,
            height=600,
        )
        fig.show()
        
    elif n_components == 3:
        # 3d scatter plot
        fig = px.scatter_3d(pca_dfs, x='pca-one', y='pca-two', z='pca-three', color='trace')

        # adjust dot size
        fig.update_traces(marker=dict(size=3))

        # adjust the figure width and height
        fig.update_layout(
            autosize=False,
            width=800,
            height=800,
        )

        # add border to points 
        fig.update_traces(marker=dict(line=dict(width=0.7, color='DarkSlateGrey')))
        fig.show()
        
    else:
        raise ValueError("n_components must be either 2 or 3")