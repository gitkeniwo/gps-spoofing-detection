import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_trace(df: pd.DataFrame, mode: str="position-only", 
               marker_size=2, x_range=(51.43, 51.56), y_range=(25.28, 25.39),
               colored: bool=False, 
               color_col: str="spoofed", 
               spoofed_color: str="red", 
               genuine_color: str="green"):
    """
    plot_trace is a function that plots the trace of the vehicle.

    Parameters
    ----------
    df : _type_
        df is a pandas DataFrame that contains the trace data.
    mode : str
        mode is a string that specifies the type of visualization to be done.
    marker_size : int

    x_range : tuple
    
    y_range : tuple
    """
    
    if mode == "position-only":
        fig = make_subplots(rows=1, cols=1)
    elif mode == "velocity":
        fig = make_subplots(rows=3, cols=2)

    if colored:
        colors = df[color_col].apply(lambda x: spoofed_color if x==1 else genuine_color)
        
        fig.add_trace(
            go.Scatter(x=df['GPS_long'], y=df['GPS_lat'], 
                        mode='markers',
                        marker=dict(color=colors)),
            row=1, col=1, 
        )

    else:
        fig.add_trace(
            go.Scatter(x=df['GPS_long'], y=df['GPS_lat']),
            row=1, col=1
        )
    # fix range of x and y axis
    fig.update_xaxes(range=[x_range[0], x_range[1]], row=1, col=1)
    fig.update_yaxes(range=[y_range[0], y_range[1]], row=1, col=1)
    
    # update marker size
    fig.update_traces(marker=dict(size=marker_size))


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
        
        fig.update_xaxes(title_text="trace", row=1, col=1)
        fig.update_xaxes(title_text="velocity_x", row=2, col=1)
        fig.update_xaxes(title_text="velocity_y", row=2, col=2)
        fig.update_xaxes(title_text="accelaration_x", row=3, col=1)
        fig.update_xaxes(title_text="accelaration_y", row=3, col=2)
        # fig.update_yaxes(title_text="accelaration_y", row=row, col=col)
        # fig.update_yaxes(title_text="accelaration_y", row=row+1, col=col)

    if mode == "position-only":
        fig.update_layout(height=400, width=400, title_text="Trace")
    else:
        fig.update_layout(height=900, width=800, title_text="Trace")
        
    fig.show()
 
 
 
 
def range_finder(spoofed_data, benign_data): 
    "decide the range of x and y axis for the plot"
    x_range_s, y_range_s = (spoofed_data.GPS_lat.min(), 
            spoofed_data.GPS_lat.max()), \
            (spoofed_data.GPS_long.min(), 
            spoofed_data.GPS_long.max())
            
    x_range_b, y_range_b = (benign_data.GPS_lat.min(), 
            benign_data.GPS_lat.max()), \
            (benign_data.GPS_long.min(), 
            benign_data.GPS_long.max())
            
    return (min(x_range_s[0], x_range_b[0]), 
            max(x_range_s[1], x_range_b[1])), \
            (min(y_range_s[0], y_range_b[0]), 
            max(y_range_s[1], y_range_b[1]))
 
 
def plot_spoofed_and_benign(spoofed_list, benign_list,
                            marker_size=2):
    
    len_trace = len(spoofed_list)
    
    subplot_titles = [f"Benign Trace {i+1}" for i in range(len_trace)] +\
                     [f"Spoofed Trace {i+1}" for i in range(len_trace)]
     
    fig = make_subplots(rows=2, cols=len_trace,
                        subplot_titles=subplot_titles,
                        )
    
    for i, (s, b) in enumerate([(i,j) for i,j in zip(spoofed_list, benign_list)]):
        
        y_range, x_range = range_finder(s, b)
        
        plot_spoofed_and_benign_trace(fig, row=1, col=i+1, spoofed=s, benign=b,
                                      x_range=x_range, y_range=y_range)
        

    fig.update_traces(marker=dict(size=marker_size))
    
    fig.update_layout(height=600, width=300*len_trace, title_text="Benign and Spoofed Traces")
    
    # disable legend
    fig.update_layout(showlegend=False)
    
    # save the figure to png
    fig.write_image("../outputs/img/traces.png")
    fig.show()
    
         
 
 
def plot_spoofed_and_benign_trace(fig: go.Figure, row: int, col: int, spoofed: pd.DataFrame, benign: pd.DataFrame,
                            marker_size=2, x_range=(51.43, 51.56), y_range=(25.28, 25.39),
                            colored: bool=True, color_col: str="spoofed",
                            spoofed_color: str="red", genuine_color: str="green"):
    
    if colored:
        colors_spoofed = spoofed[color_col].apply(lambda x: spoofed_color if x==1 else genuine_color)
        colors_benign = benign[color_col].apply(lambda x: spoofed_color if x==1 else genuine_color)
        
        fig.add_trace(
            go.Scatter(x=spoofed['GPS_long'], y=spoofed['GPS_lat'], 
                        mode='markers',
                        marker=dict(color=colors_spoofed)),
            row=row+1, col=col, 
        )
        
        fig.add_trace(
            go.Scatter(x=benign['GPS_long'], y=benign['GPS_lat'], 
                        mode='markers',
                        marker=dict(color=colors_benign)),
            row=row, col=col, 
        )
    
    else:
        fig.add_trace(
            go.Scatter(x=spoofed['GPS_long'], y=spoofed['GPS_lat']),
            row=row+1, col=col+1,
        )
        
        fig.add_trace(
            go.Scatter(x=benign['GPS_long'], y=benign['GPS_lat']),
            row=row, col=col,
        )
        
    # update subplot title

    # fig.update_xaxes(range=[x_range[0], x_range[1]], title_text='Longitude', row=row, col=col)
    fig.update_xaxes(title_text='Longitude', row=row+1, col=col)
    if col == 1:
        fig.update_yaxes( title_text='Latitude', row=row, col=col)
        fig.update_yaxes( title_text='Latitude', row=row+1, col=col)
 
    #update range
    fig.update_xaxes(range=[x_range[0], x_range[1]], row=row, col=col)
    fig.update_xaxes(range=[x_range[0], x_range[1]], row=row+1, col=col)
    fig.update_yaxes(range=[y_range[0], y_range[1]], row=row, col=col)
    fig.update_yaxes(range=[y_range[0], y_range[1]], row=row+1, col=col)
 
 
 
 
 
 
 
 
    
    
    
def plot_pca(pca_dfs: pd.DataFrame, n_components: int=2, path="../outputs/img/pca.png"):
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
    else:
        raise ValueError("n_components must be either 2 or 3")
        
    fig.write_image(path)
    fig.show()
        
        
    
    
    



    
def plot_cf_matrix(cf_matrix: np.ndarray, title='', path="../outputs/img/confusion_matrix.png"):
    """
    plot_cf_matrix # use seaborn to visualize the confusion matrix
    """
    
    LABELS = ["Malicious","Benign"]
    plt.figure(figsize=(4, 4))
    plt.tick_params(axis="x", labelsize=10)
    plt.tick_params(axis="y", labelsize=10)
    sns.heatmap(cf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, annot_kws={"size": 20}, fmt="d", cmap="Blues", linewidths=1, linecolor='black');
    plt.ylabel('True class', fontsize=10)
    plt.xlabel('Predicted class', fontsize=10)
    
    #set title
    plt.title(title, fontsize=12)
    
    plt.savefig(path)
    plt.show()