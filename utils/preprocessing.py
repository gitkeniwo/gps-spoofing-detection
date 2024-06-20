import pandas as pd
import pandasql as ps
import numpy as np

from sklearn.decomposition import PCA

from utils.haversine import haversine_component_distance

SELECT_ATTRIBUTES_WHELAN = ['GPS_lat', 'GPS_long', 'vx', 'vy', 'ax', 'ay']


def data_preprocessing(filepath: str, selected_attributes = SELECT_ATTRIBUTES_WHELAN, trace_num: int=0, method="haversine") -> pd.DataFrame:
    """
    Deals with raw data and returns a processed DataFrame with selected attributes as designated by `selected_attributes`.
    
    Parameters
    ----------
    method : str
        The method to use for data preprocessing. Default is "haversine".
        Possible values: "haversine", "pythagorean"
    """
    
    df = pd.read_csv(filepath)

    #filter out the anchor points
    df = df[df['Anchor_Number'] == 0]

    # Filter out the rows whose change of position is not reflected in the coordinates
    stmt = """SELECT * 
    FROM df
    WHERE Time in (
        SELECT min(Time) 
        FROM df
        GROUP BY GPS_lat, GPS_long 
        )
    """

    df = ps.sqldf(stmt, locals())

    # compute velocity
    
    if method == "haversine":
        haversine_dist_lat, haversine_dist_lon = [], []

        for rows in range(1, len(df)):
            dist =  haversine_component_distance(
                (df.loc[rows-1, 'GPS_lat'], df.loc[rows-1, 'GPS_long']), 
                (df.loc[rows, 'GPS_lat'], df.loc[rows, 'GPS_long'])
                )         
            haversine_dist_lat.append(dist[0])
            haversine_dist_lon.append(dist[1])
            
        df['vx'] = pd.Series(haversine_dist_lat) / df.Time.diff().dropna().reset_index(drop=True)
        df['vy'] = pd.Series(haversine_dist_lon) / df.Time.diff().dropna().reset_index(drop=True)
        df.dropna(inplace=True)
        
    elif method == "pythagorean":
        
        df['vx'] = df.GPS_long.diff() / df.Time.diff()
        df['vy'] = df.GPS_lat.diff() / df.Time.diff()
        df.dropna(inplace=True)

    # compute acceleration
    df['ax'] = df.vx.diff() / df.Time.diff()
    df['ay'] = df.vy.diff() / df.Time.diff()
    df.dropna(inplace=True)
    
        # 0-1 normalization
    def zero_one_normalization(df):
        return (df - df.min()) / (df.max() - df.min())
    for col in ['vx', 'vy', 'ax', 'ay']:
        df[col] = zero_one_normalization(df[col])

    # selected_attributes = ['GPS_lat', 'GPS_long', 'Time', 'vx', 'vy', 'ax', 'ay', 'dBm']
    
    df = df[selected_attributes]
    
    if trace_num:
        df['trace'] = np.ones(df.shape[0]).astype(int) * trace_num
    
    return df 


def zero_one_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """
    zero_one_normalization is a function that normalizes the input DataFrame to the range of 0 to 1.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
     
    """
    return (df - df.min()) / (df.max() - df.min())


# PCA_transformer

N_COMPONENTS = 3

def pca_transform(df: pd.DataFrame, n_components: int = N_COMPONENTS):
    
    """
    pca_transform is a function that performs PCA on the input DataFrame and returns the transformed DataFrame.

    Returns
    -------
    pd.DataFrame, PCA
        A tuple of the transformed DataFrame and the PCA object.
    """
    
    pca = PCA(n_components=n_components)
    pca.fit(df)
    pca_result = pca.transform(df)
    df['pca-one'] = pca_result[:,0]
    df['pca-two'] = pca_result[:,1]
    
    if n_components == 3:
        df['pca-three'] = pca_result[:,2]
    
    #normalize the pca results
    df['pca-one'] = zero_one_normalization(df['pca-one'])
    df['pca-two'] = zero_one_normalization(df['pca-two'])
    
    if n_components == 3:
        df['pca-three'] = zero_one_normalization(df['pca-three'])
        return df[['pca-one', 'pca-two', 'pca-three']].copy(deep=True), pca
    
    return df[['pca-one', 'pca-two']].copy(deep=True), pca



def add_traces(df, num) -> pd.DataFrame:
    """
    Adds a `trace` column to the DataFrame.
    """
    # df['traces'] = np.ones(pca_dfs[0].shape[0]) * num
    df_new = df.copy(deep=True)
    df_new['trace'] = np.ones(df.shape[0]) * num
    return df_new