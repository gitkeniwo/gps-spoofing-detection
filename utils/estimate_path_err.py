import numpy as np
from scipy.stats import expon
from utils.haversine import haversine


def estimate_path_err(Drx, expF, qthr):
    print(f'Estimating path error for {expF}... ')
    enXY = []
    i = 0

    while i < Drx.shape[0]:
        real_lat=Drx.loc[i,'GPS_lat']
        real_long=Drx.loc[i,'GPS_long']
        start=i
        # Find all rows with the same timepoint
        idx = Drx['Time'] == Drx.iloc[i]['Time']
        print(idx)
        # Jump to the row with a new time
        i += idx.sum()

        # Extract RSSI values associated with each anchor
        Drxt = Drx[idx]

        # Find unique rows based on specific columns (area, cell_id, cell_lat, cell_lon)
        Drxt_unique = Drxt.drop_duplicates(subset=[ 'CID', 'lat', 'lon'])

        # Compute the centroid from the anchor coordinates
        # [cell_lat, cell_lon, dBm]
        Drxtf = Drxt_unique[['lat', 'lon', 'dBm']].values

        # Compute weights
        x = np.sort(Drxtf[:, 2])[::-1]  # Sort in descending order
        xn = -(x - x[0])
        y = expon.pdf(xn, scale=expF)
        w = y / y.sum()

        # Real position as mean of the logged values
        # rpos = Drxt[['lat', 'lon']].mean().values

        # Estimated position as the weighted mean
        epos = np.dot(Drxtf[:, :2].T, w)

        # Distance (Km) between estimation and real position
        d = haversine([epos[0],real_lat],[epos[1],real_long])
        for j in range(start,i):
            Drx.loc[j,'e_lat']=epos[0]
            Drx.loc[j,'e_lon']=epos[1]
            Drx.loc[j,'difference']=d
        # nAnchors = Drxtf.shape[0]
        # enXY.append([Drxt.iloc[0]['Time'], *epos, *rpos, d, nAnchors])

    # enXY_df = pd.DataFrame(enXY, columns=['timepoint', 'est_lat', 'est_lon', 'real_lat', 'real_lon', 'distance', 'nAnchors'])
    return Drx