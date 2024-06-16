import pandas as pd
import numpy as np
import math

def haversine(lat, lon):
    # Radius of the Earth in kilometers
    R = 6371

    # Convert degrees to radians
    lat1 = math.radians(lat[0])
    lon1 = math.radians(lon[0])
    lat2 = math.radians(lat[1])
    lon2 = math.radians(lon[1])

    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance in kilometers
    distance = R * c

    return distance

def haversine_distance(point_1, point_2):
    # Radius of the Earth in kilometers
    R = 6371

    # Convert degrees to radians


    # Differences in coordinates
    dlat = point_1[0] - point_2[0]
    dlon = point_2[1] - point_1[1]

    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(point_1[0]) * math.cos(point_2[0]) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance in kilometers
    distance = R * c

    return distance

def haversine_component_distance(p1, p2):
    
    p3 = (p1[0], p2[1])
    
    return haversine_distance(p3, p2), haversine_distance(p1, p3)

