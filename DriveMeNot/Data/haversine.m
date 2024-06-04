function distance = haversine(cor1, cor2)
    % Radius of the Earth in kilometers
    R = 6371; 
    
    % Convert degrees to radians
    lat1 = deg2rad(cor1(1));
    lon1 = deg2rad(cor1(2));
    lat2 = deg2rad(cor2(1));
    lon2 = deg2rad(cor2(2));
    
    % Differences in coordinates
    dlat = lat2 - lat1;
    dlon = lon2 - lon1;
    
    % Haversine formula
    a = sin(dlat/2)^2 + cos(lat1) * cos(lat2) * sin(dlon/2)^2;
    c = 2 * atan2(sqrt(a), sqrt(1-a));
    
    % Distance
    distance = R * c;
end
