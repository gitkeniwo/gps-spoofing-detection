%% Example: sd = spoofdect(); sd = sd.loadtrace('hbku-HamadHospital.csv');  expF=25; sd = sd.estimatePathErr(expF, 0.9); sd = sd.run([25.33, 51.47], expF, false);  sd = sd.fp_analysis();  

classdef spoofdect2
    properties
        aDB            % Anchor DB
        Drx            % Acquired data + Anchor coordinates
        localDB        % Anchors in the path region (speed up computation)
        enXY           % Estimated coordinates for the moving node
        sa             % Anchor statistics
        spoof_time     % Time at which the spoofing begins
        spoof_dest     % Destination point for the spoofing attack
        SPOOF          % Spoofing in action
        spoof_traj     % Spoofed trajectory
        dest_reached   % Destination reached
        target_reached % Target reached
        curr_pos       % Current position
        spoof_thr      % Spoofing decision threshold (m)
        spoof_vector_a % Vector of spoofing events (actual)
        spoof_vector_s % Vector of spoofing events (simulated)
        spoof_begin    % Starting spoofing (n. events)
        spoof_init_pos % Spoof initial position
        spoof_pos      % Current spoof position
        detour         % Required distance to detect the spoofing event
        spoof_window   % Window size for the declaring a spoofing event 
        speed          % Speed of the node (quantile 50)
        speriod        % Sampling period (quantile 50)
        dist           % Length of the trip
        duration       % Trip duration
        step           % Average step size        
        false_positive % Number of false positive spofing event
        anchor_var     % Histogram of the number of anchors changing at every slot
        dist_spoof_cur % Distnace between the spoofed position and the actual one
        tracename      % Name of the trace
        nevents        % Number of events (different times)
        burstfreq_a    % Sequence of anomalies (frequency)
        burstfreq_s    % Sequence of anomalies (frequency)
        wDB            % WiFi DB
        Dw             % Traces from the WiFi 
        localwDB       % Anchors in the path region (speed up computation)
        ewXY           % Estimated coordinates for the moving node exploting the WiFi network
        sawifi         % BSSID statistics
        errWiFiGSM     % Error estimation considering GSM and WiFi
        distWifiGSM    % Distributions associated to a specific w (Error respect to the GPS position)
        burst          % burst lengths for selected w
        distES         % Distance between estimated and spoofed position
        disterrM       % Distance between estimated and spoofed position with malicious anchors
    end
    
    methods
        function obj = spoofdect2()
            %rng default;
            obj.aDB = [];
            obj.Drx = [];
            obj.localDB = [];
            obj.enXY = [];
            obj.spoof_time = -1;
            obj.spoof_dest = [];
            obj.SPOOF = false;
            obj.spoof_traj = [];
            obj.dest_reached = false;
            obj.target_reached = false;
            obj.spoof_thr = -1;
            obj.spoof_window = -1;
            obj.spoof_vector_a = [];
            obj.spoof_vector_s = [];
            obj.detour = -1;
            obj.spoof_init_pos = [];
            obj.curr_pos = [];
            obj.spoof_pos = [];
            obj.speed = -1;
            obj.speriod = -1;
            obj.dist = -1;
            obj.duration = -1;
            obj.false_positive = -1;
            obj.step = -1;
            obj.anchor_var = [];
            obj.dist_spoof_cur = [];
            obj.tracename = -1;
            obj.nevents = -1;
            obj.burstfreq_a = [];
            obj.burstfreq_s = [];
            obj.spoof_begin = -1;
            obj.wDB = NaN;
            obj.Dw = NaN;
            obj.localwDB = NaN;
            obj.ewXY = [];
            obj.sawifi = [];
            obj.errWiFiGSM = [];
            obj.distWifiGSM = NaN;
            obj.burst = {};
            obj.distES = [];
            obj.disterrM = [];
        end

        function obj = loadtrace(obj, filename)
            
            obj.tracename = filename;
            
            %% Load trace
            fprintf('Processing trace: %s\n', filename);
            filenamewithpath = sprintf('./Data/%s', filename);
            D = readtable(filenamewithpath);
            D = [D(:, 5).('Time'),...
                 D(:, 1).('GPS_lat'),...
                 D(:, 2).('GPS_long'),...
                 D(:, 3).('Network_lat'),...
                 D(:, 4).('Network_long'),...                 
                 D(:, 10).('LAC'),...
                 D(:, 9).('CID'),...
                 D(:, 12).('MNC'),...
                 D(:, 13).('dBm')];
                 
            % Removing lines with null lon/lat
            D = D(find(D(:, 2)), :); D = D(find(D(:, 3)), :);
        
            
            %% Load WiFi trace
            filenamew = strcat('WiFi_', filename);
            filenamewithpath = sprintf('./Data/%s', filenamew);
            if isfile(filenamewithpath)
                fprintf('Processing trace: %s\n', filenamew);
                filenamewithpath = sprintf('./Data/%s', filenamew);
                Tw = readtable(filenamewithpath);
                
                Dw.info = table2array(Tw(:, [1, 2, 5, 9]));
                Dw.bssid = cell2mat(Tw.('BSSID'));                
                obj.Dw = Dw;                                
            end                                    
            
            
            
            %% Load the cell database           
            str = sprintf('./Data/CellsDatabase.csv');
            anchorDB = readtable(str);
            obj.aDB = [anchorDB(:, 4).('area'),...
                   anchorDB(:, 5).('cell'),...
                   anchorDB(:, 7).('lon'),...
                   anchorDB(:, 8).('lat'),...
                   anchorDB(:, 3).('net')];

            %% Load the WiFi database
            str = sprintf('./Data/WiFiDatabase.csv');
            wifiDB = readtable(str, 'Delimiter', ',');
            wifiDB = rmmissing(wifiDB);
            
            wDB.bssid = cell2mat(wifiDB(:, 1).('BSSID')); 
            wDB.lonlat = [wifiDB(:, 2).('Lat'), wifiDB(:, 3).('Long')];
            obj.wDB = wDB;
               
            %% Adding anchor coodinates to D -> Drx
            % time, NodeGPS, NetGPS, Lac, Mcc, rssi, AGPS
            if ~isfile(strcat('./Data/', filename, '.DB'))
                disp('Missing pre-processed file. Generating it...');
                for i = 1:size(D, 1)
                    iaDB = ismember(obj.aDB(:, [1, 2, 5]), D(i, [6, 7, 8]), 'rows');
                    if isempty(find(iaDB))
                        fprintf('Missing anchor GPS coordinates: %i %i %i\n', D(i, 6), D(i, 7), D(i, 8));
                    else                        
                        obj.Drx = [obj.Drx; [D(i, :), obj.aDB(iaDB, 4), obj.aDB(iaDB, 3)]];                      
                    end
                end
                Drx = obj.Drx;
                save(strcat('./Data/', filename, '.DB'), 'Drx')
                disp('Done');
            else
                fprintf('Found pre-processed file. Importing data... ');
                load(strcat('./Data/', filename, '.DB'), '-mat', 'Drx'); obj.Drx = Drx;
                fprintf('Done.\n');
            end
            
            obj.nevents = size(unique(obj.Drx(:, 1)), 1);
            fprintf('Number of events: %i\n', obj.nevents);
            fprintf('Trace duration: %.2f [min]\n', (obj.Drx(end, 1) - obj.Drx(1, 1)) / 1000 / 60);
            fprintf('Number of operators: %i\n', length(unique(obj.Drx(:,8))));
                       
            % Estimating tot. trip distance and average speed
            a = D(:, [1, 2, 3]); 
            a = a(1:10:end, :);
            d = [];
            for i = 2:size(a, 1)
                d = [d; haversine([a(i, 2), a(i, 3)], [a(i-1, 2), a(i-1, 3)])];
            end
            obj.dist = sum(d)*1000;
            obj.duration = (D(end, 1) - D(1, 1) ) / 1000;
            obj.speriod = quantile(diff(unique(Drx(:,1))) / 1000, 0.5);
            obj.speed = obj.dist / obj.duration;
            obj.step = obj.dist / size(unique(D(:, 1)), 1);
                       
            fprintf('step: %.2f (m), dtime: %.3f (s) [q0.5], speed: %.2f (%.2f) [m/s] ([km/h])\n', obj.step, obj.speriod, obj.speed, obj.dist/obj.duration / 1000 * 3600);
                        
            %% Generate the attack time
            % ... using first nth of the trace to generate the random spoofing time
            obj.spoof_time = rand() * (obj.Drx(end, 1) - obj.Drx(1, 1)) / 2 + obj.Drx(1, 1);
            %obj.spoof_time = (obj.Drx(end, 1) - obj.Drx(1, 1))/2 + obj.Drx(1, 1);
            ta = (obj.spoof_time - obj.Drx(1, 1)) / 1000 / 60;
            fprintf('Spoof time: %.2f [min]\n', ta);
            
        end
        
        function obj = run(obj, target, expF, PLOT)   
            %% If target is empty there is no spoofing attack
            if isempty(obj.enXY)
                fprintf('Warning: Missing error estimations. Run estimatePathErr(obj, expF).\n');
            end
            
            fprintf('Starting node tracking... ');
            
            %% Extract boundaries
            xm = min(obj.Drx(:, 10));
            xM = max(obj.Drx(:, 10));
            ym = min(obj.Drx(:, 11));
            yM = max(obj.Drx(:, 11));
            xlim([xm, xM]);
            ylim([ym, yM]);
            
            %% Selecting local anchors in the cartesian plance
            idx = find(obj.aDB(:, 4) > xm & obj.aDB(:, 4) < xM);
            obj.localDB = obj.aDB(idx, [4, 3]);
            idx = find(obj.localDB(:, 2) > ym & obj.localDB(:, 2) < yM);
            obj.localDB = obj.localDB(idx, :);
            
            %% Selecting local WiFi networks in the cartesian plance                        
            idx = find(obj.wDB.lonlat(:, 1) > xm & obj.wDB.lonlat(:, 1) < xM);            
            obj.localwDB = obj.wDB.lonlat(idx, :);
            idx = find(obj.localwDB(:, 2) > ym & obj.localwDB(:, 2) < yM);            
            obj.localwDB = obj.localwDB(idx, :);
            
            %% Print the in-range anchors and the others
            if PLOT
                clf
                hold on
                %% Green circles
                plot(obj.localDB(:, 1), obj.localDB(:, 2), '.', 'MarkerSize', 3, 'MarkerEdgeColor', 'r');
                plot(obj.localwDB(:, 1), obj.localwDB(:, 2), '.', 'MarkerSize', 3, 'MarkerEdgeColor', 'b');     
            end
            
            %% If no target no need to wait for the spoofing to reach the target
            if isempty(target)
                obj.target_reached = true;
            end
            
            % Number of initial anchors
            var = 0;
            while var < 6 | var > 10
                R=16.826192; P=0.702868; var = nbinrnd(R, P);                    
            end
            % Sinthetic neighbors
            sneigh = [];
            
            i = 1; j = 1; k = 1; 
            while ~obj.dest_reached | ~obj.target_reached
                
                %% Extract new event from the trace
                if ~obj.dest_reached
                    idx = ismember(obj.Drx(:, 1), obj.Drx(i, 1));
                    % Jump to the row with a new time
                    i = i + length(find(idx));
                    if i >= size(obj.Drx, 1)
                        obj.dest_reached = true;
                    end
                    obj.curr_pos = [mean(obj.Drx(idx, 2)), mean(obj.Drx(idx, 3))];
                    T = unique(obj.Drx(idx, 1));
                end
                
                %% Starting the spoofing and generate the spoofed trajectory
                if mean(obj.Drx(idx, 1)) > obj.spoof_time & ~obj.SPOOF & ~obj.target_reached
                    fprintf('\nSpoofing initiated...\n');
                    obj.SPOOF = true;
                    obj.spoof_init_pos = obj.curr_pos;
                    %steps = size(obj.Drx, 1) - i;
                    obj = obj.generate_spoof_trajectory(obj.spoof_init_pos, target, PLOT);
                end
                 
                %% Plot the spoofed trajectory
                if obj.SPOOF & ~obj.target_reached                    
                    obj.spoof_pos = [obj.spoof_traj(j, 1), obj.spoof_traj(j, 2)];
                    if PLOT
                        plot(obj.spoof_pos(1), obj.spoof_pos(2), 'MarkerEdgeColor', 'r', 'MarkerSize', 5, 'Marker', '.', 'MarkerFaceColor', 'r');
                    end
                    j = j + 1;
                    if j > size(obj.spoof_traj, 1)
                        obj.target_reached = true;
                    end
                    
                    % Spoof detection
                    % Pick n in-range anchors at certain distaces
                    % Generate RSSI
                    % Generate estimated position
                    % Compute the distance between the estimated position and the spoofed one (obj.curr_pos)
                    % d > thr ?
                    
                   %% Selecting local anchors in the cartesian plane
                    idx = find(obj.aDB(:, 4) > xm & obj.aDB(:, 4) < xM);
                    obj.localDB = obj.aDB(idx, [4, 3]);
                    idx = find(obj.localDB(:, 2) > ym & obj.localDB(:, 2) < yM);
                    obj.localDB = obj.localDB(idx, :);
                    
                    % Sort local anchors as a function of the distance
                    obj.localDB = [obj.localDB, NaN(size(obj.localDB, 1), 1)];
                    for il = 1:size(obj.localDB, 1)
                        d = haversine(obj.spoof_pos, obj.localDB(il, 1:2));
                        obj.localDB(il, 3) = d;
                    end
                    [s, si] = sort(obj.localDB(:,3));
                    obj.localDB = obj.localDB(si, :);
                    
                    % Generete statistics
                    mu = -38.614804; sigma = 19.743740; RSSI = normrnd(mu, sigma);

                    % Select anchors as a function of the pre-computed
                    % statistics
                    
                    if var > 0 & size(sneigh, 1) < 12
                        for n = 1:var
                            A = 1.607108; B = 0.645374; D = gamrnd(A/2, B);
                            mu = -38.614804; sigma = 19.743740; RSSI = normrnd(mu, sigma);
                            [m, mi] = min(abs(obj.localDB(:, 3) - D));
                            sneigh = [sneigh; [obj.localDB(mi, 1:2), RSSI]];
                        end
                    elseif var < 0 & size(sneigh, 1) > 2          
                        r = rand(size(sneigh, 1), 1);
                        [s, si] = sort(r);
                        sneigh(si(1:abs(var)), :) = [];
                    end
                    
                    % Compute anchor variation for next cicle
                    mu = 4.573e-04; sigma = 2 * 2.025e-01; var = round(normrnd(mu, sigma));
                     
                    % Compute weights of the exponential distribution
                    if size(sneigh, 1) > 1
                        [x, xi] = sort(sneigh(:, 3), 'descend'); xn = -(x - x(1)); y = exppdf(xn, expF); w = y/sum(y); 
                        % Estimate position using the weighted mean
                        ep = sum([sneigh(:, 1) .* w, sneigh(:, 2) .* w]);
                    else
                        ep = sneigh;
                    end                   
                    
                    %% Plot the estimated positions
                    if PLOT & size(sneigh, 1) > 0
                        plot(ep(1), ep(2), 'xm', 'MarkerSize', 5); 
                    end
                    
                    % Estimate distance between spoofed position and
                    % estimated position by anchors
                    d = haversine(ep(1:2), obj.curr_pos);
                    obj.dist_spoof_cur = [obj.dist_spoof_cur; [T, d]];
                    
                    %% Log anomaly 
                    if d > obj.spoof_thr
                        obj.spoof_vector_s = [obj.spoof_vector_s; 1];
                        % Se e' la prima volta salvo la size di obj.spoof_vector_a
                        if obj.spoof_begin == -1
                            obj.spoof_begin = size(obj.spoof_vector_a, 1);
                        end
                    else
                        obj.spoof_vector_s = [obj.spoof_vector_s; 0];
                    end                            
                end
                
                %% Plot real coordinates and evaluate false positives (FP)
                if ~obj.dest_reached 
                    if ~isempty(obj.enXY)
                        d = haversine([obj.enXY(k, 2:3)], [obj.enXY(k, 4:5)]);
                        if d > obj.spoof_thr
                            obj.spoof_vector_a = [obj.spoof_vector_a; 1];
                        else
                            obj.spoof_vector_a = [obj.spoof_vector_a; 0];
                        end
                    else
                        fprintf('Warning: Missing error estimations. Run estimatePathErr(obj, expF).\n');
                    end 

                    %% Print RSSI associated to each anchor
                    %% std deviation and trending might be interesting to see...
                    Drxt = obj.Drx(idx, :);
                    [u, ui] = unique(Drxt(:, [6,7,10,11]), 'rows');  
                    
                    %% Plotting RSS for each anchor
%                     htxt = [];
%                     for ii = 1:size(u, 1)
%                         iii = ismember(Drxt(:, 6:7), u(ii, 1:2), 'rows');
%                         mrx = round(mean(Drxt(iii, 9)));
%                         agps = u(ii, 3:4);
%                         %disp([agps, mrx])
%                         htxt = [htxt; text(agps(1), agps(2), num2str(mrx, '%i'))];
%                     end
                    if PLOT
                        %plot(obj.Drx(idx, 10), obj.Drx(idx, 11), 'o', 'MarkerEdgeColor', 'r', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
                        %plot(obj.Drx(idx, 10), obj.Drx(idx, 11), 'o', 'MarkerEdgeColor', 'b', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
                        pause(0.01); %% set breakpoint i == 33704 (for real attack figure)
                    end
                    % delete(htxt);
                    if ~isempty(obj.enXY)
                        if PLOT
                            %plot(obj.enXY(k, 2), obj.enXY(k, 3), 'o', 'MarkerEdgeColor', 'm', 'MarkerSize', 10, 'MarkerFaceColor', 'm');   
                            [~, idx] = min(abs(T - obj.enXY(:,1) ));
                            plot(obj.enXY(idx, 2), obj.enXY(idx, 3), 'xk', 'MarkerSize', 10);              
                            %plot(obj.enXY(k, 2), obj.enXY(k, 3), 'xk', 'MarkerSize', 10);              
                        end
                        %k = k + 1;
                    end
                    
                    if ~isempty(obj.ewXY)
                        if PLOT     
                            [~, idx] = min(abs(T - obj.ewXY(:,1) ));
                            plot(obj.ewXY(idx, 2), obj.ewXY(idx, 3), 'ok', 'MarkerSize', 10);              
                        end
                    end
                    
                    if PLOT
                        plot(obj.curr_pos(1), obj.curr_pos(2), 'MarkerEdgeColor', 'k', 'MarkerSize', 10, 'Marker', '.', 'MarkerFaceColor', 'k');
                    end
                end   
            end
            if PLOT
                hold off;
            end
            fprintf('Done.\n');
        end
        
        function obj = generate_spoof_trajectory(obj, curr_pos, target, PLOT)
            
            steps = haversine(curr_pos, target) * 1000 / obj.step;
            
            %% Generate the destination for the spoofing attack
            obj.spoof_dest = target;
            
            alpha = (target(2) - curr_pos(2)) / (target(1) - curr_pos(1));
            x = linspace(curr_pos(1), target(1), steps);
            y = alpha * (x - curr_pos(1)) + curr_pos(2);  
            
            obj.spoof_traj = [x', y'];
            
            if PLOT
                plot(obj.spoof_dest(1), obj.spoof_dest(2), 'xr', 'MarkerSize', 20, 'LineWidth', 5);            
            end
        end
        
        function obj = estimatePathErrWifi(obj, expF)            
            %% Retrieve BSSIDs using time from the wifi trace
            %% Exploit the DB to map BSSIDs to thier position
            fprintf('Estimating positions exploiting WiFi...');
            obj.ewXY = [];
            i = 1;
            while i < size(obj.Drx, 1)
                idxTime = ismember(obj.Drx(:, 1), obj.Drx(i, 1));                                  

                idx = find(obj.Drx(i, 1) == obj.Dw.info(:, 3));
                info = obj.Dw.info(idx, [1, 2, 4]); 
                
                bssid = obj.Dw.bssid(idx, :);
                bssid = unique(bssid, 'rows');
                
                %% Update info and bssid for the available bssid in wDB
                idx = ismember(bssid, obj.wDB.bssid, 'rows');
                info = info(idx, :);
                bssid = bssid(idx, :);
                
                
                idx = ismember(obj.wDB.bssid, bssid, 'rows');
                wpos = obj.wDB.lonlat(idx, :);
                                
                numwifi = size(rmmissing(wpos), 1);                
                
                if numwifi > 0        
                    if numwifi == 1
                        obj.ewXY = [obj.ewXY; [obj.Drx(i, 1), wpos, obj.Drx(i, [2, 3]), d, 1]];
                    else
                        rssi = info(:, 3);                                       
                        [x, xi] = sort(rssi, 'descend'); xn = -(x - x(1)); y = exppdf(xn, expF); wrssi = y/sum(y);                 

                        wposw = sum((wpos .* wrssi));                        

                        d = haversine(wposw, obj.Drx(i, [2, 3]));
                        obj.ewXY = [obj.ewXY; [obj.Drx(i, 1), wposw, obj.Drx(i, [2, 3]), d, numwifi]];
                    end
                end
                
                % Jump to the row with a new time
                i = i + sum(idxTime);                  
            end
            fprintf('Done.\n');
        end
        
        
        function obj = estimatePathErr(obj, expF, qthr)
        %% Return enXY: [time, position, estiamted position, distance]
            fprintf('Estimating path error for %i... ', expF);
            obj.enXY = [];
            i = 1;
            while i < size(obj.Drx, 1)
                idx = ismember(obj.Drx(:, 1), obj.Drx(i, 1));
                
                % Jump to the row with a new time
                i = i + length(find(idx));
                
                %% Extract RSSI values associated with each anchor
                Drxt = obj.Drx(idx, :);
                [u, ui] = unique(Drxt(:, [6, 7, 10, 11]), 'rows');    
                
                %% Compute the centroid from the anchor coordinates
                % [agps, rssi]
                Drxtf = Drxt(ui, [10, 11, 9]);    
                % Removing outliars
                %% Commented for journal paper
                %io = isoutlier(Drxtf(:, 1:2)); io = io(:,1) | io(:, 2); Drxtf = Drxtf(~io, :);                                 
                                
                [x, xi] = sort(Drxtf(:, 3), 'descend'); xn = -(x - x(1)); y = exppdf(xn, expF); w = y/sum(y); 
                
                % Real position as mean of the logged values
                rpos = [mean(Drxt(:, 2)), mean(Drxt(:, 3))];
                % Estimated position as the weighted mean
                epos = [sum(Drxtf(xi, 1) .* w), sum(Drxtf(xi, 2) .* w)];
                % Distance (Km) between estimation and real
                d = haversine(rpos, epos);
                
                nAnchors = size(Drxtf, 1);                
                obj.enXY = [obj.enXY; [unique(obj.Drx(idx, 1)), epos, rpos, d, nAnchors]];
            end
            fprintf('Done.\n');
            obj.spoof_thr = quantile(obj.enXY(:, 6), qthr);
            fprintf('Estimating detection threshold... %.2f (Km). Done.\n', obj.spoof_thr);
                        
            % Set up the spoofing window
            obj.spoof_window = floor(floor(obj.spoof_thr * 1000 / obj.speed) / obj.speriod);
            fprintf('Estimating the spoofing window... %.2f. Done.\n', obj.spoof_window);
        end
     
        function obj = estimateA(obj)
            fprintf('Estimating anchors statistics... \n');
            na = [];
            da = [];
            i = 1;
            while i < size(obj.Drx, 1)
                idx = ismember(obj.Drx(:, 1), obj.Drx(i, 1));
                
                % Jump to the row with a new time
                i = i + length(find(idx));
                
                % Count the number of in-range anchors 
                [u, ui] = unique(obj.Drx(idx, [6, 7, 8]), 'rows');          
                na = [na; [i, length(ui)]];
                
                % States associated with node-anchors distance
                nc = obj.Drx(idx, [2, 3]);   % Node coordinate
                ac = obj.Drx(idx, [10, 11]); % Anchor coordinate
                rssi = obj.Drx(idx, 9);
                d = [];
                for j = [1:size(nc, 1)]
                    d = [d; haversine(nc(j, :), ac(j, :))];
                end
                %da = [da; [min(d), quantile(d, 0.05), quantile(d, 0.5), quantile(d, 0.95), max(d)]];
                da = [da; [d, rssi]];
            end
            obj.sa.na = na;
            obj.sa.da = da;
            fprintf('Done.\n');
        end
        
        function obj = estimateWiFi(obj)
        %% Compute sttistics about 
        %% - Number of BSSID in the neighborhood
        %% - Relation distance to the BSSID - RSSI
            fprintf('Estimating BSSID statistics... \n');
            nbssid = [];
            distrssi = [];
            i = 1;
            while i < size(obj.Drx, 1)                
                idxTime = ismember(obj.Drx(:, 1), obj.Drx(i, 1));                                  
                idx = find(obj.Drx(i, 1) == obj.Dw.info(:, 3));
                                
                % Jump to the row with a new time
                i = i + length(find(idxTime));
                
                % Count the number of in-range BSSID
                nbssid = [nbssid; [i, size(obj.Dw.bssid(idx, :), 1)]];

                % Extract info and bssid for inrange station
                info = obj.Dw.info(idx, [1, 2, 4]); bssid = obj.Dw.bssid(idx, :);                                
                wpos = obj.wDB.lonlat(ismember(obj.wDB.bssid, bssid, 'rows'), :);
                
                for j = 1:size(wpos, 1)
                    d = haversine(wpos(j, :), info(1, [1,2]));
                    distrssi = [distrssi; [d, info(j, 3)]];
                end
            end
            obj.sawifi.nbssid = nbssid;
            obj.sawifi.distrssi = distrssi;
            fprintf('Done.\n');
        end
        
        
        %% To be executed after run
        function obj = fp_analysis(obj)
            b = find(diff([0; obj.spoof_vector_a; 0]) == -1) - find(diff([0; obj.spoof_vector_a; 0]) == 1);
            x = [min(b):1:max(b)]; 
            h = hist(b, x); 
            % burst length, frequencies
            obj.burstfreq_a = [find(h)', h(find(h))'];
            %disp(obj.burstfreq_a)
            
            b = find(diff([0; obj.spoof_vector_s; 0]) == -1) - find(diff([0; obj.spoof_vector_s; 0]) == 1);
            x = [min(b):1:max(b)]; 
            h = hist(b, x); 
            obj.burstfreq_s = [find(h)', h(find(h))'];
            %disp(obj.burstfreq_s)
            %disp(obj.spoof_window)            
        end
            
        function obj = pool_variation(obj)
            obj.anchor_var = [];
            oldp = [NaN, NaN, NaN];
            times = unique(obj.Drx(:, 1));
            for t = times'
                newp = unique(obj.Drx(find(obj.Drx(:, 1) == t), [6, 7, 8]), 'rows');
                np = newp; op= oldp; join = size(setdiff(np, op, 'rows'), 1);
                np = newp; op= oldp; leave = size(setdiff(op, np, 'rows'), 1);
                var = join - leave;
                oldp = newp;
                obj.anchor_var = [obj.anchor_var; [t, var]];
            end
        end
        
        function obj = wifi_gsm_weigth(obj)
            errWiFiGSM = [];
            fprintf('Estimating weights...');
            f = waitbar(0, 'Estimating weights...');
            for i = 1:size(obj.Drx, 1)
                t = obj.Drx(i, 1);
                
                [mgsm, idx] = min(abs(t - obj.enXY(:, 1)));
                posgsm = obj.enXY(idx, [2, 3]);
                
                [mwifi, idx] = min(abs(t - obj.ewXY(:, 1)));
                poswifi = obj.ewXY(idx, [2, 3]);
                                                
                if mwifi > 3e3
                    poswifi = [NaN NaN];
                end
                                
                if sum(isnan(poswifi)) > 0
                    posest = posgsm;                    
                    errWiFiGSM = [errWiFiGSM; [1, t, posgps, posest, haversine(posgps, posest)]];
                else                    
                    for w = [0:0.1:1]
                        posest = w * posgsm + (1-w) * poswifi;                
                        [~, idx] = min(abs(t - obj.Drx(:, 1)));
                        posgps = obj.Drx(idx, [2, 3]);                
                        error = haversine(posgps, posest);
                        errWiFiGSM = [errWiFiGSM; [w, t, posgps, posest, error]];
                    end                    
                end      
                waitbar(i / size(obj.Drx, 1), f, 'Estimating weights...');
            end            
            obj.errWiFiGSM = errWiFiGSM;
            fprintf('Done.\n'); close(f)            
        end
        
        function obj = generateModels(obj, w)
            if ~isfile('errWiFiGSMall.mat')
                fprintf('Missing errWiFiGSMall.mat...\n')
                return;
            end
                
            data = load('errWiFiGSMall.mat');
            d = [];
            for i = 1:size(data.results, 2)
                d = [d; data.results{i}];
            end
            d(:, 1) = round(d(:, 1) * 10) / 10;

            distWifiGSM = {}; k = 1;
            figure(); 
            hold on
            x = linspace(0, 3, 200); 
            %xl = logspace(log10(7e-4), log10(3), 200); 
            pt = ["+", "o", "*"];
            pc = ["r", "g", "b"];
            for i = w
                di =  d(d(:, 1) == i, 7); 

                h = hist(di, x); 
                h = h/sum(h);                               
                hp = plot(x, h, strcat(pc(k), pt(k)));    
                set(get(get(hp, 'Annotation'), 'LegendInformation'), 'IconDisplayStyle','off');
                [D PD] = allfitdist(di, 'PDF');

                dname = D(1).DistName;
                prm = D(1).Params;  

                switch length(prm)
                    case 1
                        pd = makedist(dname, prm(1));
                    case 2
                        pd = makedist(dname, prm(1), prm(2));
                    case 3
                        pd = makedist(dname, prm(1), prm(2), prm(3));
                end         
                distWifiGSM{k} = pd;
                y = pdf(pd, x); plot(x, y/sum(y), strcat(pc(k), '-'));
                k = k + 1;
            end                                            
            hold off
            obj.distWifiGSM = distWifiGSM;                                    
        end
        
        function obj = spoofdetection(obj, scenario, tech, iterations, thresholdD, thresholdT, speed)
            switch scenario
                case 'benign'
                    obj.false_positive = [];
                    bgall = {};
                    for i = tech
                        r = random(obj.distWifiGSM{i}, [iterations, 1]);
                        r = r(r >= 0);                        
                        bg = r > thresholdD; 
                        bgall{i} = [0; bg; 0];                        
                    end                    
                    
                    burst = {};
                    for i = tech
                        dbg = diff(bgall{i}); 
                        burst{i} = find(dbg == -1) - find(dbg == 1); 
                        if ~isnan(thresholdT)
                            obj.false_positive = [obj.false_positive, sum(burst{i} > thresholdT) / length(burst{i})];
                        end
                    end
                       
                    obj.burst = burst;        
                    
                case 'spoofing'
                    sample_time = 0.165;
                    %% Pick a random destination in a circle of radius 10
                    theta = rand() * 2 * pi; 
                    dest = 5 * [cos(theta), sin(theta)];
                    
                    delta = speed * sample_time;
                    deltax = delta * cos(theta);
                    deltay = delta * sin(theta);
                    
                    % Generate 1e6 positions...
                    r = random(obj.distWifiGSM{tech}, [1e6, 1]);                    
                    r = r(r >= 0);    
                                        
                    curpos = [0, 0];                    
                    i = 1; dist = [];
                    %hold on
                    while sqrt((curpos(1) - dest(1))^2 + (curpos(2) - dest(2))^2) > 0.01
                        thetaep = rand() * 2 * pi;                     
                        estpos = r(i) * [cos(thetaep), sin(thetaep)];
                        
                        curpos = curpos + [deltax, deltay];                        
                        % Distance between estimate and spoofed position
                        d = sqrt((curpos(1) - estpos(1))^2 + (curpos(2) - estpos(2))^2);
                        dist = [dist; d];
                        i = i + 1;
                        %plot(curpos(1), curpos(2), 'ok', estpos(1), estpos(2), 'xk');
                    end
                    %hold off
                    obj.distES = dist;
            end
        end
        
        
        function obj = wifi_gsm_malicious(obj, tech, n, w)            
            fprintf('Estimating...');     
            
            %thetaep = rand() * 2 * pi;                     
            %targetpos = 5 * [cos(thetaep), sin(thetaep)];
            
            sample_time = 0.165;
            speed = 0.01;
            %% Pick a random destination in a circle of radius 10
            theta = rand() * 2 * pi; 
            dest = 5 * [cos(theta), sin(theta)];

            delta = speed * sample_time;
            deltax = delta * cos(theta);
            deltay = delta * sin(theta);
            
            spoofpos = [0, 0];   
            
            i = 1;
            disterr = [];
            ts = unique(obj.Drx(:,1))';
            for t = ts                                                
                [mgsm, idx] = min(abs(t - obj.enXY(:, 1)));                
                ngsm = obj.enXY(idx, 7);                
                rposgsm = obj.enXY(idx, [2, 3]) - obj.enXY(idx, [4, 5]);
                
                
                [mwifi, idx] = min(abs(t - obj.ewXY(:, 1)));
                nwifi = obj.ewXY(idx, 7);   
                rposwifi = obj.ewXY(idx, [2, 3]) - obj.ewXY(idx, [4, 5]);

                srposgsm = rposgsm;
                srposwifi = rposwifi;
                
                %% tech = 1: WiFi
                %% tech = 2: Combined
                %% tech = 3: GSM  
                %% w = 0 -> wifi
                switch tech
                    case 1
                        %srposwifi = (rposwifi * nwifi + spoofpos * n) / (nwifi + n);
                        if n >= nwifi
                            srposwifi = spoofpos;
                        end
                    case 2
                        %srposgsm = (rposgsm * ngsm + spoofpos * n/2) / (ngsm + n/2); 
                        %srposwifi = (rposwifi * nwifi + spoofpos * n/2) / (nwifi + n/2);
                        if n/2 >= nwifi
                            srposwifi = spoofpos;
                        end
                        if n/2 >= ngsm                       
                            srposgsm = spoofpos;
                        end
                        
                    case 3
                        %srposgsm = (rposgsm * ngsm + spoofpos * n) / (ngsm + n); 
                        if n >= ngsm                       
                            srposgsm = spoofpos;
                        end
                end                
                estpos = w * srposgsm + (1-w) * srposwifi;                           
                
                spoofpos = spoofpos + [deltax, deltay]; 
                
                %% Assuming current position as 0, 0
                disterr = [disterr; [sqrt((0 - estpos(1))^2 + (0 - estpos(2))^2)]];
                
                i = i+1;
                if i == 954
                    disp('');
                end
            end       
            obj.disterrM = disterr; 
            fprintf('Done.\n');               
        end
        
        
    end
end