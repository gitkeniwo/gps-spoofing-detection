%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Journal extension
% 
% T = 1; sd = spoofdect2(); sd = sd.loadtrace(DS(T)); expF=25; sd = sd.estimatePathErr(expF, 0.9); sd = sd.run([], expF, true);  sd = sd.fp_analysis();
% 
% T = 1; sd = spoofdect(); sd = sd.loadtrace(DS(T)); 
% sd = sd.estimatePathErrWifi(); expF=25; sd = sd.estimatePathErr(expF, 0.9); 
% sd = sd.run([25.3, 51.48], expF, true);


clear all
clf
format long g

% DS =["hbkuhome.csv";
%      "duhail-8.csv";
%      "home-duhail.csv";
%      "home-ceid.csv";
%      "hbku-ceid.csv";     
%      "trace1.csv"; 
%      "trace2.csv"; 
%      "trace3.csv"; 
%      "trace4.csv"; 
%      "trace5.csv"; 
%      "trace6.csv"; 
%      "trace7.csv"; 
%      "trace8.csv"];
DS=["trace4.csv"]
 
nfig = 1;
haversine([25.3695,25.3628],[51.5528,51.5528])
% %% Tracing all the paths
fig = figure(nfig); nfig = nfig + 1;
hold on;
path = [];
pathall = []; disttot = 0; durtot = 0; speedtot = 0;
stime = [];
for T = [1:size(DS, 1)]
    sd = spoofdect2(); sd = sd.loadtrace(DS(T));
    path = sd.Drx(:, 2:3);
    %hs = plot(path(:, 1), path(:, 2), '.', 'MarkerSize', 20);
    hp = plot(path(:, 1), path(:, 2), '-k');
    if T < size(DS, 1)
        set(get(get(hp,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
    end
    pathall = [pathall; path];
    fprintf('\n%s \t %.2f \t %.2f \t %.2f \n', DS(T), sd.dist, sd.duration, sd.speed)
    disttot = disttot + sd.dist;
    durtot = durtot + sd.duration;
    speedtot = speedtot + sd.speed;
    stime = [stime; sd.speriod];
end
fprintf('%.2f \t %.2f \t %.2f \n', disttot, durtot, speedtot/13 );

xm = min(pathall(:,1));
xM = max(pathall(:,1));
ym = min(pathall(:,2));
yM = max(pathall(:,2));
fprintf('xm: %.4f, xM: %.4f, ym: %.4f, yM: %.4f\n', xm,xM,ym,yM);
fprintf('length test: %.4f\n',haversine([xm, ym], [xm, yM]))
length = haversine([xm, ym], [xm, yM]);
width = haversine([xm, yM], [xM, yM]);
fprintf('Length: %.2f, Width: %.2f, Area: %.2f\n', length, width, length*width);

% Load the cell database
str = sprintf('./Data/CellsDatabase.csv');
anchorDB = readtable(str);
aDB = [anchorDB(:, 4).('area'),...
       anchorDB(:, 5).('cell'),...7
       anchorDB(:, 7).('lon'),...
       anchorDB(:, 8).('lat'),...
       anchorDB(:, 3).('net')];
idx = find(aDB(:, 4) > xm & aDB(:, 4) < xM);
localDB = aDB(idx, [4,3]);
idx = find(localDB(:, 2) > ym & localDB(:, 2) < yM);
localDB = localDB(idx, :);

% Load the WiFi database
str = sprintf('./Data/WiFiDatabase.csv');
wifiDB = readtable(str, 'Delimiter', ',');
wifiDB = rmmissing(wifiDB);
localDBw = [wifiDB(:, 2).('Lat'), wifiDB(:, 3).('Long')];
idx = find(localDBw(:, 2) > ym & localDBw(:, 2) < yM);
localDBw = localDBw(idx, :);

plot(localDB(:, 1), localDB(:, 2), '.r', 'MarkerSize', 5);
plot(localDBw(:, 1), localDBw(:, 2), '.b', 'MarkerSize', 10);

% for T = [1:size(DS, 1)]
%     sd = spoofdect2(); sd = sd.loadtrace(DS(T));
%     path = sd.Drx(:, 2:3);
%     plot(path(:, 1), path(:, 2), '-k');


set(gca, "Fontsize", 18)
%[h, icons] = legend('T1', 'T2', 'T3', 'T4', 'T5', "T6", "T7", "T8", "T9", "T10", "T11" , "T12", "T13", 'Location', 'SouthEast');
[h, icons] = legend('Path', 'GSM', 'WiFi', 'Location', 'SouthEast');
% Changing marker size in the legend
icons = findobj(icons, 'Type', 'line');
icons = findobj(icons, 'Marker', 'none', '-xor');
set(icons,'MarkerSize',40);
xlim([xm, xM]);
ylim([ym, yM]);
%set(gca,'xtick',[]); set(gca, 'ytick', []); 
xlabel('Latitude'); ylabel('Longitude');
orient(fig, 'landscape');
print(fig, '-bestfit', 'paths_j','-dpdf');
hold off; 
 
 
%% Number of in-range anchors/BSSID
na = []; rxd = [];
nb = []; rxdw = [];
for T = [1:size(DS, 1)]
    sd = spoofdect2(); sd = sd.loadtrace(DS(T));
    sd = sd.estimateA();
    % sd = sd.estimateWiFi();    
    
    %% Anchors
    na = [na; sd.sa.na(:, 2)];
    rxd = [rxd; sd.sa.da];
    % %% Bssid
    % nb = [nb; sd.sawifi.nbssid(:, 2)];
    % rxdw = [rxdw; sd.sawifi.distrssi];
end

% Removing spurious powers... why are they there?
rxd = rxd(rxd(:,2) < 20, :);
xb = unique(na); hb = hist(na, xb); hb = hb/sum(hb); 
% xb = unique(nb); hb = hist(nb, xb); hb = hb/sum(hb); 
%x = unique(na); 
h = hist(na, xb); h = h/sum(h); 

%% plot wifi(can be removed)
fig = figure(nfig); nfig = nfig + 1;
hold on;
plot(xb, 1 - cumsum(h), 'o-r')
ylabel("1-CDF");
xlabel("Number of in-range anchors");
%xticks([0:14]);
ylim([0, 1]);
set(gca, "Fontsize", 18)
legend('GSM')
grid on
hold off

axes('Position',[.5 .5 .3 .3])
box on
hold on
% b = bar(x, h, 'k', 'FaceColor', 'flat');
b = bar(xb, [hb']);
%c = b.FaceColor;
%b.FaceColor = [0.5 0.5 0.5];
% plot(x, eh / sum(eh), '-b', 'LineWidth', 2)
xlabel("Number of in-range anchors");
ylabel("Frequency");
set(gca, "Fontsize", 18)
yticks([0:.1:.5]);
ylim([0, 0.5]);
xlim([0, 15]);
grid on
orient(fig, 'landscape');
print(fig, '-bestfit', 'inrange_anchors_j','-dpdf');
hold off;


%% Analysis of the anchor-node distance %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = logspace(log10(min(rxd(:, 1))), log10(max(rxd(:, 1))), 200);
h = hist(rxd(:,1), x); h = h/sum(h); 

distw = rxd(:, 1); 
% distw = distw(distw < 0.315);
xb = logspace(log10(min(distw)), log10(max(distw)), 200);
hb = hist(distw, xb); hb = hb/sum(hb); 

fig = figure(nfig); nfig = nfig + 1;
hold on;
plot(x, h, 'or', xb, hb, 'ob')

ylabel("Frequency");
xlabel("Distance (Km)");
%xticks([0.5:0.5:14]);
set(gca, "Fontsize", 18)
%xlim([0, 7]);
legend('GSM', 'WiFi')
set(gca, 'XScale', 'log', 'YScale', 'log');
grid on
orient(fig, 'landscape');
print(fig, '-bestfit', 'node_anchors_distance_j','-dpdf');
hold off;

%% %% Analysis of the received signal strength %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

d = rxd(:,2); d = d(find(d < 13)); 
x = unique(d); 
h = hist(d, x); h = h/sum(h);

% rssiw = rxdw(:,2);
% xw = unique(rssiw); 
% hw = hist(rssiw, xw); hw = hw/sum(hw);

fig = figure(nfig); nfig = nfig + 1;
hold on
% plot(x, h, 'r-o', xw, hw, 'b-o');
plot(x, h, 'r-o')
xlim([-110, 30]);
xticks([-150:20:50]);
xlabel("Received Signal Strength [dBm]");
ylabel("Frequency");
hold off
legend('GSM');
grid on
set(gca, "Fontsize", 18)
orient(fig, 'landscape');
print(fig, '-bestfit', 'rx_analysis_j', '-dpdf');

%% %%%% Weight analysis %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% data = load('errWiFiGSMall.mat');
data = load('error_mu.mat');
size(data.res)
d = [];
% for i = 1:size(data.results, 2)
for i = 1:size(data.res, 2)
    % d = [d; data.results{i}];
    d = [d; data.res(1).egsm];
end
% disp(d{1}(1:10,:));
d{1}(:, 1) = round(d{1}(:, 1) * 10) / 10;
thr = 0.9;

fig = figure(nfig); nfig = nfig + 1;
hold on
xthr = [];
for i = unique(d{1}(:,1))' 
    di =  d{1}(:, 3); 
    x = linspace(0, 3, 100); 
    h = hist(di, x); 
    h = h/sum(h); ch = cumsum(h);
    [~, idx] = max(ch > thr); xthr = [xthr; x(idx)];
    plot(x, ch, '-');         
end
plot([0.3, 0.7], [thr, thr], '-r', 'Linewidth', 3)
hold off
legend('w=0.0', 'w=0.1', 'w=0.2', 'w=0.3', 'w=0.4', 'w=0.5', 'w=0.6', 'w=0.7', 'w=0.8', 'w=0.9', 'w=1.0');
ylim([0, 1]);
xlim([0, 1]);
yticks([0:0.1:1])
xlabel("Position Estimation Error (Km)");
ylabel("CDF");
grid on
set(gca, "Fontsize", 28)

% axes('Position',[.4 .25 .35 .35])
% box on
%     bar([0:0.1:1], xthr);
%     xlabel("w");
%     ylabel({'Position estimation', 'error [Km]'});
%     grid on
%     set(gca, "Fontsize", 18)
% box off
orient(fig, 'landscape');
print(fig, '-bestfit', 'position_estimation_error', '-dpdf');

%% %% Weight analysis %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data = load('error_mu.mat');
d = [];
for i = 1:size(data.res, 2)
    d = [d; data.res(1).egsm];
end

d{1}(:, 1) = round(d{1}(:, 1) * 10) / 10;
w = [0, 0.5, 1];

distWifiGSM = {}; k = 1;
fig = figure(nfig); nfig = nfig + 1;
hold on
x = linspace(0, 3, 200); 
%xl = logspace(log10(7e-4), log10(3), 200); 
pt = ["+", "o", "*"];
pc = ["b", "g", "r"];
for i = w
    di =  d{1}(d{1}(:, 1) == i, 3); 
    
    h = hist(di, x); 
    h = h/sum(h);                               
    hp = plot(x, h, strcat(pc(k), pt(k)));    
    set(get(get(hp, 'Annotation'), 'LegendInformation'), 'IconDisplayStyle','off');
    [D PD] = allfitdist(di, 'PDF');
    
    dname = D(1).DistName;
    prm = D(1).Params;  
    
    switch size(prm, 2)
        case 1
            pd = makedist(dname, prm(1));
        case 2
            pd = makedist(dname, prm(1), prm(2));
        case 3
            pd = makedist(dname, prm(1), prm(2), prm(3));
    end         
    pd
    y = pdf(pd, x); plot(x, y/sum(y), strcat(pc(k), '-'));
    k = k + 1;
end
hold off
xlim([0, 1]);
legend('w = 0.0', 'w = 0.5', 'w = 1.0')
xlabel('Position estimation error (Km)');
ylabel('Probability distribution function');

set(gca, "Fontsize", 28)
orient(fig, 'landscape');
print(fig, '-bestfit', 'models', '-dpdf');


%% %%%% Wifi + GSM Anomaly analysis (worst case) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
           
sd = spoofdect2();
w = [0, 0.5, 1];
sd = sd.generateModels(w);

res = {}; k = 1;
for tech = 1:3
    for i = logspace(log10(0.01), log10(2), 100)%[0.01:0.01:2]
        %% Generate iterations event and decide if FP when they cross the threshold
        sd = sd.spoofdetection('benign', 1e6, i, NaN);
        x = unique(sd.burst{tech}); 
        h = hist(sd.burst{tech}, x)';
        res{k} = {tech, i, x, h}; k = k + 1;
    end
end
save('burst_distribution.mat', 'res', '-v7.3');

load burst_distribution.mat;
fig = figure(nfig); nfig = nfig + 1;
pt = ["+", "o", "*"];
hold on
for i = 1:size(res, 2)
    x = res{i}{3};
    h = res{i}{4};
    hp = plot(x, h, strcat('k', pt(res{i}{1})));
    set(get(get(hp, 'Annotation'), 'LegendInformation'), 'IconDisplayStyle','off');
end
plot([0 0], [0 0], '+k', [0 0], [0 0], 'ok', [0 0], [0 0], '*k')
hold off
set(gca, 'XScale', 'log', 'YScale', 'log');
legend('WiFi', 'WiFi+GSM', "GSM")
xlabel('Burst length');
ylabel('Number of occurrences');


r1 = [];
for i = 1:size(res, 2) 
    if res{i}{1} == 1
        v = max(res{i}{3});
        %v = quantile(res{i}{3}, 0.99);
        r1 = [r1; [res{i}{2}, v]];
    end
end
r2 = [];
for i = 1:size(res, 2) 
    if res{i}{1} == 2
        v = max(res{i}{3});
        r2 = [r2; [res{i}{2}, v]];
    end
end
r3 = [];
for i = 1:size(res, 2) 
    if res{i}{1} == 3
        v = max(res{i}{3});
        r3 = [r3; [res{i}{2}, v]];
    end
end

st05 = 0.165;

loglog(r1(:,1), r1(:,2)*st05, 'ob', r2(:,1), r2(:,2)*st05, 'xg', r3(:,1), r3(:,2)*st05, '*r')
legend('WiFi', 'WiFi+GSM', "GSM")
xlabel('Decision threshold (Km)');
ylabel('Anomaly duration (worst case) [s]');
grid on
set(gca, "Fontsize", 28)
orient(fig, 'landscape');
print(fig, '-bestfit', 'time_threshold', '-dpdf');

%%
%%%% Wifi + GSM False positive analysis %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sd = spoofdect2();
w = [0, 0.5, 1];
sd = sd.generateModels(w);

thrD = [0.01:0.001:0.09];
thrT = [1:5:100];
tech = [1, 2, 3];

res = [];
for i = thrD
    for j = thrT
        sd = sd.spoofdetection('benign', tech, 1e5, i, j);              
        res = [res; [i, j, sd.false_positive]];
    end
end

fig = figure(nfig); nfig = nfig + 1;
[xq,yq] = meshgrid(linspace(min(thrD), max(thrD), 100), linspace(min(thrT), max(thrT), 100));
vq = griddata(res(:,1), res(:,2), res(:,3), xq, yq);

h = meshc(xq, yq, vq);
h(2).ContourZLevel = -0.5;
h(2).LineWidth = 5;
hold on
%plot3(res(:,1), res(:,2), res(:,3), 'ok');
hold off
xlabel('Decision threshold (Km)');
hy = ylabel('Anomaly duration (s)');
posy = get(hy, 'Position');
hy.Position = [0.095 100 -0.75];
zlabel('False positive');
set(get(gca,'YLabel'),'Rotation',-35)
set(get(gca,'xLabel'),'Rotation',5)
%[caz, cel] = view();
view(157.65, 18.73);

h(2).LevelList = [0.01];
y1 = h(2).ContourMatrix(1, 2:end)';
x1 = h(2).ContourMatrix(2, 2:end)';

set(gca, "Fontsize", 28)
orient(fig, 'landscape');
print(fig, '-bestfit', 'wifi3d', '-dpdf');

%%%

fig = figure(nfig); nfig = nfig + 1;
[xq,yq] = meshgrid(linspace(min(thrD), max(thrD), 100), linspace(min(thrT), max(thrT), 100));
vq = griddata(res(:,1), res(:,2), res(:,4), xq, yq);

h = meshc(xq, yq, vq);
h(2).ContourZLevel = -0.5;
h(2).LineWidth = 5;
hold on
%plot3(res(:,1), res(:,2), res(:,3), 'ok');
hold off
xlabel('Decision threshold (Km)');
hy = ylabel('Anomaly duration (s)');
posy = get(hy, 'Position');
hy.Position = [0.095 100 -0.75];
zlabel('False positive');
set(get(gca,'YLabel'),'Rotation',-35)
set(get(gca,'xLabel'),'Rotation',5)
%[caz, cel] = view();
view(157.65, 18.73);

h(2).LevelList = [0.01];
y2 = h(2).ContourMatrix(1, 2:end)';
x2 = h(2).ContourMatrix(2, 2:end)';

set(gca, "Fontsize", 28)
orient(fig, 'landscape');
print(fig, '-bestfit', 'wifigsm3d', '-dpdf');

%%%

fig = figure(nfig); nfig = nfig + 1;
[xq,yq] = meshgrid(linspace(min(thrD), max(thrD), 100), linspace(min(thrT), max(thrT), 100));
vq = griddata(res(:,1), res(:,2), res(:,5), xq, yq);

h = meshc(xq, yq, vq);
h(2).ContourZLevel = -0.5;
h(2).LineWidth = 5;
hold on
%plot3(res(:,1), res(:,2), res(:,3), 'ok');
hold off
xlabel('Decision threshold (Km)');
hy = ylabel('Anomaly duration (s)');
posy = get(hy, 'Position');
hy.Position = [0.095 100 -0.75];
zlabel('False positive');
set(get(gca,'YLabel'),'Rotation',-35)
set(get(gca,'xLabel'),'Rotation',5)
%[caz, cel] = view();
view(157.65, 18.73);

h(2).LevelList = [0.01];
y3 = h(2).ContourMatrix(1, 2:end)';
x3 = h(2).ContourMatrix(2, 2:end)';
set(gca, "Fontsize", 28)
orient(fig, 'landscape');
print(fig, '-bestfit', 'gsm3d', '-dpdf');

%%%
fig = figure(nfig); nfig = nfig + 1;
hold on 
plot(x1, y1, 'ok', x2, y2, 'xk', x3, y3, '*k', 'Markersize', 10)
fo1 = fit(x1, y1, 'power2'); p = plot(fo1, 'b-'); p.LineWidth = 3;
fo2 = fit(x2, y2, 'power2'); p = plot(fo2, 'g-'); p.LineWidth = 3;
fo3 = fit(x3, y3, 'power2'); p = plot(fo3, 'r-'); p.LineWidth = 3;
hold off
xlim([1, 100]); ylim([0.01, 0.09])
ylabel('Decision threshold (Km)');
xlabel('Anomaly duration (s)');
legend('WiFi', 'WiFi+GSM', "GSM")
text(10, 0.018, 'A', 'FontSize', 38)
text(30, 0.04, 'B', 'FontSize', 38)
text(60, 0.06, 'C', 'FontSize', 38)
set(gca, "Fontsize", 28)
orient(fig, 'landscape');
print(fig, '-bestfit', 'boundsFP', '-dpdf');



%%%% Wifi + GSM False positive analysis %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sd = spoofdect2();
w = [0, 0.5, 1];
sd = sd.generateModels(w);
speed = 0.01;

% Use tech=1, WiFI
ris = [];
for thrD = [0.01:0.01:0.09];
    detdelay1 = [];
    for i = 1:100
        sd = sd.spoofdetection('spoofing', 1, NaN, NaN, NaN, speed);
        d = sd.distES;
        e = d > thrD;
        e = [0; e; 0];
        de = diff(e); 
        b = find(de == -1) - find(de == 1); 
        f = fit(y1, x1, 'power2'); 
        thr = round(f(thrD));
        idx = min(find(b > thr));
        detdelay1 = [detdelay1; sum(b(1:idx-1)) + idx - 1 + thr];                
    end
    detdelay2 = [];
    for i = 1:100
        sd = sd.spoofdetection('spoofing', 2, NaN, NaN, NaN, speed);
        d = sd.distES;
        e = d > thrD;
        e = [0; e; 0];
        de = diff(e); 
        b = find(de == -1) - find(de == 1); 
        f = fit(y2, x2, 'power2'); 
        thr = round(f(thrD));
        idx = min(find(b > thr));
        detdelay2 = [detdelay2; sum(b(1:idx-1)) + idx - 1 + thr];                
    end
    detdelay3 = [];
    for i = 1:100
        sd = sd.spoofdetection('spoofing', 3, NaN, NaN, NaN, speed);
        d = sd.distES;
        e = d > thrD;
        e = [0; e; 0];
        de = diff(e); 
        b = find(de == -1) - find(de == 1); 
        f = fit(y3, x3, 'power2'); 
        thr = round(f(thrD));
        idx = min(find(b > thr));
        detdelay3 = [detdelay3; sum(b(1:idx-1)) + idx - 1 + thr];                
    end
        
    ris = [ris; [thrD, median(detdelay1), median(detdelay2), median(detdelay3)]];
end

fig = figure(nfig); nfig = nfig + 1;
semilogy(ris(:,1), ris(:,2)*0.165, 'o-b', ris(:,1), ris(:,3)*0.165, 'o-g', ris(:,1), ris(:,4)*0.165, 'o-r', 'LineWidth', 3)
xlabel('Decision threshold (Km)');
ylabel('Spoofing detection delay (s)');
legend('WiFi', 'WiFi+GSM', "GSM")
ylim([1, 1e3])
grid on
set(gca, "Fontsize", 28)
orient(fig, 'landscape');
print(fig, '-bestfit', 'detection_delay_j', '-dpdf');


%%%% Anchor corruption %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
format long g
nfig = 1;

% DS =["hbkuhome.csv";
%      "duhail-8.csv";
%      "home-duhail.csv";
%      "home-ceid.csv";
%      "hbku-ceid.csv";     
%      "trace1.csv"; 
%      "trace2.csv"; 
%      "trace3.csv"; 
%      "trace4.csv"; 
%      "trace5.csv"; 
%      "trace6.csv"; 
%      "trace7.csv"; 
%      "trace8.csv"];
DS=["trace4.csv"]
ris = [];
for T = [1:size(DS, 1)]
    expF = 25;
    sd = spoofdect2(); sd = sd.loadtrace(DS(T));
    sd = sd.estimatePathErr(expF, 0.9);
    sd = sd.estimatePathErrWifi();
    for nm = 1:40
        sd = sd.wifi_gsm_malicious(2, nm, 0.5);    
        ris = [ris; [T, nm, sum(sd.disterrM > 0.100) / length(sd.disterrM)]];
    end
end

r = [];
for i = unique(ris(:,2))'
    r = [r; [i, quantile(ris(ris(:, 2) == i, 3), [0.05, 0.5, 0.95])]];
end

fig = figure(nfig); nfig = nfig + 1;
idx = [1, 5:5:40];
hold on
errorbar(r(idx, 1), r(idx, 3), r(idx, 3) - r(idx, 2), r(idx, 4) - r(idx, 3), 'ok', 'Linewidth', 3);
plot(r(idx, 1), r(idx, 3), '--k', 'Linewidth', 3)
hold off
xlabel('Number of deployed malicious anchors');
ylabel('Succesful spoofing probability');
grid on;
set(gca,'ytick',[0:0.1:1]);
set(gca, "Fontsize", 28)
orient(fig, 'landscape');
print(fig, '-bestfit', 'malicious_anchors', '-dpdf');

%% %%%% Real spoofing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear all
format long g
nfig = 1;

DS =["Home_Cor_Spoof1.xlsx";
     "Home_HBKU_Spoof1.xlsx";
     "Home_PS_Spoof1.xlsx"];

err = {}; path = {};
hold on
for i = 1:size(DS, 1)
    sd = spoofdect2(); 
    sd = sd.loadtrace(DS(i));
    sd = sd.estimatePathErr(25, 0.9);
    sd = sd.estimatePathErrWifi(25);
    sd = sd.wifi_gsm_weigth();
    % plot(sd.Drx(:, 2), sd.Drx(:, 3), 'ok', sd.ewXY(:, 2), sd.ewXY(:, 3), 'xk', sd.enXY(:, 2), sd.enXY(:, 3), 'sk')
    err{i} = sd.errWiFiGSM(sd.errWiFiGSM(:,1) == 0.5, 7);
    path{i} = [sd.Drx(:, 2), sd.Drx(:, 3)];
end
hold off

apos = mean(sd.Drx(:, 4:5));

fig = figure(nfig); nfig = nfig + 1;
hold on
plot(path{1}(:, 1), path{1}(:, 2), 'ok', path{2}(:, 1), path{2}(:, 2), 'xk', path{3}(:, 1), path{3}(:, 2), 'sk', 'MarkerSize', 10, 'LineWidth', 1) 
plot(apos(:,1), apos(:,2), 'dr', 'MarkerSize', 20, 'LineWidth', 5)
hold off
legend('Spoofed Path 1', 'Spoofed Path 2', 'Spoofed Path 3', 'Actual position')
xlabel('Latitude');
ylabel('Longitude');

grid on;
set(gca, "Fontsize", 28)

axes('Position',[.55 .28 .3 .3])
box on
ds = 200; 
plot([1:ds:length(err{1})]*0.165, err{1}(1:ds:end), 'ok', [1:ds:length(err{2})]*0.165, err{2}(1:ds:end), 'xk', [1:ds:length(err{3})]*0.165, err{3}(1:ds:end), 'sk', 'MarkerSize', 10, 'LineWidth', 1)
yticks([0:1:5])
xlabel('Time [s]');
ylabel('Pos. Est. Error [Km]');
grid on;
box off

set(gca, "Fontsize", 28)
orient(fig, 'landscape');
print(fig, '-bestfit', 'real_spoof', '-dpdf');


% %% Weight analysis for the position estimation

%% matlab -nodisplay -nosplash -nodesktop -r "test; exit;"

clear all
format long g
nfig = 1;

% DS =["home-hbku.csv";
%      "duhail-8.csv";
%      "home-duhail.csv";
%      "home-ceid.csv";
%      "ceid-hbku.csv";     
%      "trace1.csv"; 
%      "trace2.csv"; 
%      "trace3.csv"; 
%      "trace4.csv"; 
%      "trace5.csv"; 
%      "trace6.csv"; 
%      "trace7.csv"; 
%      "trace8.csv"];
DS=["trace3.csv","trace4.csv" ];
disp(feature('numcores'))
numCores = feature('numcores');
% p = parpool(numCores);

egsm = {}; ewifi = {};
sd = spoofdect2();sd = sd.loadtrace(DS(1));
for T = [1:size(DS, 1)]
    sd = spoofdect2(); sd = sd.loadtrace(DS(T));
    eg = []; ew = [];
    for i = [1, 5:5:300]
        sd = sd.estimatePathErr(i, 0.9); % e(1) / 1000 / 60 are minutes
        % sd = sd.estimatePathErrWifi(i);
        eg = [eg; [i*ones(size(sd.enXY(:, 1), 1), 1), sd.enXY(:, 1), sd.enXY(:, 6)]];
        % ew = [ew; [i*ones(size(sd.ewXY(:, 1), 1), 1), sd.ewXY(:, 1), sd.ewXY(:, 6)]];
    end
    egsm{T} = eg;
    ewifi{T} = ew;
end
res.egsm = egsm;
res.ewifi = ewifi;

save('error_mu.mat', 'res');

load('error_mu.mat'); 

ris = [];
for i = [1, 5:5:100]
    gsm = []; wifi = [];
    for T = [1:size(DS, 1)]
        gsm = [gsm; res.egsm{T}(res.egsm{T}(:, 1) == i, 3)];
        % wifi = [wifi; res.ewifi{T}(res.ewifi{T}(:, 1) == i, 3)];
    end
    % ris = [ris; [i, quantile(gsm, 0.5), quantile(wifi, 0.5)]];
    ris = [ris; [i, quantile(gsm, 0.5)]];
end

fig = figure(nfig); nfig = nfig + 1;
hold on
% plot(ris(:,1), ris(:, 2), '-ro', ris(:,1), ris(:, 3), '-bo', 'LineWidth', 3)
plot(ris(:,1), ris(:, 2), '-ro', 'LineWidth', 3)


plot([20, 20], [0.05, 0.4], '-k', 'LineWidth', 5)
hold off
legend('GSM', 'WiFi')
xlabel('Exponential coefficient (\mu)');
ylabel('Error [Km]');
% ylim([0 0.5])
xticks([0:10:100])
grid on;
set(gca, "Fontsize", 28)
orient(fig, 'landscape');
print(fig, '-bestfit', 'error_exponent_j', '-dpdf');






