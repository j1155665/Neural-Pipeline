% To avoid including a large data file in the repository, the data
% loaded here have been pre-binned using 10ms windows, i.e., each entry in
% the spikes matrices indicates the number of spikes recorded in a 10ms
% window

%% input we need
% spikes  - a two element cell. spikes{1} is a p1 x T x N array containing
% the spiking activity in neuronal population 1. spikes{2} is a p2 x T x N
% array containing the spiking activity in neuronal population 2.
% 
% expCond - vector containing the trial label for each trial (N x 1)

% clear
% addpath util
% load mat_sample/sample_data.mat

%% converte your data to spikes and expCond
argIn.Binsize = 10; % 10ms
argIn.BinWidth = 1;     % 10ms
argIn.MaxDelay = 10;     % 100ms
argIn.TimeStep = 2;      % 40ms
argIn.WindowLength = 8;  %  80ms
argIn.UsePCA = 1;  
argIn.NumPCs = [20, 20]; 
argIn.CrossValidate = 1;
argIn.C_CV_NUM_FOLDS = 3;
Info.session_dates = {'20250602', '20250702', '20250523',  '20250501', '20250417', '20250710', '20250411', '20250306'};
Info.seesion_nums = [21, 22, 20, 19, 18, 23, 17, 15];

%
all_maps = {};
for i = 1:length(Info.seesion_nums)
    Info.session_date = Info.session_dates{i};
    session_num  = Info.seesion_nums(i);
%     data_spkrate = dataStruct(session_num).data.dots3DMPtuning.data_spkrate;
%     event = dataStruct(session_num).data.dots3DMPtuning.events;
    data_spkrate = dataStruct(session_num).data.dots3DMP.data_spkrate.saccOnset;
    event = dataStruct(session_num).data.dots3DMP.events;
    Info.goodtrial = find(event.goodtrial == 1 & event.PDW == 0);
    
    %
    unitInfo = create_area_map(Info, dataStruct(session_num).data);
    
    spikes{1} = create_spikes(Info, unitInfo.MST, data_spkrate);
    spikes{2} = create_spikes(Info, unitInfo.VPS, data_spkrate);
    
    %
    expCond = create_expCond(event,Info.goodtrial);
    
    %argIn.NumWorkers = Inf; % Requires Parallel Pz`rocessing Toolbox
    
    disp(argIn)
    argOut = ComputeCorrMap(spikes, expCond, argIn);
    all_maps{i} = argOut.CorrMap;
end

cat_maps = cat(4, all_maps{:});
avg_CorrMap = mean(cat_maps, 4);
%%
Info.tstart = -600; % -100, -600, -600
close all
CANONICAL_PAIR_IDX = 1;
mapDim = size(avg_CorrMap, 2);
delays = (-argIn.MaxDelay:argIn.MaxDelay)*argIn.Binsize; % Convert to ms
t = Info.tstart + (1:argIn.TimeStep:argIn.TimeStep*mapDim)*argIn.Binsize; % Convert to ms %change this to your t

figure(1);

imagesc( delays, t, avg_CorrMap(:,:,CANONICAL_PAIR_IDX)' )
yline(0, 'k--', 'LineWidth', 1);
colorbar;               
caxis([0.18 0.23]); 

ax = gca;
ax.YDir = 'Normal';


xlabel('Delay')
ylabel('Time')

%% Example computation of the interaction structure analysis
% Fig. 6

clear argIn

argIn.TimePeriods = [...
    (  0:20:40)' ( 20:20:60)'; ...
    (128:20:168)' (148:20:188)'] + 5;

% Can take up to 15min due to the 10-fold cross-validation
argOut = CovStabilityAcrossTimeAnalysis(spikes, expCond, argIn);

%%
RANK_TO_PLOT = 1;

normFactor = diag(argOut.CvR(:,:,RANK_TO_PLOT));
numTimePeriods = size(argIn.TimePeriods, 1);

figure(2);

imagesc(argOut.CvR(:,:,RANK_TO_PLOT)./repmat(normFactor', numTimePeriods, 1))

ax = gca;
ax.YDir = 'Normal';

axis square

xlabel('Time Used For Correlation')
ylabel('Time Used For Fitting')

