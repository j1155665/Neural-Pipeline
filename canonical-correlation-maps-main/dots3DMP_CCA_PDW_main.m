%% Setup
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

%% Bootstrap parameters
n_bootstrap = 1000;  % Number of bootstrap iterations
alpha = 0.05;        % Significance level

%% Collect data for high and low PDW separately
all_maps_highPDW = {};
all_maps_lowPDW = {};
n_trials_highPDW = [];
n_trials_lowPDW = [];

for i = 1:length(Info.seesion_nums)
    Info.session_date = Info.session_dates{i};
    session_num  = Info.seesion_nums(i);
    
    data_spkrate = dataStruct(session_num).data.dots3DMP.data_spkrate.postTargHold;
    event = dataStruct(session_num).data.dots3DMP.events;
    
    % Separate trials by PDW
    highPDW_trials = find(event.goodtrial == 1 & event.PDW == 1);
    lowPDW_trials = find(event.goodtrial == 1 & event.PDW == 0);
    
    n_trials_highPDW(i) = length(highPDW_trials);
    n_trials_lowPDW(i) = length(lowPDW_trials);
    
    fprintf('Session %d: High PDW trials = %d, Low PDW trials = %d\n', ...
        i, n_trials_highPDW(i), n_trials_lowPDW(i));
    
    % Create unit info
    unitInfo = create_area_map(Info, dataStruct(session_num).data);
    
    % HIGH PDW
    if ~isempty(highPDW_trials)
        Info.goodtrial = highPDW_trials;
        spikes_high{1} = create_spikes(Info, unitInfo.MST, data_spkrate);
        spikes_high{2} = create_spikes(Info, unitInfo.VPS, data_spkrate);
        expCond_high = create_expCond(event, Info.goodtrial);
        
        argOut_high = ComputeCorrMap(spikes_high, expCond_high, argIn);
        all_maps_highPDW{i} = argOut_high.CorrMap;
    else
        all_maps_highPDW{i} = [];
    end
    
    % LOW PDW
    if ~isempty(lowPDW_trials)
        Info.goodtrial = lowPDW_trials;
        spikes_low{1} = create_spikes(Info, unitInfo.MST, data_spkrate);
        spikes_low{2} = create_spikes(Info, unitInfo.VPS, data_spkrate);
        expCond_low = create_expCond(event, Info.goodtrial);
        
        argOut_low = ComputeCorrMap(spikes_low, expCond_low, argIn);
        all_maps_lowPDW{i} = argOut_low.CorrMap;
    else
        all_maps_lowPDW{i} = [];
    end
end

%% Calculate average maps
% Remove empty sessions
valid_high = ~cellfun(@isempty, all_maps_highPDW);
valid_low = ~cellfun(@isempty, all_maps_lowPDW);

cat_maps_highPDW = cat(4, all_maps_highPDW{valid_high});
cat_maps_lowPDW = cat(4, all_maps_lowPDW{valid_low});

avg_CorrMap_highPDW = mean(cat_maps_highPDW, 4);
avg_CorrMap_lowPDW = mean(cat_maps_lowPDW, 4);

%% Bootstrap analysis with subsampling
% Determine minimum number of trials for subsampling
n_subsample = min(n_trials_lowPDW);  % Subsample high PDW to match low PDW
fprintf('\nSubsampling to %d trials per session\n', n_subsample);

bootstrap_diff = zeros([size(avg_CorrMap_highPDW), n_bootstrap]);

for boot = 1:n_bootstrap
    if mod(boot, 100) == 0
        fprintf('Bootstrap iteration %d/%d\n', boot, n_bootstrap);
    end
    
    boot_maps_high = {};
    boot_maps_low = {};
    
    for i = 1:length(Info.seesion_nums)
        session_num = Info.seesion_nums(i);
        data_spkrate = dataStruct(session_num).data.dots3DMP.data_spkrate.postTargHold;
        event = dataStruct(session_num).data.dots3DMP.events;
        
        highPDW_trials = find(event.goodtrial == 1 & event.PDW == 1);
        lowPDW_trials = find(event.goodtrial == 1 & event.PDW == 0);
        
        unitInfo = create_area_map(Info, dataStruct(session_num).data);
        
        % Subsample high PDW trials
        if length(highPDW_trials) >= n_subsample
            subsample_idx = randsample(length(highPDW_trials), n_subsample);
            Info.goodtrial = highPDW_trials(subsample_idx);
            
            spikes_high{1} = create_spikes(Info, unitInfo.MST, data_spkrate);
            spikes_high{2} = create_spikes(Info, unitInfo.VPS, data_spkrate);
            expCond_high = create_expCond(event, Info.goodtrial);
            
            argOut_high = ComputeCorrMap(spikes_high, expCond_high, argIn);
            boot_maps_high{i} = argOut_high.CorrMap;
        end
        
        % Bootstrap resample low PDW trials
        if length(lowPDW_trials) >= n_subsample
            resample_idx = randsample(length(lowPDW_trials), n_subsample, true);
            Info.goodtrial = lowPDW_trials(resample_idx);
            
            spikes_low{1} = create_spikes(Info, unitInfo.MST, data_spkrate);
            spikes_low{2} = create_spikes(Info, unitInfo.VPS, data_spkrate);
            expCond_low = create_expCond(event, Info.goodtrial);
            
            argOut_low = ComputeCorrMap(spikes_low, expCond_low, argIn);
            boot_maps_low{i} = argOut_low.CorrMap;
        end
    end
    
    % Average across sessions for this bootstrap iteration
    if ~isempty(boot_maps_high) && ~isempty(boot_maps_low)
        boot_avg_high = mean(cat(4, boot_maps_high{:}), 4);
        boot_avg_low = mean(cat(4, boot_maps_low{:}), 4);
        bootstrap_diff(:,:,:,boot) = boot_avg_high - boot_avg_low;
    end
end

%% Calculate statistics
mean_diff = mean(bootstrap_diff, 4);
std_diff = std(bootstrap_diff, 0, 4);
ci_lower = prctile(bootstrap_diff, alpha/2*100, 4);
ci_upper = prctile(bootstrap_diff, (1-alpha/2)*100, 4);

% Test for significance (where CI doesn't include 0)
significant = (ci_lower > 0) | (ci_upper < 0);

%% Plotting
Info.tstart = -600;
CANONICAL_PAIR_IDX = 1;
mapDim = size(avg_CorrMap_highPDW, 2);
delays = (-argIn.MaxDelay:argIn.MaxDelay)*argIn.Binsize;
t = Info.tstart + (1:argIn.TimeStep:argIn.TimeStep*mapDim)*argIn.Binsize;

figure('Position', [100 100 1400 900]);

% High PDW
subplot(2,3,1);
imagesc(delays, t, avg_CorrMap_highPDW(:,:,CANONICAL_PAIR_IDX)');
yline(0, 'k--', 'LineWidth', 1);
colorbar;
caxis([0.18 0.23]);
ax = gca; ax.YDir = 'Normal';
xlabel('Delay (ms)'); ylabel('Time (ms)');
title('High PDW');

% Low PDW
subplot(2,3,2);
imagesc(delays, t, avg_CorrMap_lowPDW(:,:,CANONICAL_PAIR_IDX)');
yline(0, 'k--', 'LineWidth', 1);
colorbar;
caxis([0.18 0.23]);
ax = gca; ax.YDir = 'Normal';
xlabel('Delay (ms)'); ylabel('Time (ms)');
title('Low PDW');

% Difference (High - Low)
subplot(2,3,3);
imagesc(delays, t, mean_diff(:,:,CANONICAL_PAIR_IDX)');
yline(0, 'k--', 'LineWidth', 1);
colorbar;
ax = gca; ax.YDir = 'Normal';
xlabel('Delay (ms)'); ylabel('Time (ms)');
title('Difference (High - Low PDW)');

% Standard deviation of difference
subplot(2,3,4);
imagesc(delays, t, std_diff(:,:,CANONICAL_PAIR_IDX)');
yline(0, 'k--', 'LineWidth', 1);
colorbar;
ax = gca; ax.YDir = 'Normal';
xlabel('Delay (ms)'); ylabel('Time (ms)');
title('Bootstrap SD');

% Significance mask
subplot(2,3,5);
imagesc(delays, t, significant(:,:,CANONICAL_PAIR_IDX)');
yline(0, 'k--', 'LineWidth', 1);
colorbar;
ax = gca; ax.YDir = 'Normal';
xlabel('Delay (ms)'); ylabel('Time (ms)');
title(sprintf('Significant differences (p < %.2f)', alpha));

% Difference with significance overlay
subplot(2,3,6);
imagesc(delays, t, mean_diff(:,:,CANONICAL_PAIR_IDX)');
hold on;
% Overlay significant regions
[row, col] = find(significant(:,:,CANONICAL_PAIR_IDX)');
if ~isempty(row)
    plot(delays(col), t(row), 'k.', 'MarkerSize', 3);
end
yline(0, 'k--', 'LineWidth', 1);
colorbar;
ax = gca; ax.YDir = 'Normal';
xlabel('Delay (ms)'); ylabel('Time (ms)');
title('Difference with significance overlay');

%% Summary statistics
fprintf('\n=== SUMMARY ===\n');
fprintf('Number of bootstrap iterations: %d\n', n_bootstrap);
fprintf('Subsample size: %d trials\n', n_subsample);
fprintf('Mean difference (High - Low): %.4f\n', mean(mean_diff(:)));
fprintf('Percentage of significant points: %.2f%%\n', 100*mean(significant(:)));
fprintf('Max absolute difference: %.4f\n', max(abs(mean_diff(:))));

% Time-averaged analysis
time_avg_diff = squeeze(mean(mean_diff(:,:,CANONICAL_PAIR_IDX), 2));
time_avg_sig = squeeze(mean(significant(:,:,CANONICAL_PAIR_IDX), 2));

figure;
subplot(2,1,1);
plot(delays, time_avg_diff, 'LineWidth', 2);
xlabel('Delay (ms)');
ylabel('Mean correlation difference');
title('Time-averaged difference across delays');
grid on;

subplot(2,1,2);
bar(delays, time_avg_sig);
xlabel('Delay (ms)');
ylabel('Proportion significant');
title('Proportion of time points with significant difference');
grid on;