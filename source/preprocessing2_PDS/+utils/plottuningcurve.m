function plottuningcurve(data, timeInfo, eventInfo, unit, time_window)
% Plot tuning curves (firing rate vs heading) for all three modalities
% 
% Inputs:
%   data - data structure
%   timeInfo - timing information
%   eventInfo - event information  
%   unit - unit cluster ID to plot
%   time_window - [start_ms, end_ms] time window for calculating firing rates
%                 e.g., [200, 1000] for 200-1000ms after stimulus onset

%% Define parameters
binSize = timeInfo.binSize;
trial_start = timeInfo.trial_start;

% Default time window if not provided
if nargin < 5
    time_window = [200, 1000]; % Default 200-1000ms after stim onset
end

% Find the unit index
iUnit = find(data.dots3DMPtuning.unit.cluster_id == unit);

% Check if unit was found
if isempty(iUnit)
    fprintf('Unit %d not found in the data\n', unit);
    return;
elseif length(iUnit) > 1
    fprintf('Multiple units found with ID %d, using the first one\n', unit);
    iUnit = iUnit(1);
end

%% Extract data for this unit
unique_ang = unique(eventInfo.heading);
heading_values = [-45, -21.2, -10, -3.9, 3.9, 10, 21.2, 45]; % Actual heading angles
unit_num = data.dots3DMPtuning.unit.cluster_id(iUnit);
psth = data.dots3DMPtuning.data_spkrate(:, iUnit);

% Get sample for time axis
valid_indices = find(~cellfun(@(x) isempty(x) || any(isnan(x)), psth));
if isempty(valid_indices)
    fprintf('No valid data for unit %d\n', unit_num);
    return;
end
psth_sample = psth{valid_indices(1)};

% Create time axis
timeAxis = linspace(trial_start, trial_start + (length(psth_sample)-1)*binSize, length(psth_sample));
timeAxis_ms = timeAxis * 1000; % Convert to milliseconds

% Find time window indices
[~, start_idx] = min(abs(timeAxis_ms - time_window(1)));
[~, end_idx] = min(abs(timeAxis_ms - time_window(2)));

%% Calculate mean firing rates for each heading and modality
headingInd = data.dots3DMPtuning.events.headingInd;
mod = data.dots3DMPtuning.events.modality;

% Initialize firing rate arrays
ves_rates = nan(length(unique_ang), 1);
ves_sem = nan(length(unique_ang), 1);
vis_rates = nan(length(unique_ang), 1);
vis_sem = nan(length(unique_ang), 1);
com_rates = nan(length(unique_ang), 1);
com_sem = nan(length(unique_ang), 1);

% Calculate mean firing rates for each heading
for ang = 1:length(unique_ang)
    % Vestibular
    ves_idx = mod==1 & headingInd==ang & ~cellfun(@(x) any(isnan(x)), psth)';
    if any(ves_idx)
        ves_data = cell2mat(psth(ves_idx));
        ves_window_data = ves_data(:, start_idx:end_idx);
        ves_trial_means = mean(ves_window_data, 2);
        ves_rates(ang) = mean(ves_trial_means);
        ves_sem(ang) = std(ves_trial_means) / sqrt(length(ves_trial_means));
    end
    
    % Visual
    vis_idx = mod==2 & headingInd==ang & ~cellfun(@(x) any(isnan(x)), psth)';
    if any(vis_idx)
        vis_data = cell2mat(psth(vis_idx));
        vis_window_data = vis_data(:, start_idx:end_idx);
        vis_trial_means = mean(vis_window_data, 2);
        vis_rates(ang) = mean(vis_trial_means);
        vis_sem(ang) = std(vis_trial_means) / sqrt(length(vis_trial_means));
    end
    
    % Combined
    com_idx = mod==3 & headingInd==ang & ~cellfun(@(x) any(isnan(x)), psth)';
    if any(com_idx)
        com_data = cell2mat(psth(com_idx));
        com_window_data = com_data(:, start_idx:end_idx);
        com_trial_means = mean(com_window_data, 2);
        com_rates(ang) = mean(com_trial_means);
        com_sem(ang) = std(com_trial_means) / sqrt(length(com_trial_means));
    end
end

%% Create figure
figure('Position', [100, 100, 800, 600]);

% Plot tuning curves with error bars
hold on;

% Vestibular - black
h1 = errorbar(heading_values, ves_rates, ves_sem, 'ko-', ...
    'LineWidth', 6, 'MarkerSize', 10, 'MarkerFaceColor', 'k', ...
    'CapSize', 10);

% Visual - red  
h2 = errorbar(heading_values, vis_rates, vis_sem, 'ro-', ...
    'LineWidth', 6, 'MarkerSize', 10, 'MarkerFaceColor', 'r', ...
    'CapSize', 10);

% Combined - blue
h3 = errorbar(heading_values, com_rates, com_sem, 'bo-', ...
    'LineWidth', 6, 'MarkerSize', 10, 'MarkerFaceColor', 'b', ...
    'CapSize', 10);

% Add vertical line at 0 heading
line([0 0], ylim, 'Color', [0.5 0.5 0.5], 'LineStyle', '--', 'LineWidth', 2);

% Formatting
xlabel('Heading Angle (°)', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Firing Rate (spikes/s)', 'FontSize', 16, 'FontWeight', 'bold');
title(sprintf('Unit %d Tuning Curves (%d-%d ms)', unit_num, time_window(1), time_window(2)), ...
    'FontSize', 18, 'FontWeight', 'bold');

% Set x-axis limits and ticks
xlim([-50 50]);
set(gca, 'XTick', heading_values);

% Make axes thicker and more visible
ax = gca;
ax.LineWidth = 2.0;
ax.FontSize = 14;
ax.FontWeight = 'bold';
ax.Box = 'on';
ax.TickLength = [0.02 0.025];

% Add legend

grid off;


hold off;

% Print summary statistics
fprintf('\nUnit %d Tuning Curve Summary (%d-%d ms):\n', unit_num, time_window(1), time_window(2));
fprintf('Vestibular: min=%.1f, max=%.1f spikes/s\n', nanmin(ves_rates), nanmax(ves_rates));
fprintf('Visual: min=%.1f, max=%.1f spikes/s\n', nanmin(vis_rates), nanmax(vis_rates));
fprintf('Combined: min=%.1f, max=%.1f spikes/s\n', nanmin(com_rates), nanmax(com_rates));

end