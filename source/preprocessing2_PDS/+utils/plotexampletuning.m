function plotexampletuning(data, timeInfo, eventInfo, unit)

%% Define parameters
binSize = timeInfo.binSize;
trial_start = timeInfo.trial_start;

% Custom timing parameters
plot_start_time = 0.225;  % Start plotting from 225ms
stim_onset_time = 0.425;  % Stimulus onset at 425ms  
reaction_time = 1.122;    % RT at 1122ms (425 + 697)

% Define which headings to plot
headings_to_plot = [3, 4, 5, 6];  % Only plot these headingInd values

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

% Get trial mean time (stimulus off time)
if isfield(timeInfo, 'trial_mean_time')
    trial_mean_time = timeInfo.trial_mean_time;
else
    psth_all = data.dots3DMPtuning.data_spkrate(:, iUnit);
    valid_indices = find(~cellfun(@(x) isempty(x) || any(isnan(x)), psth_all));
    
    if ~isempty(valid_indices)
        first_valid_idx = valid_indices(1);
        psth_sample = data.dots3DMPtuning.data_spkrate{first_valid_idx, iUnit};
        total_time = length(psth_sample) * binSize;
        trial_mean_time = total_time - 0.4;
    else
        trial_mean_time = nanmean(data.dots3DMPtuning.events.stimOff - ...
                                 data.dots3DMPtuning.events.stimOn);
    end
end

%% Extract data for this unit
unique_ang = unique(eventInfo.heading);
unit_num = data.dots3DMPtuning.unit.cluster_id(iUnit);
psth = data.dots3DMPtuning.data_spkrate(:, iUnit);

% Get sample for time axis
valid_indices = find(~cellfun(@(x) isempty(x) || any(isnan(x)), psth));
if isempty(valid_indices)
    fprintf('No valid data for unit %d\n', unit_num);
    return;
end
psth_sample = psth{valid_indices(1)};

%% Create time axis
timeAxis = linspace(trial_start, trial_start + (length(psth_sample)-1)*binSize, length(psth_sample));
yAxis = timeAxis * 1000; % Convert to milliseconds

% Find index for plot start time (225ms)
plot_start_idx = find(timeAxis >= plot_start_time, 1, 'first');
if isempty(plot_start_idx)
    plot_start_idx = 1;
end

% Trim time axis to start from 225ms
yAxis_trimmed = yAxis(plot_start_idx:end);

%% Organize data by condition and heading
headingInd = data.dots3DMPtuning.events.headingInd;
mod = data.dots3DMPtuning.events.modality;

% Initialize PSTH matrices - only for selected headings
n_headings = length(headings_to_plot);
vis_psth = nan(n_headings, length(yAxis));
ves_psth = nan(n_headings, length(yAxis));
com_psth = nan(n_headings, length(yAxis));

% Calculate mean PSTH for each condition and selected headings
for i = 1:n_headings
    ang = headings_to_plot(i);
    
    ves_idx = mod==1 & headingInd==ang & ~cellfun(@(x) any(isnan(x)), psth)';
    vis_idx = mod==2 & headingInd==ang & ~cellfun(@(x) any(isnan(x)), psth)';
    com_idx = mod==3 & headingInd==ang & ~cellfun(@(x) any(isnan(x)), psth)';
    
    if any(ves_idx)
        ves_data = cell2mat(psth(ves_idx));
        ves_psth(i,:) = mean(ves_data, 1);
    end
    
    if any(vis_idx)
        vis_data = cell2mat(psth(vis_idx));
        vis_psth(i,:) = mean(vis_data, 1);
    end
    
    if any(com_idx)
        com_data = cell2mat(psth(com_idx));
        com_psth(i,:) = mean(com_data, 1);
    end
end

% Trim PSTH data to match time window
ves_psth_trimmed = ves_psth(:, plot_start_idx:end);
vis_psth_trimmed = vis_psth(:, plot_start_idx:end);
com_psth_trimmed = com_psth(:, plot_start_idx:end);

%% Calculate y-axis limits using trimmed data
all_data = [ves_psth_trimmed(:); vis_psth_trimmed(:); com_psth_trimmed(:)];
all_data_clean = all_data(~isnan(all_data));
if isempty(all_data_clean)
    max_ylim = 10;
else
    max_ylim = max(10, max(all_data_clean) * 1.1);
end

%% Setup color scheme for 4 headings
% headingInd 3,4 are negative (red shades), 5,6 are positive (blue shades)
colors = [
    0.5, 0.2, 0.0;  % Very dark orange for heading 3
    0.8, 0.4, 0.0;  % Medium orange for heading 4
    0.0, 0.5, 0.5;  % Medium teal for heading 5
    0.0, 0.3, 0.3;  % Very dark teal for heading 6
];

%% Create figure
figure('Position', [100, 100, 1200, 1000]);

sgtitle(sprintf('Unit %d, Selected Heading Tuning (3-6), Depth %d', ...
    unit_num, data.dots3DMPtuning.unit.depth(iUnit)), ...
    'FontWeight', 'bold', 'FontSize', 20);

condition_labels = {'Vestibular', 'Visual', 'Combined'};
psth_data_trimmed = {ves_psth_trimmed, vis_psth_trimmed, com_psth_trimmed};

%% Plot each condition
for cond = 1:3
    subplot(3, 1, cond);
    hold on;
    
    % Add shaded region between stim onset and RT
    stim_onset_ms = stim_onset_time * 1000;
    rt_ms = reaction_time * 1000;
    stim_off_ms = trial_mean_time * 1000;
%     
%     patch([stim_onset_ms rt_ms rt_ms stim_onset_ms], ...
%           [0 0 max_ylim max_ylim], ...
%           [0.9 0.9 0.9], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    
    % Plot lines for each selected heading
    for i = 1:n_headings
        if ~all(isnan(psth_data_trimmed{cond}(i, :)))
            plot(yAxis_trimmed, psth_data_trimmed{cond}(i, :), 'LineWidth', 6, 'Color', colors(i, :));
        end
    end
    if cond == 1
    % Create legend for heading lines
    heading_labels = {'-10°', '-3.9°', '3.9°', '10°'};
    legend(heading_labels, 'Location', 'northeast', 'FontSize', 14, 'FontWeight', 'bold');
    end

    % Add reference lines
    % Stimulus onset at 425ms
    line([stim_onset_ms, stim_onset_ms], [0 max_ylim], 'Color', 'k', 'LineStyle', '--', 'LineWidth', 6);
    
    % Stimulus off
    line([stim_off_ms, stim_off_ms], [0 max_ylim], 'Color', 'k', 'LineStyle', '--', 'LineWidth', 6);
    
    % Reaction time at 1122ms
    line([rt_ms, rt_ms], [0 max_ylim], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 6);
    
    % Add labels (only on top subplot)
    if cond == 1
        text(stim_onset_ms, max_ylim*1.1, 'Stim On', 'HorizontalAlignment', 'center', ...
            'FontWeight', 'bold', 'FontSize', 16);
        text(stim_off_ms, max_ylim*1.1, 'Stim Off', 'HorizontalAlignment', 'center', ...
            'FontWeight', 'bold', 'FontSize', 16);
        text(rt_ms, max_ylim*1.1, 'RT', 'HorizontalAlignment', 'center', ...
            'FontWeight', 'bold', 'FontSize', 16, 'Color', 'r');
    end
    
    % Formatting
    ylabel('spikes/s', 'FontSize', 18, 'FontWeight', 'bold');
    text(-0.15, 0.5, condition_labels{cond}, 'Units', 'normalized', ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'Rotation', 90, 'FontWeight', 'bold', 'FontSize', 18);
    
    ylim([0 max_ylim]);
    xlim([min(yAxis_trimmed) max(yAxis_trimmed)]);
    
    % Format axes
    ax = gca;
    ax.LineWidth = 3.0;
    ax.FontSize = 16;
    ax.FontWeight = 'bold';
    ax.TickLength = [0.02 0.03];
    ax.XColor = [0 0 0];
    ax.YColor = [0 0 0];
    
    % Set x-axis ticks for all subplots
    if cond < 3
        % For top and middle subplots, show ticks but no labels
        set(gca, 'XTickLabel', []);
    else
        % For bottom subplot, set ticks every 1000ms
        xlabel('Time (ms)', 'FontSize', 18, 'FontWeight', 'bold');
        
        % Set ticks every 1000ms
        x_min = 0;  % Start from 0
        x_max = ceil(max(yAxis_trimmed)/1000)*1000;  % Round up to nearest 1000
        x_ticks = x_min:1000:x_max;
        
        % Make sure to include key time points if they're not too close to regular ticks
        key_times = [stim_onset_ms, stim_off_ms, rt_ms];
        for kt = key_times
            if kt <= max(yAxis_trimmed) && kt >= min(yAxis_trimmed)
                % Only add if not too close to existing tick (within 100ms)
                if min(abs(x_ticks - kt)) > 100
                    x_ticks = sort([x_ticks, kt]);
                end
            end
        end
        
        set(gca, 'XTick', x_ticks);
        
        % Format tick labels
        x_labels = arrayfun(@(x) sprintf('%.0f', x), x_ticks, 'UniformOutput', false);
        set(gca, 'XTickLabel', x_labels);
    end
    
    hold off;
end

% Add annotation with timing information
annotation(gcf, 'textbox', [0.15, 0.02, 0.8, 0.05], ...
    'String', sprintf('Headings 3-6: -10°, -3.9°, 3.9°, 10° | Stim: %.0fms-%.0fms | RT: %.0fms', ...
    stim_onset_ms, stim_off_ms, rt_ms), ...
    'HorizontalAlignment', 'center', 'FontSize', 14, 'EdgeColor', 'none', 'FontWeight', 'bold');

fprintf('Plotted unit %d with headings 3-6 (225ms start, 425ms stim on, %.0fms stim off, 1122ms RT)\n', ...
    unit_num, stim_off_ms);

end