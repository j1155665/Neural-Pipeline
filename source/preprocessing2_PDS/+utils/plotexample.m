function plotexample_subset(data, timeInfo, eventInfo, unit, var)

%% Define constants
binSize = timeInfo.binSize;
alignEvent = timeInfo.alignEvent;
center_start = timeInfo.center_start;
center_stop = timeInfo.center_stop;
reaction_time = 0.697; 
% Define which headings to plot
headings_to_plot = [1, 2, 6, 7]; % Only plot these headingInd values

% Find the unit index
iUnit = find(data.dots3DMP.unit.cluster_id == unit);
if isempty(iUnit)
    fprintf('Unit %d not found in the data\n', unit);
    return;
elseif length(iUnit) > 1
    fprintf('Multiple units found with ID %d, using the first one\n', unit);
    iUnit = iUnit(1);
end

head_Info = eventInfo.name(3);
dataAnals = data.dots3DMP.data_spkrate;
unit_num = data.dots3DMP.unit.cluster_id(iUnit);
mod = data.dots3DMP.events.modality;
coh = data.dots3DMP.events.coherenceInd;
del = data.dots3DMP.events.delta;

% Define condition labels and indices (only 3 conditions as requested)
condition_labels = {'Vestibular', 'Visual High', 'Combined High'};
condition_indices = {
    @(m,c) m==1,                    % Vestibular
    @(m,c) m==2 & c==2,            % Visual High coherence
    @(m,c) m==3 & c==2             % Combined High coherence
    };

% Only plot stimOn and saccade onset (assuming these are the first 2 align events)
plot_events = [1, 2]; % Adjust indices based on your alignEvent order

%% Setup variable-specific parameters - Modified for subset
[Info_name, class_info, num_classes] = setupVariableParams_subset(var, eventInfo);

%% Calculate PSTH for both time periods
y_lim = 0.1;
all_psth_data = cell(length(plot_events), length(condition_indices));
all_time_axes = cell(length(plot_events), 1);
all_time_axes_original = cell(length(plot_events), 1); % Keep original time axes for labeling

% Calculate PSTH for selected conditions and time windows
for i = 1:length(plot_events)
    event_idx = plot_events(i);
    
    % Get the original time axis and PSTH data first
    timeAxis = center_start(event_idx):binSize:center_stop(event_idx);
    yAxis = timeAxis * 1000;
    all_time_axes_original{i} = yAxis; % Store original for labeling
    
    field_name = alignEvent{event_idx};
    psth = dataAnals.(field_name)(:, iUnit);
    headingInd = data.dots3DMP.events.(head_Info{1});
    
    % For saccade period, we'll trim the data to start from -400ms
    if i == 2 % Saccade period
        saccade_start_ms = -400; % -400ms
        % Find the index corresponding to -400ms
        start_idx = find(yAxis >= saccade_start_ms, 1, 'first');
        if ~isempty(start_idx)
            yAxis = yAxis(start_idx:end);
            % We'll trim the PSTH data after calculating it
            trim_saccade = true;
            trim_start_idx = start_idx;
        else
            trim_saccade = false;
            trim_start_idx = 1;
        end
    else
        trim_saccade = false;
        trim_start_idx = 1;
    end
    
    all_time_axes{i} = yAxis;

    for cond = 1:length(condition_indices)
        psth_data = calculatePSTH_subset(psth, headingInd, mod, coh, del, condition_indices{cond}, ...
            var, class_info, num_classes, data, Info_name, headings_to_plot);
        
        % Trim saccade data if needed
        if trim_saccade && ~isempty(psth_data)
            psth_data = psth_data(:, trim_start_idx:end);
        end
        
        all_psth_data{i, cond} = psth_data;

        % Update y_lim
        if ~isempty(psth_data)
            current_max = max(psth_data(:));
            if y_lim < current_max
                y_lim = current_max;
            end
        end
    end
end

% Apply smoothing to all PSTH data
smoothing_window = 1; % No smoothing - keep it real!
for i = 1:length(plot_events)
    for cond = 1:length(condition_indices)
        if ~isempty(all_psth_data{i, cond})
            for class = 1:size(all_psth_data{i, cond}, 1)
                if ~all(isnan(all_psth_data{i, cond}(class, :)))
                    all_psth_data{i, cond}(class, :) = moving_average_smooth(all_psth_data{i, cond}(class, :), smoothing_window);
                end
            end
        end
    end
end

%% Create combined time axes with proper spacing
yAxis_stim = all_time_axes{1};
yAxis_sacc_original = all_time_axes_original{2}; % Original saccade time axis for labeling
yAxis_sacc = all_time_axes{2};

% Calculate gap: max time of stim + small gap
gap_size = 50; % 50ms gap
sacc_plot_start = max(yAxis_stim) + gap_size;

% Create plotting positions for saccade data (offset from original times)
yAxis_sacc_plot = yAxis_sacc - min(yAxis_sacc) + sacc_plot_start;

%% Main plotting - MUCH LARGER for poster
figure;
set(gcf, 'Position', [100, 100, 1400, 800]); % Much larger figure

sgtitle(sprintf('Unit %d, %s (Headings 1,2,6,7), Depth %d', unit_num, Info_name{1}, ...
    data.dots3DMP.unit.depth(iUnit)), 'FontWeight', 'bold', 'FontSize', 20); % Much larger title

%% Plot all conditions in combined format
for cond = 1:length(condition_indices)
    subplot(length(condition_indices), 1, cond);
    hold on;
    
    % Plot STIM period
    psth_data_stim = all_psth_data{1, cond};
    if ~isempty(psth_data_stim)
        plotCondition_subset(yAxis_stim, psth_data_stim, var, num_classes);
    end
    
    % Plot SACCADE period
    psth_data_sacc = all_psth_data{2, cond};
    if ~isempty(psth_data_sacc)
        plotCondition_subset(yAxis_sacc_plot, psth_data_sacc, var, num_classes);
    end
    
    % Add reference lines with VERY thick lines
    % Stim onset (at 0 for stim period)
    line([0 0], [0 y_lim*1.2], 'Color', 'k', 'LineStyle', '--', 'LineWidth', 6);
    reaction_pos = reaction_time *1000;
    line([reaction_pos reaction_pos], [0 y_lim*1.2], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 6);
    % Saccade onset (at the start of saccade plot area, representing time 0 for saccade)
    saccade_zero_plot_pos = sacc_plot_start - min(yAxis_sacc); % Position where saccade time = 0
    line([saccade_zero_plot_pos, saccade_zero_plot_pos], [0 y_lim*1.2], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 6);
    
    % Add period labels (only on top subplot)
    if cond == 1
        text(0, y_lim*1.35, 'Stim On', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 18);
        text(saccade_zero_plot_pos, y_lim*1.35, 'Saccade', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 18);
    end
    
    % Formatting with VERY thick axes
    ylabel('spikes/s', 'FontSize', 18, 'FontWeight', 'bold');
    
    % Move modality labels further left to avoid overlap
    text(-0.15, 0.5, condition_labels{cond}, 'Units', 'normalized', ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'Rotation', 90, 'FontWeight', 'bold', 'FontSize', 18);
    
    ylim([0 y_lim*1.2]);
    xlim([min(yAxis_stim) max(yAxis_sacc_plot)]);
    
    % Make axes VERY thick and more visible for poster
    ax = gca;
    ax.LineWidth = 3.0; % Very thick axis lines
    ax.FontSize = 16; % Large font for tick labels
    ax.FontWeight = 'bold'; % Bold tick labels
    ax.TickLength = [0.02 0.03]; % Longer tick marks
    ax.XColor = [0 0 0]; % Ensure black axes
    ax.YColor = [0 0 0]; % Ensure black axes
    
    % Custom x-axis ticks and labels
    if cond == length(condition_indices) % Only on bottom subplot
        % Create custom ticks - more reasonable spacing
        stim_range = max(yAxis_stim) - min(yAxis_stim);
        sacc_range = max(yAxis_sacc_plot) - min(yAxis_sacc_plot);
        
        % Stim ticks every ~200ms or reasonable intervals
        stim_tick_interval = max(1, round(length(yAxis_stim)/5));
        stim_ticks = yAxis_stim(1:stim_tick_interval:end);
        
        % Saccade ticks every ~200ms or reasonable intervals  
        sacc_tick_interval = max(1, round(length(yAxis_sacc_plot)/5));
        sacc_ticks = yAxis_sacc_plot(1:sacc_tick_interval:end);
        
        % Corresponding labels (original time values)
        stim_labels = arrayfun(@(x) sprintf('%.0f', x), stim_ticks, 'UniformOutput', false);
        
        % For saccade labels, convert back to original saccade times
        sacc_original_times = yAxis_sacc(1:sacc_tick_interval:end);
        sacc_labels = arrayfun(@(x) sprintf('%.0f', x), sacc_original_times, 'UniformOutput', false);
        
        % Set ticks and labels
        all_ticks = [stim_ticks, sacc_ticks];
        all_labels = [stim_labels, sacc_labels];
        
        set(gca, 'XTick', all_ticks, 'XTickLabel', all_labels);
        xlabel('Time (ms): Stim-aligned | Saccade-aligned', 'FontSize', 18, 'FontWeight', 'bold');
        
    else
        % For non-bottom subplots, remove x-tick labels but keep ticks
        stim_tick_interval = max(1, round(length(yAxis_stim)/5));
        stim_ticks = yAxis_stim(1:stim_tick_interval:end);
        sacc_tick_interval = max(1, round(length(yAxis_sacc_plot)/5));
        sacc_ticks = yAxis_sacc_plot(1:sacc_tick_interval:end);
        all_ticks = [stim_ticks, sacc_ticks];
        set(gca, 'XTick', all_ticks, 'XTickLabel', []);
    end
    
    hold off;
end

% Add legend for heading subset
if var == 6
    annotation(gcf, 'textbox', [0.15, 0.02, 0.8, 0.05], ...
        'String', 'Headings 1,2,6,7 only: dark red = -10°, light red = -3.9°, light blue = 3.9°, dark blue = 10°', ...
        'HorizontalAlignment', 'center', 'FontSize', 16, 'EdgeColor', 'none', 'FontWeight', 'bold');
end

fprintf('Plotted unit %d with headings 1,2,6,7 only (POSTER-READY)\n', unit_num);

end

%% Modified Helper Functions

function smoothed_data = moving_average_smooth(data, window_size)
    % Simple moving average smoothing
    if window_size <= 1
        smoothed_data = data;
        return;
    end
    
    % Pad the data at edges
    half_window = floor(window_size/2);
    padded_data = [repmat(data(1), 1, half_window), data, repmat(data(end), 1, half_window)];
    
    % Apply moving average
    smoothed_data = zeros(size(data));
    for i = 1:length(data)
        smoothed_data(i) = mean(padded_data(i:i+window_size-1));
    end
end

function [Info_name, class_info, num_classes] = setupVariableParams_subset(var, eventInfo)
switch var
    case 6 % Heading angle - modified for subset
        Info_name = {'heading angle'};
        class_info = [];
        num_classes = 4; % Only 4 headings now
    case 7 % Choice and PDW
        Info_name = {'choice and PDW', eventInfo.name(4), eventInfo.name(5)};
        class_info = [
            [0, eventInfo.class_2(4), eventInfo.class_2(5)];
            [0, eventInfo.class_2(4), eventInfo.class_1(5)];
            [0, eventInfo.class_1(4), eventInfo.class_1(5)];
            [0, eventInfo.class_1(4), eventInfo.class_2(5)]
            ];
        num_classes = 4;
    otherwise
        Info_name = eventInfo.name(var);
        class_info.class_1 = eventInfo.class_1(var);
        class_info.class_2 = eventInfo.class_2(var);
        num_classes = 2;
end
end

function psth_data = calculatePSTH_subset(psth, headingInd, mod, coh, del, condition_func, ...
    var, class_info, num_classes, data, Info_name, headings_to_plot)

% Get the actual length from a sample PSTH
sample_psth = psth{find(~cellfun(@isempty, psth), 1, 'first')};
if isempty(sample_psth)
    psth_data = [];
    return;
end

psth_data = nan(num_classes, length(sample_psth));

switch var
    case 6 % Heading angle - modified for subset
        for i = 1:length(headings_to_plot)
            log = headings_to_plot(i);
            idx = condition_func(mod, coh) & headingInd==log & del'==0 & ...
                ~cellfun(@(x) any(isnan(x)), psth)';
            if any(idx)
                data_mat = cell2mat(psth(idx));
                psth_data(i,:) = nanmean(data_mat, 1);
            end
        end

    case 7 % Choice and PDW
        choice_Indx = data.dots3DMP.events.(Info_name{2});
        pdw_Indx = data.dots3DMP.events.(Info_name{3});

        for log = 1:4
            logit_ind = choice_Indx == class_info(log,2) & pdw_Indx == class_info(log,3);
            idx = condition_func(mod, coh) & ismember(headingInd, [3,4,5]) & del'==0 & ...
                logit_ind & ~cellfun(@(x) any(isnan(x)), psth)';
            if any(idx)
                data_mat = cell2mat(psth(idx));
                psth_data(log,:) = nanmean(data_mat, 1);
            end
        end

    otherwise % Binary classification
        Indx = data.dots3DMP.events.(Info_name{1});
        logit_ind = nan(size(Indx));
        logit_ind(ismember(Indx, class_info.class_1{1})) = 0;
        logit_ind(ismember(Indx, class_info.class_2{1})) = 1;

        for log = 1:2
            idx = condition_func(mod, coh) & logit_ind==(log-1) & del'==0 & ...
                ~cellfun(@(x) any(isnan(x)), psth)';
            if any(idx)
                data_mat = cell2mat(psth(idx));
                psth_data(log,:) = mean(data_mat, 1);
            end
        end
end
end

function plotCondition_subset(yAxis, psth_data, var, num_classes)
hold on;

switch var
    case 6 % Heading angle - modified for subset
        colors = getHeadingColors_subset();
        for i = 1:4  % Only 4 headings now
            if ~all(isnan(psth_data(i, :)))
                plot(yAxis, psth_data(i, :), 'LineWidth', 6, 'Color', colors(i, :)); % Very thick plot lines
            end
        end

    case 7 % Choice and PDW
        colors = getChoicePDWColors();
        for log = 1:4
            if ~all(isnan(psth_data(log, :)))
                plot(yAxis, psth_data(log, :), 'LineWidth', 6, 'Color', colors(log, :)); % Very thick plot lines
            end
        end

    otherwise % Binary
        colors = {'r', 'b'};
        for log = 1:2
            if ~all(isnan(psth_data(log, :)))
                plot(yAxis, psth_data(log, :), colors{log}, 'LineWidth', 6); % Very thick plot lines
            end
        end
end
hold off;
end

function colors = getHeadingColors_subset()
% Colors for headings 1, 2, 6, 7
% headingInd 1,2 are negative (red shades), 6,7 are positive (blue shades)
colors = [
    0.5, 0.2, 0.0;  % Very dark orange for heading 3
    0.8, 0.4, 0.0;  % Medium orange for heading 4
    0.0, 0.5, 0.5;  % Medium teal for heading 5
    0.0, 0.3, 0.3;  % Very dark teal for heading 6
];
end

function colors = getChoicePDWColors()
numShades = 2;
redShades = [flip(linspace(0.5, 1, numShades)); flip(linspace(0, 0.5, numShades)); flip(linspace(0, 0.5, numShades))]';
blueShades = [linspace(0, 0.5, numShades); linspace(0, 0.5, numShades); linspace(0.5, 1, numShades)]';
colors = [blueShades; redShades];
end