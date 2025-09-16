function plotspkrate(dataStruct, timeInfo, var, eventInfo, session_num, iUnit)

%% Define constants
binSize = timeInfo.binSize;
alignEvent = timeInfo.alignEvent;
center_start = timeInfo.center_start;
center_stop = timeInfo.center_stop;
head_Info = eventInfo.name(3);
dataAnals = dataStruct(session_num).data.dots3DMP.data_spkrate;
unit_num = dataStruct(session_num).data.dots3DMP.unit.cluster_id(iUnit);
mod = dataStruct(session_num).data.dots3DMP.events.modality;
coh = dataStruct(session_num).data.dots3DMP.events.coherenceInd;
del = dataStruct(session_num).data.dots3DMP.events.delta;

% Define condition labels and indices
condition_labels = {'Ves', 'Vis Low', 'Vis High', 'Com Low', 'Com High'};
condition_indices = {
    @(m,c) m==1,                    % Vestibular
    @(m,c) m==2 & c==1,            % Visual Low
    @(m,c) m==2 & c==2,            % Visual High
    @(m,c) m==3 & c==1,            % Combined Low
    @(m,c) m==3 & c==2             % Combined High
    };

%% Setup variable-specific parameters
[Info_name, class_info, num_classes] = setupVariableParams(var, eventInfo);

%% Main plotting
figure;
set(gcf, 'Position', [100, 100, 600, 600]);
% subplot_tight = @(m,n,p,margins) subtightplot(m,n,p,margins,[0.1 0.05],[0.1 0.05]);

sgtitle(sprintf('unit %d, %s, depth %d', unit_num, Info_name{1}, ...
    dataStruct(session_num).data.dots3DMP.unit.depth(iUnit)));

y_lim = 0.1;
all_psth_data = cell(length(alignEvent), length(condition_indices));

% Calculate PSTH for all conditions and time windows
for i = 1:length(alignEvent)
    timeAxis = center_start(i):binSize:center_stop(i);

    yAxis = timeAxis * 1000;
    field_name = alignEvent{i};
    psth = dataAnals.(field_name)(:, iUnit);
    headingInd = dataStruct(session_num).data.dots3DMP.events.(head_Info{1});

    for cond = 1:length(condition_indices)
        psth_data = calculatePSTH(psth, headingInd, mod, coh, del, condition_indices{cond}, ...
            var, class_info, num_classes, yAxis, dataStruct, session_num, Info_name);
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

% Plot all conditions
for i = 1:length(alignEvent)
    timeAxis = center_start(i):binSize:center_stop(i);
    yAxis = timeAxis * 1000;

    for cond = 1:length(condition_indices)
        subplot_idx = (cond-1)*length(alignEvent) + i;
        subplot(length(condition_indices), length(alignEvent), subplot_idx);

        psth_data = all_psth_data{i, cond};
        if ~isempty(psth_data)
            plotCondition(yAxis, psth_data, var, num_classes);
        end

        % Formatting
        formatSubplot(i, cond, timeInfo.plotname, condition_labels, yAxis, y_lim, ...
            length(alignEvent), length(condition_indices));
    end
end

% Add legend and annotations
addLegendAndAnnotations(var, num_classes, length(alignEvent));

% Export and close
if y_lim >= 1.5
    fig = gcf;
    exportgraphics(fig, eventInfo.output_pdf{var}, 'Append', true);
end
close(gcf);

end

%% Helper Functions

function [Info_name, class_info, num_classes] = setupVariableParams(var, eventInfo)
switch var
    case 6 % Heading angle
        Info_name = {'heading angle'};
        class_info = [];
        num_classes = 7;
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

function psth_data = calculatePSTH(psth, headingInd, mod, coh, del, condition_func, ...
    var, class_info, num_classes, yAxis, dataStruct, session_num, Info_name)

psth_data = nan(num_classes, length(yAxis));

switch var
    case 6 % Heading angle
        for log = 1:7
            idx = condition_func(mod, coh) & headingInd==log & del'==0 & ...
                ~cellfun(@(x) any(isnan(x)), psth)';
            if any(idx)
                data = cell2mat(psth(idx));
                psth_data(log,:) = nanmean(data, 1);
            end
        end

    case 7 % Choice and PDW
        choice_name = strcat('dataStruct(', num2str(session_num), ').data.dots3DMP.events.', Info_name{2});
        choice_Indx = eval(choice_name{1});
        pdw_name = strcat('dataStruct(', num2str(session_num), ').data.dots3DMP.events.', Info_name{3});
        pdw_Indx = eval(pdw_name{1});

        for log = 1:4
            logit_ind = choice_Indx == class_info{log,2} & pdw_Indx == class_info{log,3};
            idx = condition_func(mod, coh) & ismember(headingInd, [3,4,5]) & del'==0 & ...
                logit_ind & ~cellfun(@(x) any(isnan(x)), psth)';
            if any(idx)
                data = cell2mat(psth(idx));
                psth_data(log,:) = nanmean(data, 1);
            end
        end

    otherwise % Binary classification
        field_name = strcat('dataStruct(', num2str(session_num), ').data.dots3DMP.events.', Info_name{1});
        Indx = eval(field_name);
        logit_ind = nan(size(Indx));
        logit_ind(ismember(Indx, class_info.class_1{1})) = 0;
        logit_ind(ismember(Indx, class_info.class_2{1})) = 1;

        for log = 1:2
            idx = condition_func(mod, coh) & logit_ind==(log-1) & del==0 & ...
                ~cellfun(@(x) any(isnan(x)), psth)';
            if any(idx)
                data = cell2mat(psth(idx));
                psth_data(log,:) = mean(data, 1);
            end
        end
end
end

function plotCondition(yAxis, psth_data, var, num_classes)
hold on;

switch var
    case 6 % Heading angle
        colors = getHeadingColors();
        for log = 1:7
            if ~all(isnan(psth_data(log, :)))
                plot(yAxis, psth_data(log, :), 'LineWidth', 1.2, 'Color', colors(log, :));
            end
        end

    case 7 % Choice and PDW
        colors = getChoicePDWColors();
        for log = 1:4
            if ~all(isnan(psth_data(log, :)))
                plot(yAxis, psth_data(log, :), 'LineWidth', 1.2, 'Color', colors(log, :));
            end
        end

    otherwise % Binary
        colors = {'b', 'r'};
        for log = 1:2
            if ~all(isnan(psth_data(log, :)))
                plot(yAxis, psth_data(log, :), colors{log}, 'LineWidth', 1.2);
            end
        end
end
hold off;
end

function colors = getHeadingColors()
numShades = 4;
redShades = [flip(linspace(0.5, 1, numShades)); flip(linspace(0, 0.5, numShades)); flip(linspace(0, 0.5, numShades))]';
blueShades = [linspace(0, 0.5, numShades); linspace(0, 0.5, numShades); linspace(0.5, 1, numShades)]';
colors = [redShades(1:3, :); [0.5,0.5,0.5]; blueShades(1:3, :)];
end

function colors = getChoicePDWColors()
numShades = 2;
redShades = [flip(linspace(0.5, 1, numShades)); flip(linspace(0, 0.5, numShades)); flip(linspace(0, 0.5, numShades))]';
blueShades = [linspace(0, 0.5, numShades); linspace(0, 0.5, numShades); linspace(0.5, 1, numShades)]';
colors = [blueShades; redShades];
end

function formatSubplot(time_idx, cond_idx, alignEvent, condition_labels, yAxis, y_lim, num_time, num_cond)
% Set limits and reference line
xlim([min(yAxis) max(yAxis)]);
ylim([0 y_lim*1.2]);
xline(0, 'k--');

% Y-axis label (only leftmost column)
if time_idx == 1
    ylabel('spikes / s', 'FontSize',8);
    text(-0.4, 0.5, condition_labels{cond_idx}, 'Units', 'normalized', ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'Rotation', 90, 'FontWeight', 'bold');

    % Add vertical lines and labels for motion profile landmarks
    % Max velocity at 662.3 ms
    if 662.3 >= min(yAxis) && 662.3 <= max(yAxis)
        xline(662.3, 'g--', 'LineWidth', 0.8);
        if cond_idx == 1
            text(662.3, y_lim*1.2, 'vel', 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'bottom', 'FontSize', 8, 'Color', 'g');
        end
    end

    % Max acceleration at 396.7 ms
    if 396.7 >= min(yAxis) && 396.7 <= max(yAxis)
        xline(396.7, 'm--', 'LineWidth', 0.8);
        if cond_idx == 1
            text(396.7, y_lim*1.2, 'acc', 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'bottom', 'FontSize', 8, 'Color', 'm');
        end
    end

    % Max deceleration at 927.9 ms (only if within time window)
    if 927.9 >= min(yAxis) && 927.9 <= max(yAxis)
        xline(927.9, 'c--', 'LineWidth', 0.8);
        if cond_idx == 1
            text(927.9, y_lim*1.2, 'deacc', 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'bottom', 'FontSize', 8, 'Color', 'c');
        end
    end
end


% Condition labels (left side of leftmost column)
if cond_idx == num_cond
    if time_idx == 2
        xlabel('Time (ms)');
    end
end

% Event labels (only top row, as text annotations aligned to zero)
if cond_idx == 1
    text(0, y_lim*1.2,alignEvent{time_idx} , 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom');
end
end

function addLegendAndAnnotations(var, num_classes, num_time)
switch var
    case 6
        annotation(gcf, 'textbox', [0.15, 0.02, 0.8, 0.05], ...
            'String', 'blue = positive, red = negative, gray = zero', ...
            'HorizontalAlignment', 'center', 'FontSize', 12, 'EdgeColor', 'none');
    case 7
        annotation(gcf, 'textbox', [0.15, 0.02, 0.8, 0.05], ...
            'String', 'blue = right, red = left, dark = high, light = low, only +-1.5 and 0 are included', ...
            'HorizontalAlignment', 'center', 'FontSize', 12, 'EdgeColor', 'none');
end
end