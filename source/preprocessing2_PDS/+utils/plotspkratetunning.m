function plotspkratetunning(dataStruct, timeInfo,eventInfo,session_num,iUnit)

%% define
binSize = timeInfo.binSize;

if isfield(timeInfo, 'trial_mean_time')
    trial_mean_time = timeInfo.trial_mean_time;
else
    psth_all = dataStruct(session_num).data.dots3DMPtuning.data_spkrate(:, iUnit);
    valid_indices = find(~cellfun(@(x) isempty(x) || any(isnan(x)), psth_all));

    if ~isempty(valid_indices)
        first_valid_idx = valid_indices(1);
        psth_sample = dataStruct(session_num).data.dots3DMPtuning.data_spkrate{first_valid_idx, iUnit};
        total_time = length(psth_sample) * binSize;
        trial_mean_time = total_time - 0.4;
    else
        % Fallback to calculating from events if no valid PSTH is available
        trial_mean_time = nanmean(dataStruct(session_num).data.dots3DMPtuning.events.stimOff - dataStruct(session_num).data.dots3DMPtuning.events.stimOn);
    end
end

trial_start = timeInfo.trial_start;


unique_ang = unique(eventInfo.heading);

Info_name = {'heading angle'};
unit_num = dataStruct(session_num).data.dots3DMPtuning.unit.cluster_id(iUnit);

timeAxis = linspace(trial_start, trial_start + (length(psth_sample)-1)*binSize, length(psth_sample));
yAxis = timeAxis * 1000;
psth = dataStruct(session_num).data.dots3DMPtuning.data_spkrate(:, iUnit);
valid_psth = psth(~cellfun(@(x) any(isnan(x)), psth));
valid_psth = cell2mat(valid_psth);
mean_psth = mean(valid_psth, 'all'); 

if mean_psth <= 1
    return
end

figure
set(gcf, 'Position', [100, 100, 600, 400]);
annotation('textbox', [0, 0.95, 1, 0.05], ...
    'String', sprintf('trial %d, unit %d, %s, depth %d',session_num, unit_num, Info_name{1}, dataStruct(session_num).data.dots3DMPtuning.unit.depth(iUnit)), ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', ...
    'EdgeColor', 'none', ...
    'FontWeight', 'bold', ...
    'FontSize', 12);

headingInd = dataStruct(session_num).data.dots3DMPtuning.events.headingInd;

vis_psth = nan(length(unique_ang), length(yAxis));
ves_psth = nan(length(unique_ang), length(yAxis));
com_psth = nan(length(unique_ang), length(yAxis));

mod = dataStruct(session_num).data.dots3DMPtuning.events.modality;

for ang = 1:length(unique_ang)
    ves_idx = mod==1 &  headingInd==ang & ~cellfun(@(x) any(isnan(x)), psth)';
    vis_idx = mod==2 &  headingInd==ang & ~cellfun(@(x) any(isnan(x)), psth)';
    com_idx = mod==3 &  headingInd==ang & ~cellfun(@(x) any(isnan(x)), psth)';
    ves_data = cell2mat(psth(ves_idx));
    vis_data = cell2mat(psth(vis_idx));
    com_data = cell2mat(psth(com_idx));
    ves_psth(ang,:) = mean(ves_data, 1);
    vis_psth(ang,:) = mean(vis_data, 1);
    com_psth(ang,:) = mean(com_data, 1);
end

% Calculate consistent y-axis limits across all subplots
all_data = [ves_psth(:); vis_psth(:); com_psth(:)];
all_data = all_data(~isnan(all_data));
if isempty(all_data)
    y_lim = 10;
else
    y_max = max(all_data);
    if y_max <= 10
        y_lim = 10;
    else
        y_lim = y_max * 1.1; % Add 10% padding
    end
end

% Define the number of shades
if rem(length(unique_ang), 2) == 0
    numShades = length(unique_ang) / 2;
    plotblue = (numShades+1):length(unique_ang) ;
else
    numShades = (length(unique_ang)-1) / 2;
    plotgray = numShades+1;
    plotblue = (numShades+2):length(unique_ang) ;
end

plotred = 1:numShades;

redShades = [flip(linspace(0.5, 1, numShades)); flip(linspace(0, 0.5, numShades)); flip(linspace(0, 0.5, numShades))]';
blueShades = [linspace(0, 0.5, numShades); linspace(0, 0.5, numShades); linspace(0.5, 1, numShades)]';

% Define modality labels
condition_labels = {'Vestibular', 'Visual', 'Combined'};

% Plot vestibular data
subplot(3, 1, 1);
for ang = plotred
    plot(yAxis, ves_psth(ang, :), 'LineWidth', 1.5, 'Color', redShades(numShades - ang+1, :));
    hold on;
end

if exist('plotgray', 'var')
    plot(yAxis, ves_psth(plotgray, :), 'LineWidth', 1.5, 'Color', [0.5,0.5,0.5]);
end

for ang = plotblue
    plot(yAxis, ves_psth(ang, :), 'LineWidth', 1.5, 'Color', blueShades(numShades * 2 - ang +1, :));
    hold on;
end
xline(0, 'k--', 'LineWidth', 1.5);
xline(trial_mean_time* 1000, 'k--', 'LineWidth', 1.5);

% Add modality label and motion profile markers
ylabel('spikes / s', 'FontSize', 8);
text(-0.1, 0.5, condition_labels{1}, 'Units', 'normalized', ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
    'Rotation', 90, 'FontWeight', 'bold');

% Add vertical lines and labels for motion profile landmarks
% Max acceleration at 821.7 ms
if 821.7 >= min(yAxis) && 821.7 <= max(yAxis)
    xline(821.7, 'm--', 'LineWidth', 1.5);
    text(821.7, y_lim*1, 'acc', 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', 'Color', 'm');
end

% Max velocity at 1087.3 ms
if 1087.3 >= min(yAxis) && 1087.3 <= max(yAxis)
    xline(1087.3, 'g--', 'LineWidth', 1.5);
    text(1087.3, y_lim*1, 'vel', 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', 'Color', 'g');
end

% Max deceleration at 1352.9 ms
if 1352.9 >= min(yAxis) && 1352.9 <= max(yAxis)
    xline(1352.9, 'c--', 'LineWidth', 1.5);
    text(1352.9, y_lim, 'deacc', 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', 'Color', 'c');
end

ylim([0 y_lim]);
xlim([min(yAxis) max(yAxis)]);
hold off

% Plot visual data
subplot(3, 1, 2);
if exist('plotgray', 'var')
    plot(yAxis, vis_psth(plotgray, :), 'LineWidth', 1.5, 'Color', [0.5,0.5,0.5]);
    hold on;
end

for ang = plotblue
    plot(yAxis, vis_psth(ang, :), 'LineWidth', 1.5, 'Color', blueShades(numShades*2 - ang +1, :));
    hold on;
end
for ang = plotred
    plot(yAxis, vis_psth(ang, :), 'LineWidth', 1.5, 'Color', redShades(numShades - ang+1, :));
    hold on;
end
xline(0, 'k--', 'LineWidth', 1.5);
xline(trial_mean_time* 1000, 'k--', 'LineWidth', 1.5);

% Add modality label
ylabel('spikes / s', 'FontSize', 8);
text(-0.1, 0.5, condition_labels{2}, 'Units', 'normalized', ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
    'Rotation', 90, 'FontWeight', 'bold');

% Add vertical lines and labels for motion profile landmarks
% Max acceleration at 821.7 ms
if 821.7 >= min(yAxis) && 821.7 <= max(yAxis)
    xline(821.7, 'm--', 'LineWidth', 1.5);
end

% Max velocity at 1087.3 ms
if 1087.3 >= min(yAxis) && 1087.3 <= max(yAxis)
    xline(1087.3, 'g--', 'LineWidth', 1.5);

end

% Max deceleration at 1352.9 ms
if 1352.9 >= min(yAxis) && 1352.9 <= max(yAxis)
    xline(1352.9, 'c--', 'LineWidth', 1.5);
end

ylim([0 y_lim]);
xlim([min(yAxis) max(yAxis)]);
hold off

% Plot combined data
subplot(3, 1, 3);
if exist('plotgray', 'var')
    plot(yAxis, com_psth(plotgray, :), 'LineWidth', 1.5, 'Color', [0.5,0.5,0.5]);
    hold on;
end

for ang = plotblue
    plot(yAxis, com_psth(ang, :), 'LineWidth', 1.5, 'Color', blueShades(numShades*2 - ang +1, :));
    hold on;
end

for ang = plotred
    plot(yAxis, com_psth(ang, :), 'LineWidth', 1.5, 'Color', redShades(numShades - ang +1, :));
    hold on;
end

xline(0, 'k--', 'LineWidth', 1.5);
xline(trial_mean_time* 1000, 'k--', 'LineWidth', 1.5);

% Add modality label
ylabel('spikes / s', 'FontSize', 8);
text(-0.1, 0.5, condition_labels{3}, 'Units', 'normalized', ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
    'Rotation', 90, 'FontWeight', 'bold');


% Add vertical lines and labels for motion profile landmarks
% Max acceleration at 821.7 ms
if 821.7 >= min(yAxis) && 821.7 <= max(yAxis)
    xline(821.7, 'm--', 'LineWidth', 1.5);
end

% Max velocity at 1087.3 ms
if 1087.3 >= min(yAxis) && 1087.3 <= max(yAxis)
    xline(1087.3, 'g--', 'LineWidth', 1.5);

end

% Max deceleration at 1352.9 ms
if 1352.9 >= min(yAxis) && 1352.9 <= max(yAxis)
    xline(1352.9, 'c--', 'LineWidth', 1.5);
end


ylim([0 y_lim]);
xlim([min(yAxis) max(yAxis)]);
hold off

exportgraphics(gcf, eventInfo.output_pdf , 'Append', true);
close(gcf);
fprintf('Saved unit %d to PDF.\n', unit_num);

end