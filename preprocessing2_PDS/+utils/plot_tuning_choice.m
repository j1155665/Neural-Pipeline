function plot_tuning_choice(dataStruct, timeInfo, eventInfo, session_num, iUnit)

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
headingInd = dataStruct(session_num).data.dots3DMP.events.(head_Info{1});
headings = dataStruct(session_num).data.dots3DMP.events.heading;
choice = dataStruct(session_num).data.dots3DMP.events.choice;
goodtrial = dataStruct(session_num).data.dots3DMP.events.goodtrial;
correct = dataStruct(session_num).data.dots3DMP.events.correct;
rt = dataStruct(session_num).data.dots3DMP.events.RT';

heading_values = unique(headings);
heading_values = heading_values(~isnan(heading_values));

% Merge similar heading values by rounding to 1 decimal place
heading_values_rounded = round(heading_values, 1);
[unique_rounded, ~, ic] = unique(heading_values_rounded);
% For each unique rounded value, take the mean of original values
heading_values_merged = arrayfun(@(x) mean(heading_values(heading_values_rounded == x)), unique_rounded);

condition_labels = {'Vestibular', 'Visual (Low Coh)', 'Visual (High Coh)', 'Combined (Low Coh)', 'Combined (High Coh)'};
condition_indices = {
    @(m,c) m==1,
    @(m,c) m==2 & c==1,
    @(m,c) m==2 & c==2,
    @(m,c) m==3 & c==1,
    @(m,c) m==3 & c==2
};

% Color scheme: 
% Vestibular: black
% Visual: red (high coh) / light red (low coh)
% Combined: blue (high coh) / light blue (low coh)
colors = {
    [0, 0, 0],           % Black for Vestibular
    [1, 0.3, 0.3],       % Light red for Visual Low Coh
    [1, 0, 0],           % Red for Visual High Coh
    [0.3, 0.3, 1],       % Light blue for Combined Low Coh
    [0, 0, 1]            % Blue for Combined High Coh
};

figure;
set(gcf, 'Position', [100, 100, 1800, 600]);

sgtitle(sprintf('unit %d, Tuning Curves Split by Choice, depth %d', unit_num, ...
    dataStruct(session_num).data.dots3DMP.unit.depth(iUnit)));

align_idx = find(strcmp(alignEvent, 'stimOn'));
if isempty(align_idx)
    align_idx = 1;
end

timeAxis = center_start(align_idx):binSize:center_stop(align_idx);

field_name = alignEvent{align_idx};
psth = dataAnals.(field_name)(:, iUnit);



y_lim = 0;

for cond = 1:5
    subplot(2, 3, cond);
    hold on;
    
    mean_fr = nan(1, 7);
    fr_choice_left = nan(1, 7);
    fr_choice_right = nan(1, 7);

    
    cond_idx = condition_indices{cond}(mod, coh) & del'==0 & ...
            ~cellfun(@(x) any(isnan(x)), psth)' & goodtrial ==1;
    valid_rt = rt(cond_idx);  
    mean_rt = nanmean(valid_rt);
    yAxis = (timeAxis -mean_rt) * 1000;
    time_window = yAxis >= -400 & yAxis <= -200;
    
    for log = 1:7
        idx_all = condition_indices{cond}(mod, coh) & headingInd==log & del'==0 & ...
            ~cellfun(@(x) any(isnan(x)), psth)' & goodtrial ==1;
        
        
        if any(idx_all)
            data = cell2mat(psth(idx_all));
            mean_fr(log) = nanmean(data(:, time_window), 'all');
            
            idx_left = idx_all & choice == 1;
            if any(idx_left)
                data_left = cell2mat(psth(idx_left));
                fr_choice_left(log) = nanmean(data_left(:, time_window), 'all');
            end
            
            idx_right = idx_all & choice == 2;
            if any(idx_right)
                data_right = cell2mat(psth(idx_right));
                fr_choice_right(log) = nanmean(data_right(:, time_window), 'all');
            end
        end
    end
    
    valid_mean = ~isnan(mean_fr);
    plot(heading_values_merged(valid_mean), mean_fr(valid_mean), '-o', 'Color', colors{cond}, 'LineWidth', 2, ...
        'MarkerSize', 8, 'MarkerFaceColor', colors{cond});
    
    valid_left = ~isnan(fr_choice_left);
    plot(heading_values_merged(valid_left), fr_choice_left(valid_left), '--<', 'Color', colors{cond}, 'LineWidth', 1.5, ...
        'MarkerSize', 8, 'MarkerFaceColor', colors{cond}, 'MarkerEdgeColor', colors{cond});
    
    valid_right = ~isnan(fr_choice_right);
    plot(heading_values_merged(valid_right), fr_choice_right(valid_right), '-->', 'Color', colors{cond}, 'LineWidth', 1.5, ...
        'MarkerSize', 8, 'MarkerFaceColor', colors{cond}, 'MarkerEdgeColor', colors{cond});
    
    xlabel('Heading (deg)');
    ylabel('Firing Rate (spikes/s)');
    title(condition_labels{cond});
    xlim([min(heading_values_merged) max(heading_values_merged)]);
    grid on;
    
    current_max = max([mean_fr(valid_mean), fr_choice_left(valid_left), fr_choice_right(valid_right)]);
    if current_max > y_lim
        y_lim = current_max;
    end
    
    hold off;
end

for cond = 1:5
    subplot(2, 3, cond);
    ylim([0 y_lim*1.1]);
end

subplot(2, 3, 1);
lgd = legend({'Mean FR', 'Left Choice', 'Right Choice'}, 'Location', 'best');

if y_lim >= 1.5
    fig = gcf;
    [filepath, name, ext] = fileparts(eventInfo.output_pdf{6});
    output_file = fullfile(filepath, [name '_tuning' ext]);
    exportgraphics(fig, output_file, 'Append', true);
end
close(gcf);

end