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

heading_values = unique(headings);
heading_values = heading_values(~isnan(heading_values));

condition_labels = {'Vestibular', 'Visual (High Coh)', 'Combined (High Coh)'};
condition_indices = {
    @(m,c) m==1,
    @(m,c) m==2 & c==2,
    @(m,c) m==3 & c==2
};
colors = {'k', 'r', 'b'};

figure;
set(gcf, 'Position', [100, 100, 1400, 400]);

sgtitle(sprintf('unit %d, Tuning Curves Split by Choice, depth %d', unit_num, ...
    dataStruct(session_num).data.dots3DMP.unit.depth(iUnit)));

align_idx = find(strcmp(alignEvent, 'saccade'));
if isempty(align_idx)
    align_idx = 1;
end

timeAxis = center_start(align_idx):binSize:center_stop(align_idx);
yAxis = timeAxis * 1000;
field_name = alignEvent{align_idx};
psth = dataAnals.(field_name)(:, iUnit);

time_window = yAxis >= -200 & yAxis <= 200;

y_lim = 0;

for cond = 1:3
    subplot(1, 3, cond);
    hold on;
    
    mean_fr = nan(1, 7);
    fr_choice_left = nan(1, 7);
    fr_choice_right = nan(1, 7);
    
    for log = 1:7
        idx_all = condition_indices{cond}(mod, coh) & headingInd==log & del'==0 & ...
            ~cellfun(@(x) any(isnan(x)), psth)';
        
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
    plot(heading_values(valid_mean), mean_fr(valid_mean), '-o', 'Color', colors{cond}, 'LineWidth', 2, ...
        'MarkerSize', 8, 'MarkerFaceColor', colors{cond});
    
    valid_left = ~isnan(fr_choice_left);
    plot(heading_values(valid_left), fr_choice_left(valid_left), '--^', 'Color', colors{cond}, 'LineWidth', 1.5, ...
        'MarkerSize', 8, 'MarkerFaceColor', colors{cond}, 'MarkerEdgeColor', colors{cond});
    
    valid_right = ~isnan(fr_choice_right);
    plot(heading_values(valid_right), fr_choice_right(valid_right), '--v', 'Color', colors{cond}, 'LineWidth', 1.5, ...
        'MarkerSize', 8, 'MarkerFaceColor', colors{cond}, 'MarkerEdgeColor', colors{cond});
    
    xlabel('Heading (deg)');
    ylabel('Firing Rate (spikes/s)');
    title(condition_labels{cond});
    xlim([min(heading_values) max(heading_values)]);
    grid on;
    
    current_max = max([mean_fr(valid_mean), fr_choice_left(valid_left), fr_choice_right(valid_right)]);
    if current_max > y_lim
        y_lim = current_max;
    end
    
    hold off;
end

for cond = 1:3
    subplot(1, 3, cond);
    ylim([0 y_lim*1.1]);
end

subplot(1, 3, 1);
lgd = legend({'Mean FR', 'Left Choice', 'Right Choice'}, 'Location', 'best');

if y_lim >= 1.5
    fig = gcf;
    [filepath, name, ext] = fileparts(eventInfo.output_pdf{6});
    output_file = fullfile(filepath, [name '_tuning' ext]);
    exportgraphics(fig, output_file, 'Append', true);
end
close(gcf);

end