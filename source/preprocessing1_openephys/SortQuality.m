resultsDirectory = "D:\20250710\kilosort4_phy";
clustertoprocess = 679;
[clusterIDs, unitQuality, contaminationRate, isiViolations , cluster_data] = sqKilosort.computeAllMeasures(resultsDirectory);

%%
% Get the group data
groups = cluster_data.group;

% Get firing rates and scale them for dot sizes
firing_rates = cluster_data.fr;
% Scale firing rates to reasonable dot sizes (e.g., 10 to 200)
min_size = 10;
max_size = 200;
scaled_sizes = min_size + (max_size - min_size) * (firing_rates - min(firing_rates)) / (max(firing_rates) - min(firing_rates));

% Define colors for each group
colors = containers.Map();
colors('good') = [1, 0, 0];    % red
colors('mua') = [0, 0, 1];     % blue
colors('noise') = [0, 1, 0];   % green
colors('') = [0.5, 0.5, 0.5];  % gray

% Create color array for plotting
plot_colors = zeros(length(groups), 3);
for i = 1:length(groups)
    group_name = groups{i};
    if isKey(colors, group_name)
        plot_colors(i, :) = colors(group_name);
    else
        plot_colors(i, :) = colors('');  % default to gray
    end
end

% Create single plot: uQ vs cR
clustertoplot = find(unitQuality<1000);
close all;
figure;
hold on;
scatter(contaminationRate(clustertoplot), unitQuality(clustertoplot), scaled_sizes(clustertoplot), plot_colors(clustertoplot,:), 'filled');
xlabel('Contamination Rate (cR)');
ylabel('Unit Quality (uQ)');
title('Unit Quality vs Contamination Rate');
grid on;

% Add legend
legend_handles = [];
legend_labels = {};
unique_groups = unique(groups);
for i = 1:length(unique_groups)
    group_name = unique_groups{i};
    if isempty(group_name)
        display_name = 'unclassified';
        color = colors('');
    else
        display_name = group_name;
        color = colors(group_name);
    end
%     h = scatter(NaN, NaN, 50, color, 'filled');
%     legend_handles = [legend_handles, h];
%     legend_labels{end+1} = display_name;
end
legend(legend_handles, legend_labels, 'Location', 'best');

% Add a colorbar to show firing rate scale
c = colorbar;
c.Label.String = 'Firing Rate (Hz)';
% Create a dummy scatter plot for the colorbar
dummy_scatter = scatter(NaN, NaN, scaled_sizes, firing_rates, 'filled');

% Adjust figure size
set(gcf, 'Position', [100, 100, 800, 600]);