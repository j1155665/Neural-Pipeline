function plot_unit_idx = plotunit(Info, group)

% Extract unit information
% cluster_id = dataStruct.data.dots3DMP.unit.cluster_group.cluster_id;
info_index = Info.session_idx;
dataStruct = Info.dataStruct;
data = dataStruct(info_index).data;

if isempty(data) || ~isfield(data, 'dots3DMP') || isempty(data.dots3DMP)
    fprintf('Error: Data is empty for session %d. No dots3DMP data found.\n', info_index);
    plot_unit_idx = [];
    return;
end

cluster_group = data.dots3DMP.unit.cluster_group.group;
depth = data.dots3DMP.unit.cluster_group.depth;


if nargin < 2
    % If no group specified, include all clusters
    valid_idx = true(size(depth));
else
    % Convert to cell array if needed
    if ~iscell(cluster_group)
        cluster_group = cellstr(cluster_group);
    end
    % Filter by specified group (contains the word)
    % Special handling for 'noise' or 'nan' - include both if either is specified
    if strcmpi(group, 'noise') || strcmpi(group, 'nan')
        valid_idx = contains(cluster_group, 'noise', 'IgnoreCase', true) | ...
            contains(cluster_group, 'nan', 'IgnoreCase', true);
    else
        valid_idx = contains(cluster_group, group, 'IgnoreCase', true);
    end
end

plot_unit_idx = find(valid_idx);
selected_depths = depth(valid_idx);

% Sort by depth (smallest to largest)
[~, sort_idx] = sort(selected_depths);
plot_unit_idx = plot_unit_idx(sort_idx);

% Display information
if nargin < 2
    fprintf('Plotting all units, sorted by depth:\n');
else
    if strcmpi(group, 'noise') || strcmpi(group, 'nan')
        fprintf('Plotting units containing "noise" or "nan", sorted by depth:\n');
    else
        fprintf('Plotting units containing "%s", sorted by depth:\n', group);
    end
end

fprintf('Total units to plot: %d\n', length(plot_unit_idx));