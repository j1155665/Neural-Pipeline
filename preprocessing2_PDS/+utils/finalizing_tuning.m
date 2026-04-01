function Info = finalizing_tuning(Info)

data = Info.dataStruct_session.data;

headingInd = data.dots3DMPtuning.events.headingInd;
coherenceInd = data.dots3DMPtuning.events.coherenceInd;
goodtrial = data.dots3DMPtuning.events.goodtrial;
deltaInd = data.dots3DMPtuning.events.deltaInd;

% Define mapping for headingInd to real values
if strcmp(Info.dataStruct(Info.session_idx).date, '2025-03-06')
    heading_map = [-90, -45, -21.5, -10, -3.9, -1.5, 0, 1.5, 3.9, 10, 21.5, 45, 90];
else
    heading_map = [-45, -21.5, -10, -3.9, 3.9, 10, 21.5, 45];
end

% Define mapping for coherenceInd to real values  
coherence_map = [0.7]; % Only one coherence level

% Define mapping for deltaInd to real values
delta_map = [0]; % Only one delta level

% Initialize output arrays with NaN
n_trials = length(headingInd);
heading = nan(n_trials, 1);
coherence = nan(n_trials, 1);
delta = nan(n_trials, 1);

% Convert indices to real values only for good trials
for i = 1:n_trials
    if goodtrial(i) == 1  % Only process good trials
        
        % Convert headingInd to real heading value
        if headingInd(i) >= 1 && headingInd(i) <= length(heading_map)
            heading(i) = heading_map(headingInd(i));
        end
        
        % Convert coherenceInd to real coherence value
        if coherenceInd(i) >= 1 && coherenceInd(i) <= length(coherence_map)
            coherence(i) = coherence_map(coherenceInd(i));
        end
        
        % Convert deltaInd to real delta value
        if deltaInd(i) >= 1 && deltaInd(i) <= length(delta_map)
            delta(i) = delta_map(deltaInd(i));
        end
        
    end
    % If goodtrial(i) ~= 1, values remain NaN (already initialized)
end

% Save to data structure
data.dots3DMPtuning.events.heading = heading';
data.dots3DMPtuning.events.coherence = coherence';
data.dots3DMPtuning.events.delta = delta';

% Delete spike rate data from both structures
% if isfield(data.dots3DMP, 'data_spkrate')
%     data.dots3DMP = rmfield(data.dots3DMP, 'data_spkrate');
%     fprintf('Deleted data.dots3DMP.data_spkrate\n');
% else
%     fprintf('data.dots3DMP.data_spkrate not found\n');
% end
% 
% if isfield(data.dots3DMPtuning, 'data_spkrate')
%     data.dots3DMPtuning = rmfield(data.dots3DMPtuning, 'data_spkrate');
%     fprintf('Deleted data.dots3DMPtuning.data_spkrate\n');
% else
%     fprintf('data.dots3DMPtuning.data_spkrate not found\n');
% end
% 
% fprintf('Spike rate data cleanup complete.\n');

Info.dataStruct(Info.session_idx).data = data;
Info.dataStruct_session.data = data;
end