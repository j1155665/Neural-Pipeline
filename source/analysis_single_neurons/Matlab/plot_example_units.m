%% add the info
load('D:\Neural-Pipeline\data\20250710\zarya20250710dots3DMP.mat');

headingInd = data.dots3DMPtuning.events.headingInd;
coherenceInd = data.dots3DMPtuning.events.coherenceInd;
goodtrial = data.dots3DMPtuning.events.goodtrial;
deltaInd = data.dots3DMPtuning.events.deltaInd;

% Define mapping for headingInd to real values
heading_map = [-45, -21.5, -10, -3.9, 3.9, 10, 21.5, 45];

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

%% plot spike rate
