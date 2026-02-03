function [dataStruct] = psthMatrix(dataStruct, timeInfo, var, unit_todo)

% Set timeInfo
offset = timeInfo.offset;
binSize = timeInfo.binSize;
alignEvent = timeInfo.alignEvent{var};
center_start = timeInfo.center_start(var);
center_stop = timeInfo.center_stop(var);
field_name = alignEvent;
timeAxis = center_start:binSize:center_stop;
num_raster = length(timeAxis);

% Apply stimulus onset offset if needed
if strcmpi(alignEvent, 'stimOn')
    timeAxis = timeAxis + 0.4250;
end

sigma = timeInfo.sigma;
num_datasets = size(dataStruct, 1);

for i = 1:num_datasets
    try
        % Define good trials
        test_q = dataStruct(i).data.dots3DMP.events.stimOff - dataStruct(i).data.dots3DMP.events.stimOn;
        goodtrial = test_q ~= 0 & ~isnan(test_q);

        num_unit = length(dataStruct(i).data.dots3DMP.unit.spiketimes);
        num_events = length(dataStruct(i).data.dots3DMP.events.stimOn);
        
        % Initialize spkrate cell array for all units (to maintain indexing)
        spkrate = cell(num_events, num_unit);
        
        % Fill all units with NaN first (for units not in unit_todo)
        for j = 1:num_unit
            for k = 1:num_events
                spkrate{k, j} = NaN(1, num_raster);
            end
        end

        % Only process units specified in unit_todo
        for j_idx = 1:length(unit_todo)
            j = unit_todo(j_idx); % Get the actual unit index
            
            if j > num_unit
                fprintf('Warning: Unit index %d exceeds available units (%d), skipping\n', j, num_unit);
                continue;
            end
            
            spike_time = dataStruct(i).data.dots3DMP.unit.spiketimes{1, j};
            
            for k = 1:num_events
                if goodtrial(k)
                    t_event = dataStruct(i).data.dots3DMP.events.(alignEvent)(k);
                    t_points = t_event + timeAxis;
                    
                    % Calculate causal firing rate
                    if sigma == 0
                        % Method 1: Causal binning
                        spkrate_sliding = calculateCausalBinnedRate(spike_time, t_points, offset);
                        
                        % Optional very light smoothing to reduce noise
                        smooth_window = 5; % Minimal smoothing
                        spkrate_sliding = smoothdata(spkrate_sliding, 'movmean', smooth_window);
                        
                    else
                        % Method 2: Causal Gaussian kernel
                        spkrate_sliding = calculateCausalGaussianRate(spike_time, t_points, sigma);
                    end

                    spkrate{k, j} = spkrate_sliding;
                else
                    spkrate{k, j} = NaN(1, num_raster);
                end
            end
            fprintf('Regular Unit %d/%d (index %d) done\n', j_idx, length(unit_todo), j);
        end

        % Assign spkrate to dataStruct
        if any(goodtrial)
            dataStruct(i).data.dots3DMP.data_spkrate.(field_name) = spkrate;
        else
            fprintf('Dataset %d: all trials are invalid\n', i);
        end
        
        fprintf('Dataset %d, processed %d/%d units for %s alignment\n', i, length(unit_todo), num_unit, field_name);
        
    catch ME
        fprintf('Dataset %d failed: %s\n', i, ME.message);
    end
end
end

function rate = calculateCausalBinnedRate(spike_times, time_points, offset)
    % Causal binning - only count spikes from the past
    rate = zeros(1, length(time_points));
    
    for i = 1:length(time_points)
        t_center = time_points(i);
        t_start = t_center - offset;  % Look back 'offset' seconds
        t_end = t_center;             % Up to current time (causal)
        
        spike_count = sum(spike_times > t_start & spike_times <= t_end);
        rate(i) = spike_count / offset; % Convert to Hz (offset already in seconds)
    end
end

function rate = calculateCausalGaussianRate(spike_times, time_points, sigma)
    % Causal Gaussian kernel - only consider past spikes
    rate = zeros(1, length(time_points));
    
    for i = 1:length(time_points)
        t_center = time_points(i);
        
        % Only consider spikes that happened before or at current time (causal)
        past_spikes = spike_times(spike_times <= t_center);
        
        if ~isempty(past_spikes)
            % Time differences (how long ago each spike occurred)
            time_diffs = t_center - past_spikes; % All positive for past spikes
            
            % Gaussian weights (but only for past spikes)
            weights = exp(-(time_diffs.^2) / (2 * sigma^2));
            
            % Sum weighted spikes and normalize
            rate(i) = sum(weights) / (sigma/1000 * sqrt(2*pi));
        else
            rate(i) = 0;
        end
    end
end