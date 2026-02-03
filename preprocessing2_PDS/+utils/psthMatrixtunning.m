function [dataStruct, timeInfo] = psthMatrixtunning(dataStruct, timeInfo, unit_todo)

binSize = timeInfo.binSize;
offset = timeInfo.offset;
sigma = timeInfo.sigma;

session_num = size(dataStruct, 1);
timeInfo.trial_mean_time = zeros(session_num, 1);

for session = 1:session_num
    try
        % Define good trials
        events = dataStruct(session).data.dots3DMPtuning.events;
        unit = dataStruct(session).data.dots3DMPtuning.unit;
        test_q = events.stimOff - events.stimOn;
        badtrial = test_q < 2 | events.goodtrial == 0;

        trial_mean_time = mean(test_q(~badtrial));
        timeInfo.trial_mean_time(session) = trial_mean_time;

        num_unit = length(unit.spiketimes);
        num_events = length(events.stimOff);
        
        % Initialize data_spkrate cell array for all units (to maintain indexing)
        dataStruct(session).data.dots3DMPtuning.data_spkrate = cell(num_events, num_unit);

        % Fill all units with NaN first (for units not in unit_todo)
        for j = 1:num_unit
            for i = 1:num_events
                dataStruct(session).data.dots3DMPtuning.data_spkrate{i, j} = NaN;
            end
        end

        % Only process units specified in unit_todo
        for j_idx = 1:length(unit_todo)
            j = unit_todo(j_idx); % Get the actual unit index
            
            if j > num_unit
                fprintf('Warning: Unit index %d exceeds available units (%d), skipping\n', j, num_unit);
                continue;
            end
            
            spike_time = unit.spiketimes{j};

            for i = 1:num_events
                if ~badtrial(i)
                    event_start = events.stimOn(i) + timeInfo.trial_start;
                    event_stop = events.stimOn(i) + trial_mean_time + timeInfo.trial_stop;
                    t_center = event_start:binSize:event_stop;
                    
                    % Calculate causal firing rate
                    if sigma == 0
                        % Method 1: Causal binning
                        spkrate_sliding = calculateCausalBinnedRateTuning(spike_time, t_center, offset);
                        
                        smooth_window = 8; % Minimal smoothing
                        spkrate_sliding = smoothdata(spkrate_sliding, 'movmean', smooth_window);
                        
                    else
                        % Method 2: Causal Gaussian kernel
                        spkrate_sliding = calculateCausalGaussianRateTuning(spike_time, t_center, sigma);
                    end

                    dataStruct(session).data.dots3DMPtuning.data_spkrate{i, j} = spkrate_sliding;
                else
                    dataStruct(session).data.dots3DMPtuning.data_spkrate{i, j} = NaN;
                end
            end
            fprintf('Tuning Unit %d/%d (index %d) done\n', j_idx, length(unit_todo), j);
        end

        fprintf('Session %d, events %d, processed %d/%d units done\n', session, num_events, length(unit_todo), num_unit);
    catch ME
        fprintf('Error in session %d: %s\n', session, ME.message);
    end
end
end

function rate = calculateCausalBinnedRateTuning(spike_times, time_points, offset)
    % Causal binning - only count spikes from the past
    rate = zeros(1, length(time_points));
    
    for i = 1:length(time_points)
        t_current = time_points(i);
        t_start = t_current - offset;  % Look back 'offset' seconds
        t_end = t_current;             % Up to current time (causal)
        
        spike_count = sum(spike_times > t_start & spike_times <= t_end);
        rate(i) = spike_count / offset; % Convert to Hz (offset already in seconds)
    end
end

function rate = calculateCausalGaussianRateTuning(spike_times, time_points, sigma)
    % Causal Gaussian kernel - only consider past spikes
    rate = zeros(1, length(time_points));
    
    for i = 1:length(time_points)
        t_current = time_points(i);
        
        % Only consider spikes that happened before or at current time (causal)
        past_spikes = spike_times(spike_times <= t_current);
        
        if ~isempty(past_spikes)
            % Time differences (how long ago each spike occurred)
            time_diffs = t_current - past_spikes; % All positive for past spikes
            
            % Gaussian weights (but only for past spikes)
            weights = exp(-(time_diffs.^2) / (2 * sigma^2));
            
            % Sum weighted spikes and normalize
            rate(i) = sum(weights) / (sigma/1000 * sqrt(2*pi));
        else
            rate(i) = 0;
        end
    end
end