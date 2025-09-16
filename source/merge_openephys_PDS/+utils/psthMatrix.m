function [dataStruct] = psthMatrix(dataStruct, timeInfo, var)

% set timeInfo
offset = timeInfo.offset;
binSize = timeInfo.binSize;
alignEvent = timeInfo.alignEvent{var};
center_start = timeInfo.center_start(var);
center_stop = timeInfo.center_stop(var);
field_name = (alignEvent);
timeAxis = center_start:binSize:center_stop;
num_raster = length(timeAxis);
if strcmpi(alignEvent, 'stimOn')
    timeAxis = timeAxis + 0.4250;
end

sigma = timeInfo.sigma;

num_datasets = size(dataStruct, 1);

for i = 1:num_datasets
    try
        % Define good trial
        test_q = dataStruct(i).data.dots3DMP.events.stimOff - dataStruct(i).data.dots3DMP.events.stimOn;
        goodtrial = test_q ~= 0 & ~isnan(test_q);

        num_unit = length(dataStruct(i).data.dots3DMP.unit.spiketimes);
        num_events = length(dataStruct(i).data.dots3DMP.events.stimOn);
        spkrate = cell(num_events, num_unit);

        for j = 1:num_unit
            spike_time = dataStruct(i).data.dots3DMP.unit.spiketimes{1, j};
            
            for k = 1:num_events
                if goodtrial(k)
                    t_event = dataStruct(i).data.dots3DMP.events.(alignEvent)(k);
                    t_points = t_event + timeAxis;
                    spkrate_sliding = zeros(1, length(t_points));

                    if sigma == 0
                        % Rectangular window - only look backward
                        for sld = 1:num_raster 
                            t_current = t_points(sld);
                            t_start = t_current - 2 * offset;
                            t_stop = t_current;
                            spike_count = sum(spike_time > t_start & spike_time <= t_stop);
                            spkrate_sliding(1, sld) = spike_count / (2 * offset);
                        end
                        spkrate_sliding = smoothdata(spkrate_sliding, 'gaussian', 4);
                        
                    else
                        % Exponential decay kernel - causal
                        tau = sigma; % Use sigma as time constant
                        for sld = 1:num_raster
                            t_current = t_points(sld);
                            % Only consider spikes that happened BEFORE or AT current time
                            past_spikes = spike_time(spike_time <= t_current);
                            % Limit to reasonable window (5*tau back in time)
                            past_spikes = past_spikes(past_spikes >= (t_current - 5*tau));
                            
                            if ~isempty(past_spikes)
                                % Time differences (how long ago each spike occurred)
                                time_diffs = t_current - past_spikes; % All positive for past spikes
                                % Exponential weights (recent spikes weighted more)
                                weights = exp(-time_diffs / tau);
                                % Sum weighted spikes and normalize to get rate in Hz
                                spkrate_sliding(sld) = sum(weights) / tau;
                            else
                                spkrate_sliding(sld) = 0;
                            end
                        end
                    end

                    spkrate{k, j} = spkrate_sliding;
                else
                    spkrate{k, j} = NaN(1, num_raster);
                end
            end
            fprintf('unit %d / %d done\n', j, num_unit);
        end

        % Assign spkrate to dataStruct
        if any(goodtrial)
            dataStruct(i).data.dots3DMP.data_spkrate.(field_name) = spkrate;
        else
            fprintf('Dataset %d: all trials are invalid\n', i);
        end
        
    catch ME
        fprintf('Dataset %d failed: %s\n', i, ME.message);
    end
end

end