function [dataStruct, timeInfo] = psthMatrixtunning(dataStruct, timeInfo)

binSize = timeInfo.binSize;
offset = timeInfo.offset;

% Gaussian kernal
sigma = timeInfo.sigma;
kernel_width = offset;

session_num = size(dataStruct, 1);
timeInfo.trial_mean_time = zeros(session_num);

for session = 1:session_num
    try
        % Define good trial
        events = dataStruct(session).data.dots3DMPtuning.events;
        unit = dataStruct(session).data.dots3DMPtuning.unit;
        test_q = events.stimOff - events.stimOn;
        badtrial = test_q < 2 | events.goodtrial == 0;

        trial_mean_time = mean(test_q(~badtrial)); %change it to good trial, mnot sure if it is good
        timeInfo.trial_mean_time(session) = trial_mean_time;

        num_unit = length(unit.spiketimes);
        num_events = length(events.stimOff);
        dataStruct(session).data.dots3DMPtuning.data_spkrate =cell(num_events, num_unit);

        for j = 1:num_unit
            spike_time = unit.spiketimes{j};

            for i = 1:num_events
                if ~badtrial(i)

                    event_start = (events.stimOn(i)) + timeInfo.trial_start;
                    event_stop = events.stimOn(i) + trial_mean_time + timeInfo.trial_stop;
                    t_center = event_start:binSize:event_stop;
                    
                    % Adjust time axis for stimOn alignment, comment out,
                    % only for dots3DMP
%                     t_center = t_center + 0.4250;
                    
                    t_start = t_center - offset * 2;
                    t_stop = t_center;
                    spkrate_sliding = zeros(1, length(t_center));

                    if sigma == 0

                        for sld = 1:length(t_center)

                            spkrate_sliding(1, sld) = sum(spike_time > t_start(sld) & spike_time < t_stop(sld)) / (2 * offset);
                        end
                        
                        spkrate_sliding = smoothdata(spkrate_sliding, 'gaussian', 4);

                    else

                        % Perform KDE to
                        for k = 1:length(t_center)
                            t = t_center(k);
                            valid_spikes = spike_time(abs(spike_time - t) <= kernel_width);
                            dists = t - valid_spikes;
                            weights = exp(-dists.^2 / (2 * sigma^2));
                            spkrate_sliding(k) = sum(weights) / (sigma * sqrt(2 * pi));

                        end

                    end


                    dataStruct(session).data.dots3DMPtuning.data_spkrate{i, j} = spkrate_sliding;

                else
                    dataStruct(session).data.dots3DMPtuning.data_spkrate{i, j} = nan;

                end
            end
            fprintf('unit %d / %d done\n', j, num_unit);
        end

        fprintf('session %d, event %d, unit %d done\n', session, num_events, num_unit);
    catch ME
        % Display the error message
        fprintf('Error in session %d: %s\n', session, ME.message);
    end
end

end