% PLDAPS_preprocessing_nolocal, modified by YC, run on trellis computer

% ditch the overflow of rmfield calls in pdsCleanup
% https://undocumentedmatlab.com/articles/rmfield-performance

% today    = str2double(datestr(now,'yyyymmdd'));

subject = 'zarya';
paradigm = 'dots3DMP';
datestart = [20250801];
dateend = [20250831];
% where dataStruct should be saved
% localDir = '/Users/stevenjerjian/Desktop/FetschLab/PLDAPS_data/';
% localDir = uigetdir([], 'Choose directory to save data to');
localDir = '\\172.30.3.33\homes\fetschlab\labMembers\Yueh-Chen\zarya';

for i = 1:length(datestart)
    dateRange = datestart(i):dateend(i);


    opts = struct('eyemovement', 0, 'reward', 1, 'eventtiming', 0, 'useVPN', 0, 'useMount', 1);

    data = pds_preprocessing_dots3DMP(subject, paradigm, dateRange, localDir, [], opts);
    data = clean_data(data);

    if opts.eyemovement == 1
        filename = [subject '_' num2str(dateRange(1)) '-' num2str(dateRange(end)) '_wEM.mat'];
    else
        filename = [subject '_' num2str(dateRange(1)) '-' num2str(dateRange(end)) '.mat'];
    end


    save(fullfile(localDir, filename), 'data', '-v7.3');

end
function [data] = pds_preprocessing_dots3DMP(subject, paradigm, dateRange, localDir, data, options)

if nargin < 6, options = struct(); end
if nargin < 4 || isempty(localDir), localDir = uigetdir([], 'Choose directory to save data to'); end

use_existing_file = ~(nargin < 5 || isempty(data));

if ~isfield(options, 'eventtiming')
    options.eventtiming = options.eyemovement || options.nexonar;
end

opt_fields = {'nexonar', 'dotposition', 'eyemovement', 'reward', 'eventtiming', 'useSCP', 'useVPN'};
for f = 1:length(opt_fields)
    if ~isfield(options, opt_fields{f})
        options.(opt_fields{f}) = 0; % default = False
    end
end


if options.eyemovement
    % for aligning and cutting each datapixx trial to a 'time zero'
    align_event = 'timeGoCue';
    t_range = [-0.5, 1]; % seconds
    Fs = 1000;
    t_range = round(t_range*Fs);

    eyewin_samples = (t_range(1):t_range(2)-1);

end

if options.eventtiming
    pt_names = {'timeTargFixEntered', 'timeConfTargEntered', 'timeToConfidence'};

    st_names = {'timeFpEntered', 'timeTargetOn', 'timeTargetOff', ...
        'timeMotionStateBegin', 'timeMotionDone', 'timeGoCue', ...
        'TimeTargEntered', 'motionStartTime', 'timeBreakFix', 'timeComplete', ...
        'delayToGoCue', 'delayToDots', 'holdAfterDotsOnset', 'timeLastFrame'};
else
    pt_names = {};
    st_names = {};
end
bhv_names = {'choice', 'RT', 'PDW', 'correct', 'TargMissed', 'oneTargChoice', 'oneTargConf'};
rwd_names = {'amountRewardLowConfOffered', 'amountRewardHighConfOffered', 'fixRewarded', 'rewardDelay', 'rewardGiven'};



% set directories
[~, hostname] = system('hostname');
hostname = strip(hostname);

if ismac % someone's local Mac
    remoteDir = fullfile('/var/services/home/fetschlab/data/', subject);
    mountDir  = fullfile('/Volumes/homes/fetschlab/data/', subject);
elseif strcmp(hostname, 'DESKTOP-JRJ0F9N')
    remoteDir = fullfile('\\172.30.3.33\homes\fetschlab\data', subject);
    mountDir  = remoteDir;
end

% initialize data struct
if ~use_existing_file

    data.filename = {};
    data.subj = {};
    data.date = [];

    %         ntr_est = 3e5;
    %         data.filename = cell(ntr_est, 1);
    %         data.subj = cell(ntr_est, 1);
    %         data.date = nan(ntr_est, 1);
    %
    %         all_flds = unique([bhv_names, pt_names, st_names, {'iTrial', 'trialNum'}]);
    %
    %         for f = 1:length(all_flds)
    %             data.(all_flds{f}) = nan(ntr_est, 1);
    %
    %         end
    %         if options.eyemovement
    %             data.ADCdata = cell(ntr_est, 1);
    %             data.ADCtime = cell(ntr_est, 1);
    %         end

    T = 0;

else
    % use existing fields..NEED TO ADD CHECKS BELOW TO ONLY ADD NEW
    % DATA IF FIELD EXISTS
    all_flds = fieldnames(data);
    T = find(isnan(data.date), 1, 'first');
end


%% get data from NAS
% requires password(s) to be entered if a auth key not available
% https://www.howtogeek.com/66776/how-to-remotely-copy-files-over-ssh-without-entering-your-password/

% VERSION 3.0: 10-06-2023 SJ
% YC don't know why the password is wrong??? use mount dir for now

% get file list from mount dir
if ismac || options.useMount
    fprintf('get file list from mount dir...');
    remoteFiles = dir([mountDir, '/*.mat']); % ignore non .mat files (including old .PDS)
    fprintf('done\n');
    remoteFiles = {remoteFiles.name}';
    options.useSCP = 0; % force to 0
    options.useVPN = 0;

else
    % get file list from remote dir
    if ~options.useVPN
        ip_add = 'fetschlab@172.30.3.33'; % MBI
    else
        ip_add = 'fetschlab@10.161.240.133'; % probably off campus, try proxy IP (requires VPN)
    end
    cmd = ['ssh ' ip_add ' ls ' remoteDir];
    [~,remoteFileList] = system(cmd, "-echo");  % system(cmd, "-echo") % print output
    if any(strfind(remoteFileList,'timed out')); error(remoteFileList); end
    remoteFiles = splitlines(remoteFileList);
    remoteFiles = remoteFiles(contains(remoteFiles, subject) & ~contains(remoteFiles, '_'));
end
[~, remoteFileNames, ~] = cellfun(@fileparts, remoteFiles, 'UniformOutput', false);

dateStart = length(subject)+1;

for f = 1:length(remoteFiles)
    thisDate = str2double(remoteFiles{f}(dateStart:dateStart+7)); % yyyymmdd date format
    thisPar  = remoteFiles{f}(dateStart+8:length(remoteFileNames{f})-4);
    thisBlock = str2double(remoteFiles{f}(length(remoteFileNames{f})-3:length(remoteFileNames{f})));

    %         dateRange = [20240104 20240213 20240621];
    if ~any(strcmp(data.filename, remoteFileNames{f})) && any(dateRange == thisDate) && strcmp(thisPar, paradigm)
        % ok, we want this file
        try

            tstart = tic;
            fprintf('Loading file: %s\n', remoteFileNames{f})


            if options.useSCP
                % copy file to local machine
                % to save reduced version(?)
                cmd = ['scp -r ' ip_add ':' remoteDir remoteFiles{f} ' ' localDir];
                system(cmd, "-echo")
                load(fullfile(localDir, remoteFiles{f}), '-mat', 'PDS')

            else
                load(fullfile(mountDir, remoteFiles{f}), '-mat', 'PDS');
                fprintf('done\n');
            end

            % SJ added 02/2022, to generate 3DMP dots offline from trialSeeds, no
            % need to save online for storage space reasons
            if options.dotposition
                try
                    if ~isfield(PDS.data{1}.stimulus,'dotX_3D')
                        [dotX_3D,dotY_3D,dotZ_3D,dotSize] = generateDots3D_offline(PDS);
                    end
                catch
                    disp("offline dot generation did not work...")
                end
            end

            % extract desired fields for each trial, append to struct
            for t = 1:length(PDS.data)
                % excluding trials with missing data. When the blocks are
                % is stoped by qqqqq, the final trials won't have choices
                if isfield(PDS.data{t}.behavior,'choice') 

                    T = T+1; % increment trial counter

                    data.trialNum(T) = t;

                    data.filename{T} = remoteFileNames{f};
                    data.date(T) = thisDate;

                    if contains(subject,'human')
                        data.subj{T} = remoteFileNames{f}(dateStart(1)-3:dateStart(1)-1); % 3-letter code
                    else
                        data.subj{T} = subject;
                    end

                    % independent variables (conditions) are stored in PDS.conditions.stimulus
                    fnames = fieldnames(PDS.conditions{t}.stimulus);
                    for F = 1:length(fnames)
                        data.(fnames{F})(T) = PDS.conditions{t}.stimulus.(fnames{F});
                    end

                    % dependent variables (outcomes) stored in PDS.data.behavior
                    for F = 1:length(bhv_names)
                        data.(bhv_names{F})(T) = PDS.data{t}.behavior.(bhv_names{F});
                    end

                    % misc
                    try
                        data.iTrial(T) = PDS.conditions{t}.pldaps.iTrial;
                    catch
                        data.iTrial(T) = NaN;
                    end

                    data.unique_trial_number(T,:) = PDS.data{t}.unique_number;
                    data.parName{T} = thisPar;
                    data.blockNum(T) = thisBlock;


                    % options
                    if options.reward
                        fnames = rwd_names;
                        for F = 1:length(fnames)
                            try
                                data.(fnames{F})(T) = PDS.data{t}.reward.(fnames{F});
                            catch
                                data.(fnames{F})(T) = NaN;
                            end
                        end
                    end

                    if options.eventtiming
                        for F = 1:length(pt_names)
                            if isfield(PDS.data{t}.postTarget, pt_names{F})
                                data.(pt_names{F})(T) = PDS.data{t}.postTarget.(pt_names{F});
                            else
                                data.(pt_names{F})(T) = NaN;
                            end
                        end

                        for F = 1:length(st_names)
                            if isfield(PDS.data{t}.stimulus, st_names{F})
                                data.(st_names{F})(T) = PDS.data{t}.stimulus.(st_names{F});
                            else
                                data.(st_names{F})(T) = NaN;
                            end
                        end
                    end

                    if options.eyemovement
                        try
                            adc_data = PDS.data{t}.datapixx.adc.data;
                            dp_time = PDS.data{t}.datapixx.unique_trial_time(2); % datapixx 'start' time
                            adc_time = PDS.data{t}.datapixx.adc.dataSampleTimes - dp_time;

                            [~, zero_pos] = min(abs(adc_time - data.(align_event)(T))); % T not t!!

                            % find 'RT' in high-freq data
                            % TODO set threshold based on st dev
                            % search for consecutive values above threshold
                            %                                 xdata = adc_data(1, eyewin_samples+zero_pos);
                            %                                 xdata = abs(xdata-xdata(1));
                            %                                 zero_pos = find(xdata>0.1, 1) + zero_pos + eyewin_samples(1);

                            data.eyeX(1:length(eyewin_samples), T) = adc_data(1, eyewin_samples+zero_pos);
                            data.eyeY(1:length(eyewin_samples), T) = adc_data(2, eyewin_samples+zero_pos);

                        catch
                            data.eyeX(1:length(eyewin_samples), T) = NaN;
                            data.eyeY(1:length(eyewin_samples), T) = NaN;
                        end
                        data.eyeT(1:length(eyewin_samples), T) = eyewin_samples/Fs;

                    end


                    if options.dotposition
                        % TODO modify to store dotPos as data x trials array
                        try
                            if ~isfield(PDS.data{t}.stimulus,'dotX_3D')
                                PDS.data{t}.stimulus.dotX_3D = dotX_3D{t};
                                PDS.data{t}.stimulus.dotY_3D = dotY_3D{t};
                                PDS.data{t}.stimulus.dotZ_3D = dotZ_3D{t};
                                PDS.data{t}.stimulus.dotSize = dotSize{t};
                            end
                            data.dotX_3D{T} = PDS.data{t}.stimulus.dotX_3D;
                            data.dotY_3D{T} = PDS.data{t}.stimulus.dotY_3D;
                            data.dotZ_3D{T} = PDS.data{t}.stimulus.dotZ_3D;
                            data.dotSize{T} = PDS.data{t}.stimulus.dotSize;
                        catch
                        end
                    end

                    if options.nexonar
                        disp('not yet implemented')
                        % need to move over from original
                    end
                end
            end

            telapsed = toc(tstart);
            fprintf('numtrials = %d, time taken: %.2fs', length(PDS.data), telapsed)

        catch
            %             keyboard
            T = T-1;
            warning(['Processing issue, or could not load ' remoteFiles{f} '. File may be corrupt -- skipping']);

        end

        fprintf('\ncumulative trials = %d\n', T);
    end
end
disp('done')
end

function [data] = clean_data(data)

fnames = fieldnames(data);
goodtrial = ~isnan(data.RT);


for F = 1:length(fnames)
    try
        sz = size(data.(fnames{F}));
        if sz(1) == length(goodtrial) 
            data.(fnames{F}) = data.(fnames{F})(goodtrial, :);
        elseif sz(2) == length(goodtrial) 
            data.(fnames{F}) = data.(fnames{F})(:, goodtrial);
        else 
            data.(fnames{F}) = data.(fnames{F})(goodtrial);
        end
    catch
        fprintf("Field '%s' size doesn't match.\n", fnames{F});
    end
end
end



