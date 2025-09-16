function Info = mergePdsStruct(Info)
% mergePdsStruct - Create PDS structure for neuralpixels offline process
% Merges neural data with behavioral data from PLDAS preprocessing
%
% Requirements:
% 1. cleaned behavior data from PLDAPS_preprocessing_nolocal.mat
% 2. cleaned neural data from OpenEphysOfflineProcessing.ipynb
%
% Input:
%   Info - structure containing session information and paths
%
% Output:
%   Info - updated structure with merged PDS data

%% Define parameters
param = 'dots3DMP'; % 'dots3DMP'
date_str = Info.formatted_date; 

%% Find and load behavior data
fprintf('Searching for behavior files in: %s\n', Info.behavior_dir);

% Look for behavior files in the behavior directory
behavior_files = dir(fullfile(Info.behavior_dir, [Info.subject '_*.mat']));

Info.behaviorFile = '';
for i = 1:length(behavior_files)
    filename = behavior_files(i).name;
    % Extract date range from filename (e.g., zarya_20240101-20240131.mat)
    date_pattern = [Info.subject '_(\d{8})-(\d{8})\.mat'];
    tokens = regexp(filename, date_pattern, 'tokens');
    
    if ~isempty(tokens)
        start_date = str2double(tokens{1}{1});
        end_date = str2double(tokens{1}{2});
        session_date_num = str2double(Info.session_date);
        
        % Check if session date falls within the range
        if session_date_num >= start_date && session_date_num <= end_date
            Info.behaviorFile = fullfile(Info.behavior_dir, filename);
            fprintf('Found behavior file: %s\n', filename);
            break;
        end
    end
end

% Check if behavior file was found
if isempty(Info.behaviorFile)
    error('No behavior file found for session date: %s\nAvailable files: %s', ...
        Info.session_date, strjoin({behavior_files.name}, ', '));
end

% Load behavior data
fprintf('Loading behavior data from: %s\n', Info.behaviorFile);
behavior_data = load(Info.behaviorFile);
data = behavior_data.data; % Extract behavior data

%% Get neural data source
if Info.computpsth && ~isempty(Info.dataStruct_session)
    % Use session-specific data if computing PSTH
    targetData = Info.dataStruct_session.data.(param);
    fprintf('Using session-specific neural data\n');
else
    % Use data from main dataStruct
    if isempty(Info.session_idx)
        error('Session index not found for date: %s', date_str);
    end
    targetData = Info.dataStruct(Info.session_idx).data.(param);
    fprintf('Using neural data from session index: %d\n', Info.session_idx);
end

%% Convert date for matching
dt = datetime(date_str, 'InputFormat', 'yyyy-MM-dd');
date = str2double(datestr(dt, 'yyyymmdd'));

%% Check if blocks match
valid_block = unique(targetData.events.block);
block_idx = cell(1, length(valid_block));
pds_idx = cell(1, length(valid_block));

fprintf('Checking block matches:\n');
for vb = 1:length(valid_block)
    block_val = valid_block(vb);
    block_idx{vb} = (targetData.events.block) == block_val & (targetData.events.goodtrial == 1);
    pds_idx{vb} = (data.date == date) & (data.blockNum == block_val);
    fprintf('Block %d: block_idx = %d, pds_idx = %d  ', ...
        block_val, sum(block_idx{vb}), sum(pds_idx{vb}));
    if sum(block_idx{vb}) ~= sum(pds_idx{vb})
        fprintf('MISMATCH!\n');
        warning('Block %d has mismatched trial counts', block_val);
    else
        fprintf('\n');
    end
end

%% Create PDS info in dataStruct
pds_name = {'unique_trial_number', 'iTrial', 'blockNum'};
events_name = {'oneTargChoice', 'oneTargConf', 'heading', 'coherence', 'delta', 'RT'};
nTrials = length(targetData.events.goodtrial);

% Initialize PDS fields
for p = 1:length(pds_name)
    if iscell(data.(pds_name{p}))
        targetData.pds.(pds_name{p}) = cell(size(data.(pds_name{p})));
    else
        targetData.pds.(pds_name{p}) = nan(size(data.(pds_name{p})));
    end
end

% Initialize events fields
for e = 1:length(events_name)
    targetData.events.(events_name{e}) = nan(nTrials, 1);
end

% Copy data block by block
fprintf('Copying PDS data...\n');
for vb = 1:length(valid_block)
    for p = 1:length(pds_name)
        sz = size(data.(pds_name{p}));
        if sz(1) == length(pds_idx{1})
            targetData.pds.(pds_name{p})(block_idx{vb}, :) = data.(pds_name{p})(pds_idx{vb}, :);
        elseif sz(2) == length(pds_idx{1})
            targetData.pds.(pds_name{p})(:, block_idx{vb}) = data.(pds_name{p})(:, pds_idx{vb});
        else
            targetData.pds.(pds_name{p})(block_idx{vb}) = data.(pds_name{p})(pds_idx{vb});
        end
        targetData.pds.parName(block_idx{vb}) = data.parName(pds_idx{vb});
    end
    for e = 1:length(events_name)
        targetData.events.(events_name{e})(block_idx{vb}) = data.(events_name{e})(pds_idx{vb});
    end
end

%% Validate heading indices
fprintf('Validating heading indices...\n');
isSame = false(1, length(valid_block));
for vb = 1:length(valid_block)
    headingVals = targetData.events.heading(block_idx{vb}); 
    [~, ~, headingIndCheck] = unique(headingVals);
    
    storedHeadingInd = targetData.events.headingInd(block_idx{vb});
    isSame(vb) = all(headingIndCheck == storedHeadingInd');
    if ~isSame(vb)
        fprintf('Block %d: headingInd does NOT match.\n', vb);
        mismatchIdx = find(headingIndCheck ~= storedHeadingInd);
        disp(table(mismatchIdx, headingVals(mismatchIdx), ...
            headingIndCheck(mismatchIdx), storedHeadingInd(mismatchIdx), ...
            'VariableNames', {'Trial', 'Heading', 'CalcInd', 'StoredInd'}))
    end
end

%% Update dataStruct and save
if all(isSame)
    fprintf('All heading indices match, updating dataStruct\n');
    
    % Update the appropriate data structure
    if Info.computpsth && ~isempty(Info.dataStruct_session)
        Info.dataStruct_session.data.(param) = targetData;
        % Also update main dataStruct
        if ~isempty(Info.dataStruct(Info.session_idx).data)
            Info.dataStruct(Info.session_idx).data.(param) = targetData;
        end
    else
        Info.dataStruct(Info.session_idx).data.(param) = targetData;
    end
    
else
    error('Heading index validation failed. Cannot proceed with data merge.');
end

fprintf('PDS structure merge completed.\n');

end