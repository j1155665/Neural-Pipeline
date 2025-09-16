% createPdsStruct, for neuralpixels offline process
% to run this you will need
% 1. cleaned behavior datas from PLDAS_preprocessing_nolocal.mat
% 2. cleaned neural datas from OpenEphysOfflineProcessing.ipynb
% 3. 


%% Define and Load data

neural_path = ['\\172.30.3.33\homes\fetschlab\labMembers\Yueh-Chen\zarya\Neural data\zarya_neurodata_cleaned.mat'];
behavior_path = ['/Volumes/homes/fetschlab/labMembers/Yueh-Chen/zarya' ...
    '/zarya_20250601-20250631.mat'];
param = 'dots3DMP';
date_str = '2025-06-02';

load(neural_path);
% load(behavior_path);

%% Check if the blocks matches

dt = datetime(date_str, 'InputFormat', 'yyyy-MM-dd');
date = str2double(datestr(dt, 'yyyymmdd'));

for i = 1:length(dataStruct)
    if dataStruct(i).date == date_str
        session = i;
        break
    end
end

targetData = dataStruct(session).data.(param);
valid_block = unique(targetData.events.block);
block_idx = cell(1, length(valid_block));
pds_idx = cell(1, length(valid_block));

for vb = 1:length(valid_block)

    block_val= valid_block(vb);
    block_idx{vb} = (targetData.events.block) == block_val & (targetData.events.goodtrial == 1);
    pds_idx{vb} = (data.date == date) & (data.blockNum == block_val);
    fprintf('Block %d: block_idx = %d, pds_idx = %d  ', ...
        block_val, sum(block_idx{vb}), sum(pds_idx{vb}));
    if sum(block_idx{vb}) ~= sum(pds_idx{vb})
        fprintf('MISMATCH!\n');
    else
        fprintf('\n');
    end

end

%% Create the pds info into the dataStruct

pds_name = {'unique_trial_number', 'iTrial', 'blockNum'};
events_name = {'oneTargChoice', 'oneTargConf', 'heading', 'coherence', 'delta', 'RT'};
nTrials = length(targetData.events.goodtrial);
for p = 1:length(pds_name)
    if iscell(data.(pds_name{p}))
        targetData.pds.(pds_name{p}) = cell(size(data.(pds_name{p})));
    else
        targetData.pds.(pds_name{p}) = nan(size(data.(pds_name{p})));
    end
end
for e = 1:length(events_name)
    targetData.events.(events_name{e}) = nan(nTrials, 1);
end

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

%% Check if the target data is what you want, and put it into the dataStruct

for vb = 1:length(valid_block)
    headingVals = targetData.events.heading(block_idx{vb}); 
    [~, ~, headingIndCheck] = unique(headingVals); % sorted by default

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
if all(isSame)
    fprintf('All headinEnd matches, copying data to dataStruct\n');
    dataStruct(session).data.(param) = targetData;
    userResponse = input('Do you want to save the updated dataStruct? (y/n): ', 's');
    if strcmpi(userResponse, 'y')
        save(neural_path, 'dataStruct');
        fprintf('Data saved successfully.\n');
    else
        fprintf('Save canceled by user.\n');
    end
end


