function Info = setPathInfo(Info)
% setPathInfo - Configure file paths and load data for neural analysis
%
% Input:
%   Info - structure with basic configuration (subject, session_date, task_name, flags)
%
% Output:
%   Info - enhanced structure with all paths, loaded data, and session info

%% Construct file paths
Info.data_dir = 'D:\Neural-Pipeline\data';
Info.results_dir = 'D:\Neural-Pipeline\results';
Info.save_dir = fullfile('\\172.30.3.33\homes\fetschlab\labMembers\Yueh-Chen', Info.subject, 'Neural data');
Info.behavior_dir = fullfile('\\172.30.3.33\homes\fetschlab\labMembers\Yueh-Chen', Info.subject);
Info.dataFile = fullfile(Info.data_dir, Info.session_date, [Info.subject Info.session_date Info.task_name '.mat']);
Info.saveFile = fullfile(Info.save_dir, ['@' Info.subject '_neurodata_cleaned.mat']);

%% Ensure results directory exists
if ~exist(Info.results_dir, 'dir')
    mkdir(Info.results_dir);
end

%% Load data
if Info.computpsth
    Info.dataStruct_session = load(Info.dataFile); 
end

% Load cleaned neural data only on first iteration
if Info.s == 1 && Info.reloadcleaneddata
    temp = load(Info.saveFile);
    Info.dataStruct = temp.dataStruct; % Extract dataStruct from loaded file
    clear temp;
    fprintf('Loading cleaned dataStruct\n');
else
    fprintf('Cleaned dataStruct already loaded\n');
end

%% Format date and find matching session
Info.formatted_date = [Info.session_date(1:4) '-' Info.session_date(5:6) '-' Info.session_date(7:8)];

% Find the matching session by date
Info.session_idx = [];
for i = 1:length(Info.dataStruct)
    if strcmp(Info.dataStruct(i).date, Info.formatted_date)
        Info.session_idx = i;
        break;
    end
end

if isempty(Info.session_idx)
    warning('No matching session found for date: %s', Info.formatted_date);
end

end