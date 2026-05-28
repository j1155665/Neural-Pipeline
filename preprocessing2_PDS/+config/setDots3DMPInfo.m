function [unitInfo, eventInfo, tuneventInfo, timeInfo, tuntimeInfo] = setDots3DMPInfo(Info)
%SETDOTS3DMPINFO Configure dots3DMP task parameters for specified subject
%
% Inputs:
%   Info - struct containing session info (for file paths)
%
% Outputs:
%   unitInfo, eventInfo, tuneventInfo, timeInfo, tuntimeInfo
switch Info.subject
    case 'zarya'
        %% === Unit Info ===
        unitInfo.unit_profile = 'good';  % 'good', 'all', etc.
        
        switch Info.session_date
            case '20250523'
                unitInfo.area_map = struct('MST', [0, 4000], 'VPS', [4001, 8000], 'MT', [], 'dual', [0, 8000]);
            case '20250602'
                unitInfo.area_map = struct('MST', [0, 3500], 'VPS', [3501, 8000], 'MT', [], 'dual', [0, 8000]);
            case '20250702'
                unitInfo.area_map = struct('MST', [1300, 7000], 'VPS', [7000, 8000], 'MT', [0, 1300], 'dual', [0, 8000]);
            case '20250710'
                unitInfo.area_map = struct('MST', [0, 4000], 'VPS', [4001, 8000], 'MT', [], 'dual', [0, 8000]);
            case '20250501'
                unitInfo.area_map = struct('MST', [0, 3500], 'VPS', [3501, 8000], 'MT', [], 'dual', [0, 8000]);
            case '20250417'
                unitInfo.area_map = struct('MST', [1501, 7000], 'VPS', [7001, 10000], 'MT', [0, 1500], 'dual', [0, 10000]);
            case '20250306'
                unitInfo.area_map = struct('MST', [1001, 5500], 'VPS', [5501, 10000], 'MT', [0, 1000], 'dual', [0, 10000]);
            case '20250411'
                unitInfo.area_map = struct('MST', [0, 3000], 'VPS', [3001, 8000], 'MT', [], 'dual', [0, 8000]);
            case '20260417'
                 unitInfo.area_map = nan;
            otherwise
                error('No area map found for session date: %s', session_date_str);
        end
        % Note: unitInfo.plot_indices will be set after loading data

        %% === Event Info (Main Analysis) ===
        % % Modality,  1 = vestibular, 2 = visual, 3 = combined
        % mod = {'vestibular';'visual'; 'combined'};
        % % Coherence, 1 = 0.2, 2 = 0.7
        % coh = [0.2 0.7];
        % % Heading angle, 1-9, +- 12, 6, 3, 1.5, 0
        % hedang = [-12 -6 -3 -1.5 0 1.5 3 6 12];
        % %stimulus motion
        % stim = {'left'; 'right'};
        % % Choice, 1 = left, 2 = right
        % Choice = {'left'; 'right'};
        % % Confidence, PDW 1 = high bet, 0 = low bet
        % Conf = {'low bet'; 'high bet'};
        eventInfo.name = {'modality';'coherenceInd';'headingInd';'choice';'PDW'};
        eventInfo.class_1 = {[]; []; [1 2 3]; 1; 0};
        eventInfo.class_2 = {[]; []; [5 6 7]; 2; 1};
        eventInfo.output_pdf = { [], [], [],[],[]...
            fullfile(Info.results_dir, [Info.session_date '_dots3DMP_' unitInfo.unit_profile '_headingplots.pdf']), ...
            fullfile(Info.results_dir, [Info.session_date '_dots3DMP_' unitInfo.unit_profile '_choiceandpdwplots.pdf'])};

        %% === Tuning Event Info ===
        tuneventInfo.name = {'modality';'coherenceInd';'headingInd'};
        tuneventInfo.heading = [-45, -21.2, -10, -3.9, 3.9, 10, 21.2, 45];
        tuneventInfo.class_1 = {[]; []; [1 2 3 4]};
        tuneventInfo.class_2 = {[]; []; [5 6 7 8]};
        tuneventInfo.output_pdf = fullfile(Info.results_dir, [Info.session_date '_dots3DMPtunning_' unitInfo.unit_profile '_frplots.pdf']);

        %% === Time Info (Main Analysis) ===
        timeInfo.offset = 0.025;
        timeInfo.binSize = 0.01;
        timeInfo.alignEvent = {'stimOn','saccOnset','postTargHold'};
        timeInfo.plotname = {'Stim On','Choice', 'PDW'};
        timeInfo.center_start = [-0.1, -0.6, -0.6];
        timeInfo.center_stop = [0.9, 0.4, 0.4];
        timeInfo.sigma = 0;
        timeInfo.velprofiledt = 0.0083;

        %% === Time Info (Tuning Analysis) ===
        tuntimeInfo.offset = 0.025;
        tuntimeInfo.binSize = 0.01;
        tuntimeInfo.alignEvent = {'stimOn'};
        tuntimeInfo.plotname = {'Stim On'};
        tuntimeInfo.sigma = 0;
        tuntimeInfo.velprofiledt = 0.0083;
        tuntimeInfo.trial_start = -0.2;
        tuntimeInfo.trial_stop = 0.2;
end
fprintf('Loaded dots3DMP configuration for subject: %s\n', Info.subject);


end