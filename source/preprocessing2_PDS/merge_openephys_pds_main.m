%% Main PSTH Analysis Script
% Run from: D:\Neural-Pipeline\source\merge_openephys_PDS

% clear; clc; close all;

%% === Basic Configuration ===
Info.subject = 'zarya';
Info.session_dates = {  '20250801'};
Info.task_name = 'dots3DMP';
Info.computpsth = 0; % do you want to compute psth?
Info.mergepds = 1; % pool pds data in
Info.savedata = 1;
Info.reloadcleaneddata = 0;
Info.plotpsth = 0;

for s = 1:length(Info.session_dates)
    Info.session_date = Info.session_dates{s};
    Info.s = s;
    try
        %% === Setup and Load ===
        Info = config.setPathInfo(Info);
        [unitInfo, eventInfo, tuneventInfo, timeInfo, tuntimeInfo] = config.setDots3DMPInfo(Info);

        %% === Compute PSTH ===
        
        if Info.computpsth

            fprintf('Computing PSTH for %d alignment events...\n', length(timeInfo.alignEvent));
            for i = 1:length(timeInfo.alignEvent)
                Info.dataStruct_session = utils.psthMatrix(Info.dataStruct_session, timeInfo, i);
            end
            %
            [Info.dataStruct_session, tuntimeInfo] = utils.psthMatrixtunning(Info.dataStruct_session, tuntimeInfo);
        end

        %% === Merge PDS file ===

        if Info.mergepds

            Info = utils.mergePdsStruct(Info);

        end

        %% === Save Cleaned Data ===

        if Info.savedata

            Info = utils.saveCleanedData(Info);
        end

        %% === Optional: Generate Plots ===

        if Info.plotpsth

            unitInfo.plot_indices = utils.plotunit(Info, unitInfo.unit_profile);
            if isempty(Info.dataStruct(Info.session_idx).data)
                fprintf('ERROR: dataStruct(%d).data is empty!\n', Info.session_idx);
                fprintf('Please compute PSTH first before generating plots.\n');
                continue;
            end

            % Tuning plots
            fprintf('Generating tuning plots for %d units...\n', length(unitInfo.plot_indices));
            for iUnit = 1:length(unitInfo.plot_indices)
                unit2plot = unitInfo.plot_indices(iUnit);
                utils.plotspkratetunning(Info.dataStruct, tuntimeInfo, tuneventInfo, Info.session_idx, unit2plot);
            end
            close all;
            disp('All tuning plots saved.');

            % Main analysis plots
            fprintf('Generating main analysis plots for %d units...\n', length(unitInfo.plot_indices));
            for iUnit = 1:length(unitInfo.plot_indices)
                unit2plot = unitInfo.plot_indices(iUnit);
                for var = 6:7  % e.g. 6 = headings, 7 = l/r, h/l
                    utils.plotspkrate(Info.dataStruct, timeInfo, var, eventInfo, Info.session_idx, unit2plot);
                end
            end
            close all;
            disp('dots3DMP plots saved.');
            fprintf('Analysis complete! Results saved to: %s\n', Info.results_dir);

        end
    catch ME
        fprintf('ERROR in session %s: %s\n', Info.session_date, ME.message);
        fprintf('Error occurred in: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
        fprintf('Continuing to next session...\n');
        continue;
    end


end
