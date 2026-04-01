%% Main PSTH Analysis Script
% Run from: D:\Neural-Pipeline\source\merge_openephys_PDS

% clear; clc; close all;

%% === Basic Configuration ===
Info.subject = 'zarya';
Info.session_dates = {'20250602', '20250702', '20250523',  '20250501', '20250417', '20250710', '20250411', '20250306'};
close all
Info.session_dates = {'20250702'};
Info.task_name = 'dots3DMP';
Info.computpsth = 0; % do you want to compute psth?
Info.mergepds = 0; % pool pds data in
Info.savedata = 0;
Info.reloadcleaneddata = 0;
Info.plotpsth = 0;
Info.plotpsth_correct =0 ;
Info.plot_tuning_choice = 1;

for s = 1:length(Info.session_dates)
    Info.session_date = Info.session_dates{s};
    Info.s = s;
    try
        %% === Setup and Load ===
        Info = config.setPathInfo(Info);
        [unitInfo, eventInfo, tuneventInfo, timeInfo, tuntimeInfo] = config.setDots3DMPInfo(Info);

        %% === Compute PSTH ===

        if Info.computpsth
            unitInfo.plot_indices = 1:length(Info.dataStruct_session.data.dots3DMP.unit.groups);
            fprintf('Computing PSTH for %d alignment events...\n', length(timeInfo.alignEvent));
            for i = 1:length(timeInfo.alignEvent)
                Info.dataStruct_session = utils.psthMatrix(Info.dataStruct_session, timeInfo, i, unitInfo.plot_indices);
            end

            [Info.dataStruct_session, tuntimeInfo] = utils.psthMatrixtunning(Info.dataStruct_session, tuntimeInfo, unitInfo.plot_indices);
        end

        %% === Merge PDS file ===

        if Info.mergepds

            Info = utils.mergePdsStruct(Info);
            Info = utils.finalizing_tuning(Info);


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

        if Info.plotpsth_correct
            unitInfo.plot_indices = utils.plotunit(Info, unitInfo.unit_profile);
            if isempty(Info.dataStruct(Info.session_idx).data)
                fprintf('ERROR: dataStruct(%d).data is empty!\n', Info.session_idx);
                fprintf('Please compute PSTH first before generating plots.\n');
                continue;
            end

            fprintf('Generating main analysis plots for %d units...\n', length(unitInfo.plot_indices));
            for iUnit = 1:length(unitInfo.plot_indices)
                unit2plot = unitInfo.plot_indices(iUnit);
                for var = 6  % e.g. 6 = headings, 7 = l/r, h/l
                    utils.plotspkrate_correct(Info.dataStruct, timeInfo, var, eventInfo, Info.session_idx, unit2plot);
                end
            end
            close all;
            disp('dots3DMP plots saved.');
            fprintf('Analysis complete! Results saved to: %s\n', Info.results_dir);
        end

        if Info.plot_tuning_choice
            unitInfo.plot_indices = utils.plotunit(Info, unitInfo.unit_profile);
            for iUnit = 1:length(unitInfo.plot_indices)
                unit2plot = unitInfo.plot_indices(iUnit);
                utils.plot_tuning_choice(Info.dataStruct, timeInfo, eventInfo, Info.session_idx, unit2plot);
            end
        end
    catch ME
        fprintf('ERROR in session %s: %s\n', Info.session_date, ME.message);
        fprintf('Error occurred in: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
        fprintf('Continuing to next session...\n');
        continue;
    end


end
%%
% %% Appnedix plot example units from single session
% close all;
% session = 21;
% unit_id = 381;
% utils.plotexampletuning(Info.dataStruct(session).data, tuntimeInfo, tuneventInfo, unit_id);
% utils.plotexample(Info.dataStruct(session).data, timeInfo, eventInfo, unit_id, 6);
% utils.plottuningcurve(Info.dataStruct(session).data, tuntimeInfo, tuneventInfo, unit_id, [800, 1800]);
% % %%
% close all
% utils.plotexample(Info.dataStruct(23).data, timeInfo, eventInfo, 707, 4);
