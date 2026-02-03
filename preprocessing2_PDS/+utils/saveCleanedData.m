function Info = saveCleanedData(Info)
% saveCleanedData - Save updated dataStruct with PSTH information
%
% Input:
%   Info - structure containing all data and file paths

if Info.savedata
    
    if isempty(Info.session_idx)
        error('Session with date %s not found in dataStruct', Info.formatted_date);
    else
        fprintf('Found session at index %d for date %s\n', Info.session_idx, Info.formatted_date);
    end
    
    % Update the specific session with processed data
    Info.dataStruct(Info.session_idx).data = Info.dataStruct_session.data;
    
    % Save session-specific data
    data = Info.dataStruct_session.data; % Save the session data structure
    fprintf('Saving updated session dataStruct to: %s\n', Info.dataFile);
    save(Info.dataFile, 'data', '-v7.3');
    
    % Save full dataStruct only when processing the last session
    if Info.s == length(Info.session_dates)
        dataStruct = Info.dataStruct; % Now save the full dataStruct array
        fprintf('Saving complete dataStruct to: %s\n', Info.saveFile);
        save(Info.saveFile, 'dataStruct', '-v7.3');
        fprintf('Complete dataset saved with %d sessions!\n', length(dataStruct));
    else
        fprintf('Session %d/%d processed. Not saving complete dataset yet.\n', ...
            Info.s, length(Info.session_dates));
    end
    
    fprintf('Data successfully saved!\n');
    
else
    fprintf('Save option disabled. Skipping data save.\n');
end

end