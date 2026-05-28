function expCond = create_expCond(event, goodtrial_idx)
    % Extract only good tria
    modality_good = event.modality(goodtrial_idx);
    heading_good = event.headingInd(goodtrial_idx);
    coh_good = event.coherenceInd(goodtrial_idx);
    
    % Get max heading from good trials only
    max_heading = max(heading_good);
    
    % Vectorized computation (faster, cleaner)
%     expCond = heading_good + (modality_good .* coh_good - 1) * max_heading;

%     expCond = modality_good;
%     expCond = ones(1, length(heading_good));
%     expCond = heading_good;
    expCond = heading_good + (modality_good -1 ) * max_heading;

    
    % Ensure column vector
    expCond = expCond(:);
    
    % Optional: verify
    fprintf('Created %d unique conditions from %d trials\n', ...
        length(unique(expCond)), length(expCond));
end

