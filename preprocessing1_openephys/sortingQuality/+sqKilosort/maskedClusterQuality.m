function [clusterIDs, unitQuality, contaminationRate] = maskedClusterQuality(resultsDirectory, clusterID)

fprintf(1, 'loading data...\n');
%% Precompute the locations of files to be loaded
pcFeaturesPath = fullfile(resultsDirectory,'pc_features.npy');
pcFeaturesIndPath = fullfile(resultsDirectory,'pc_feature_ind.npy');
spikeClustersPath = fullfile(resultsDirectory,'spike_clusters.npy');
spikeTemplatesPath = fullfile(resultsDirectory,'spike_templates.npy');

%% Main code.
try
    pc_features = readNPY(pcFeaturesPath);
catch me
    if ~exist(pcFeaturesPath, 'file')
        fprintf(1, 'PC Features loading failed. File does not exist.\n');
    else
        fprintf(1, 'PC Features loading failed. You may need to clone the npy-matlab repo and add to path.\n');
    end
    rethrow(me)
end
pc_feature_ind = readNPY(pcFeaturesIndPath);

if exist(spikeClustersPath,'file')
    fprintf('building features matrix from clusters/templates\n')
    spike_clusters = readNPY(spikeClustersPath);
    spike_templates = readNPY(spikeTemplatesPath);
    
    availableClusterIDs = unique(spike_clusters);
    
    % Determine which clusters to process, but keep ALL spike data
    if nargin > 1 && ~isempty(clusterID)
        requestedClusterIDs = clusterID(:)';
        clustersToProcess = intersect(requestedClusterIDs, availableClusterIDs);
        missingClusters = setdiff(requestedClusterIDs, availableClusterIDs);
        
        if ~isempty(missingClusters)
            fprintf('Warning: The following cluster IDs were not found in data: %s\n', mat2str(missingClusters));
        end
        
        if isempty(clustersToProcess)
            error('None of the specified cluster IDs exist in the data. Available clusters: %s', mat2str(availableClusterIDs));
        end
        
        fprintf('Processing %d valid clusters: %s\n', length(clustersToProcess), mat2str(clustersToProcess));
        
        % DON'T filter the spike data - keep everything for comparison
        % keepSpikes = ismember(spike_clusters, clustersToProcess);
        % spike_clusters = spike_clusters(keepSpikes);
        % spike_templates = spike_templates(keepSpikes);
        % pc_features = pc_features(keepSpikes, :, :);
        
    else
        clustersToProcess = availableClusterIDs;
    end
    
    % Process ALL clusters for feature matrix building (needed for comparisons)
    clusterIDs = availableClusterIDs;  % Use all clusters, not just requested ones
    nClusters = length(clusterIDs);
    nSpikes = length(spike_clusters);
    nFet = 4; nFetPerChan = size(pc_features,2);
    nTemplates = size(pc_feature_ind,1);
    
    newFet = zeros(nSpikes, nFetPerChan, nFet);
    newFetInds = zeros(nClusters, nFet);
    
    uniqueClusters = unique(spike_clusters);
    for c = 1:length(uniqueClusters)
        thisID = uniqueClusters(c);
%         if nargin > 1 && ~isempty(clusterID) && ~ismember(thisID, clustersToProcess)
%             continue;
%         end
       
        theseSpikes = spike_clusters==thisID;
        theseTemplates = spike_templates(theseSpikes);
        [inclTemps, inst] = countUnique(theseTemplates); 
        
        thisTemplate = inclTemps(inst==max(inst),1);
        
        theseChans = pc_feature_ind(thisTemplate+1,1:nFet);
        
        newFetInds(c,:) = theseChans;
        
        for f = 1:nFet
            thisChanInds = pc_feature_ind==theseChans(f);
            [chanInds,tempsWithThisChan] = find(thisChanInds');
                        
            inclTempsWithThisFet = find(ismember(inclTemps, tempsWithThisChan));
            for t = 1:numel(inclTempsWithThisFet)
                thisSubTemp = inclTemps(inclTempsWithThisFet(t));
                thisTfetInd = chanInds(tempsWithThisChan==thisSubTemp);
                newFet(theseSpikes&spike_templates==thisSubTemp,:,f) = ...
                    pc_features(theseSpikes&spike_templates==thisSubTemp,:,thisTfetInd);
            end
        end
    end
    
    pc_features = newFet;
    pc_feature_ind = newFetInds;
else
    fprintf(1, 'warning, spike_clusters does not exist, using spike_templates instead\n');
    spike_clusters = readNPY(spikeTemplatesPath);
    
    availableClusterIDs = unique(spike_clusters);
    if nargin > 1 && ~isempty(clusterID)
        requestedClusterIDs = clusterID(:)';
        clustersToProcess = intersect(requestedClusterIDs, availableClusterIDs);
        
        if isempty(clustersToProcess)
            error('None of the specified cluster IDs exist in the data. Available clusters: %s', mat2str(availableClusterIDs));
        end
        
        % DON'T filter spike data here either
    else
        clustersToProcess = availableClusterIDs;
    end
end

assert(numel(size(pc_features)) == 3)

fprintf(1, 'computing cluster qualities...\n');
% Pass which clusters to actually compute metrics for
pc_Nchans = min(4, size(pc_feature_ind,2));
if nargin > 1 && ~isempty(clusterID)
    
   [clusterIDs, unitQuality, contaminationRate] = maskedClusterQualitySparse(spike_clusters, pc_features, pc_feature_ind, pc_Nchans, clustersToProcess);
else
    [clusterIDs, unitQuality, contaminationRate] = maskedClusterQualitySparse(spike_clusters, pc_features, pc_feature_ind,pc_Nchans);
end

end