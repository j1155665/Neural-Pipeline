function isiV = isiViolations(resultsDirectory, clusterID)

%% Precompute the locations of files to be loaded
spikeClustersPath = fullfile(resultsDirectory,'spike_clusters.npy');
spikeTemplatesPath = fullfile(resultsDirectory,'spike_templates.npy');
spikeTimesPath= fullfile(resultsDirectory,'spike_times.npy');
paramsPath= fullfile(resultsDirectory,'params.py');

%% 
refDur = 0.0015;
minISI = 0.0005;

fprintf(1, 'loading data for ISI computation\n');
if exist(spikeClustersPath)
    spike_clusters = readNPY(spikeClustersPath);
else
    spike_clusters = readNPY(spikeTemplatesPath);
end

spike_times = readNPY(spikeTimesPath);
params = readKSparams(paramsPath);
spike_times = double(spike_times)/params.sample_rate;

fprintf(1, 'computing ISI violations\n');

availableClusterIDs = unique(spike_clusters);

if nargin > 1 && ~isempty(clusterID)
    requestedClusterIDs = clusterID(:)';
    
    clusterIDs = intersect(requestedClusterIDs, availableClusterIDs);
    
    missingClusters = setdiff(requestedClusterIDs, availableClusterIDs);
    if ~isempty(missingClusters)
        fprintf('Warning: The following cluster IDs were not found in data: %s\n', mat2str(missingClusters));
    end
    
    if isempty(clusterIDs)
        error('None of the specified cluster IDs exist in the data. Available clusters: %s', mat2str(availableClusterIDs));
    end
else
    clusterIDs = availableClusterIDs;
end

isiV = zeros(1,numel(clusterIDs));
for c = 1:numel(clusterIDs)
    [fpRate, numViolations] = ISIViolations(spike_times(spike_clusters==clusterIDs(c)), minISI, refDur);
    isiV(c) = fpRate;
    nSpikes = sum(spike_clusters==clusterIDs(c));    
    fprintf(1, 'isiVioation, cluster %3d: %.2f\n', clusterIDs(c), numViolations, nSpikes, fpRate);
end

end