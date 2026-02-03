function [cids, uQ, cR, isiV, cluster_data] = computeAllMeasures(resultsDirectory, clusterID)

clusterPath = fullfile(resultsDirectory, 'cluster_info.tsv');
spikeClustersPath = fullfile(resultsDirectory,'spike_clusters.npy');
spikeTemplatesPath = fullfile(resultsDirectory,'spike_templates.npy');

if exist(clusterPath, 'file')
    cluster_data = readtable(clusterPath, 'FileType', 'text', 'Delimiter', '\t');
elseif exist(spikeClustersPath, 'file')
    clu = readNPY(spikeClustersPath);
    cluster_cgs = 3*ones(size(unique(clu))); % all unsorted
else
    clu = readNPY(spikeTemplatesPath);
    cluster_cgs = 3*ones(size(unique(clu))); % all unsorted
end

% Pass clusterID to the functions
if nargin > 1 && ~isempty(clusterID)
    [cids, uQ, cR] = sqKilosort.maskedClusterQuality(resultsDirectory, clusterID);
    isiV = sqKilosort.isiViolations(resultsDirectory, clusterID);
else
    [cids, uQ, cR] = sqKilosort.maskedClusterQuality(resultsDirectory);
    isiV = sqKilosort.isiViolations(resultsDirectory);
end

end