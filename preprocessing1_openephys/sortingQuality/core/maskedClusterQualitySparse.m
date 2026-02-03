function [clusterIDs, unitQuality, contaminationRate] = maskedClusterQualitySparse(clu, fet, fetInds, fetNchans, targetClusters)
% - clu is 1 x nSpikes
% - fet is nSpikes x nPCsPerChan x nInclChans
% - fetInds is nClusters x nInclChans (sorted in descending order of
% relevance for this template)
% - targetClusters is vector of cluster IDs to compute metrics for (optional)
% - fetNchans is an integer, the number of features to use

% Handle different input argument combinations
if nargin < 4
    fetNchans = min(4, size(fetInds,2));
    targetClusters = unique(clu);
elseif nargin < 5
    targetClusters = unique(clu);
end


nFetPerChan = size(fet,2);
fetN = fetNchans*nFetPerChan; % now number of features total

N = numel(clu);
assert(fetNchans <= size(fet, 3) && size(fet, 1) == N , 'bad input(s)')

% Process the requested clusters (or all if none specified)
clusterIDs = targetClusters(:)';
unitQuality = zeros(size(clusterIDs));
contaminationRate = zeros(size(clusterIDs));

% Get all available clusters for indexing into fetInds
allClusterIDs = unique(clu);

fprintf('%12s\tQuality\tContamination\n', 'ID');
for c = 1:numel(clusterIDs)
    
    thisClusterID = clusterIDs(c);
    theseSp = clu==thisClusterID;
    n = sum(theseSp); % #spikes in this cluster
    
    if n < fetN || n >= N/2
        % cannot compute mahalanobis distance if less data points than
        % dimensions or if > 50% of all spikes are in this cluster
        unitQuality(c) = 0;
        contaminationRate(c) = NaN;
        continue
    end
    
    % Find the index of this cluster in the allClusterIDs list for fetInds indexing
    clusterIdx = find(allClusterIDs == thisClusterID);
    if isempty(clusterIdx)
        unitQuality(c) = 0;
        contaminationRate(c) = NaN;
        continue
    end
    
fetThisCluster = reshape(fet(theseSp,:,1:fetNchans), n, []);

% now we need to find other spikes that exist on the same channels
theseChans = fetInds(clusterIdx,1:fetNchans);

% for each other cluster, determine whether it has at least one of
% those channels. If so, add its spikes, with its features put into the
% correct places
nInd = 1; fetOtherClusters = zeros(0,size(fet,2),fetNchans);
for c2 = 1:numel(allClusterIDs)
    if allClusterIDs(c2) ~= thisClusterID
        c2Idx = c2;  % c2 is already the index in allClusterIDs
        chansC2Has = fetInds(c2Idx,:);
        
        % Check if this other cluster shares any channels
        if any(ismember(chansC2Has, theseChans))
            theseOtherSpikes = clu==allClusterIDs(c2);
            nOtherSpikes = sum(theseOtherSpikes);
            
            
            if nOtherSpikes > 0  % Make sure there are spikes to add
                for f = 1:length(theseChans)
                    if ismember(theseChans(f), chansC2Has)
                        thisCfetInd = find(chansC2Has==theseChans(f),1);
                        fetOtherClusters(nInd:nInd+nOtherSpikes-1,:,f) = ...
                            fet(theseOtherSpikes,:,thisCfetInd);
                    end                
                end
                nInd = nInd + nOtherSpikes;
            end
        end
    end
end
    
    fetOtherClusters = reshape(fetOtherClusters, size(fetOtherClusters,1), []);
    
    [uQ, cR] = maskedClusterQualityCore(fetThisCluster, fetOtherClusters);
    
    unitQuality(c) = uQ;
    contaminationRate(c) = cR;
    
    fprintf('cluster %3d: \t%6.1f\t%6.2f\n', thisClusterID, unitQuality(c), contaminationRate(c));
    
    if uQ>1000
        keyboard;
    end
    
end

end