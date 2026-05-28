function spike_table = create_spikes(Info, unitIdx, data_spikerate)

unit_num = length(unitIdx);
trial_num = length(Info.goodtrial);
for i = 1:trial_num
    if ~isnan(data_spikerate{i, 1})
        bin_num = length(data_spikerate{i, 1});
        break
    end
end

spike_table = nan(unit_num, bin_num, trial_num);

for u = 1:unit_num
    for n = 1:trial_num
        for b = 1:bin_num
            spike_table(u,b,n) = data_spikerate{Info.goodtrial(n), unitIdx(u)}(b);
        end
    end
end

end