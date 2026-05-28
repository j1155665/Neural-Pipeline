function unitInfo = create_area_map(Info, data)
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
    otherwise
        error('No area map found for session date: %s', session_date_str);
end

group_cell = cellstr(data.dots3DMP.unit.cluster_group.group);
group_matches = strcmp(group_cell, unitInfo.unit_profile)';

unitInfo.MST = find((data.dots3DMP.unit.depth > unitInfo.area_map.MST(1)) ...
    & (data.dots3DMP.unit.depth < unitInfo.area_map.MST(2)) ...
    & group_matches);

unitInfo.VPS = find((data.dots3DMP.unit.depth > unitInfo.area_map.VPS(1)) ...
    & (data.dots3DMP.unit.depth < unitInfo.area_map.VPS(2)) ...
    & group_matches);
if ~isempty(unitInfo.area_map.MT)
    unitInfo.MT = find((data.dots3DMP.unit.depth > unitInfo.area_map.MT(1)) ...
        & (data.dots3DMP.unit.depth < unitInfo.area_map.MT(2)) ...
        & group_matches);
else
    unitInfo.MT = [];
end

unitInfo.dual = find((data.dots3DMP.unit.depth > unitInfo.area_map.dual(1)) ...
    & (data.dots3DMP.unit.depth < unitInfo.area_map.dual(2)) ...
    & group_matches);

end