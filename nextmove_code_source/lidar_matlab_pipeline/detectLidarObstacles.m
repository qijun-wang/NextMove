function [obstacleList, debug] = detectLidarObstacles(ptCloudIn, opts)
% detectLidarObstacles Cluster non-ground lidar points into obstacle objects.
%
% [obstacleList, debug] = detectLidarObstacles(ptCloudIn, opts) returns a
% table with one row per detected obstacle. The output is designed for
% downstream MPC / Pure Pursuit avoidance logic and does not use any
% learning-based detector.

arguments
    ptCloudIn (1,1) pointCloud
    opts.FrameId (1,1) double = NaN
    opts.SourceFile (1,1) string = ""
    opts.Roi (1,6) double = [-20 20 -35 35 -2.5 3.0]
    opts.EgoRoi (1,6) double = [-1.0 1.0 -2.5 1.8 -2.0 0.6]
    opts.RemoveEgoVehicle (1,1) logical = true
    opts.RemoveGround (1,1) logical = false
    opts.GroundMethod (1,1) string {mustBeMember(opts.GroundMethod, ["smrf", "plane"])} = "smrf"
    opts.DownsampleGridStep (1,1) double = 0.10
    opts.Denoise (1,1) logical = true
    opts.ClusterDistance (1,1) double = 0.65
    opts.MinClusterPoints (1,1) double = 8
    opts.MaxClusterPoints (1,1) double = inf
    opts.MinObstacleHeight (1,1) double = 0.10
    opts.MaxObstacleHeight (1,1) double = 3.50
    opts.MinObstaclePlanarSize (1,1) double = 0.10
    opts.MaxObstaclePlanarSize (1,1) double = 12.0
    opts.MaxAspectRatio (1,1) double = 12.0
end

ptCloud = preprocessForClustering(ptCloudIn, opts);
debug = struct( ...
    "filteredPointCloud", ptCloud, ...
    "clusterLabels", zeros(ptCloud.Count, 1), ...
    "numClustersRaw", 0, ...
    "numObstacles", 0);

if ptCloud.Count < opts.MinClusterPoints
    obstacleList = emptyObstacleTable();
    return;
end

[labels, numClusters] = pcsegdist(ptCloud, opts.ClusterDistance);
debug.clusterLabels = labels;
debug.numClustersRaw = numClusters;

xyz = ptCloud.Location;
xyz = reshape(xyz, [], 3);
rows = cell(0, 1);

for clusterId = 1:numClusters
    idx = find(labels == clusterId);
    numPoints = numel(idx);

    if numPoints < opts.MinClusterPoints || numPoints > opts.MaxClusterPoints
        continue;
    end

    clusterPoints = xyz(idx, :);
    candidate = buildObstacleRow( ...
        clusterPoints, opts.FrameId, opts.SourceFile, clusterId, numPoints);

    if ~passesObstacleFilters(candidate, opts)
        continue;
    end

    rows{end + 1, 1} = candidate; %#ok<AGROW>
end

if isempty(rows)
    obstacleList = emptyObstacleTable();
else
    obstacleList = struct2table(vertcat(rows{:}));
    obstacleList = sortrows(obstacleList, "range");
    obstacleList.obstacleId = (1:height(obstacleList))';
end

debug.numObstacles = height(obstacleList);
end

function ptCloudOut = preprocessForClustering(ptCloudIn, opts)
xyz = ptCloudIn.Location;
xyz = reshape(xyz, [], 3);
valid = all(isfinite(xyz), 2);

roi = opts.Roi;
insideRoi = ...
    xyz(:, 1) >= roi(1) & xyz(:, 1) <= roi(2) & ...
    xyz(:, 2) >= roi(3) & xyz(:, 2) <= roi(4) & ...
    xyz(:, 3) >= roi(5) & xyz(:, 3) <= roi(6);
valid = valid & insideRoi;

if opts.RemoveEgoVehicle
    ego = opts.EgoRoi;
    insideEgo = ...
        xyz(:, 1) >= ego(1) & xyz(:, 1) <= ego(2) & ...
        xyz(:, 2) >= ego(3) & xyz(:, 2) <= ego(4) & ...
        xyz(:, 3) >= ego(5) & xyz(:, 3) <= ego(6);
    valid = valid & ~insideEgo;
end

ptCloudOut = pointCloud(xyz(valid, :));

if ptCloudOut.Count < 5
    return;
end

if opts.RemoveGround
    ptCloudOut = removeGroundIfPossible(ptCloudOut, opts.GroundMethod);
end

if opts.DownsampleGridStep > 0 && ptCloudOut.Count >= 5
    ptCloudOut = pcdownsample(ptCloudOut, "gridAverage", opts.DownsampleGridStep);
end

if opts.Denoise && ptCloudOut.Count >= 10
    ptCloudOut = pcdenoise(ptCloudOut, "NumNeighbors", 6, "Threshold", 1.0);
end
end

function ptCloudOut = removeGroundIfPossible(ptCloudIn, method)
try
    if method == "smrf"
        [~, ptCloudOut] = segmentGroundSMRF( ...
            ptCloudIn, 1.0, ...
            "MaxWindowRadius", 12, ...
            "ElevationThreshold", 0.20, ...
            "SlopeThreshold", 0.15, ...
            "ElevationScale", 1.25);
    else
        [~, ~, outliers] = pcfitplane(ptCloudIn, 0.22, [0 0 1], 15);
        ptCloudOut = select(ptCloudIn, outliers);
    end
catch
    ptCloudOut = ptCloudIn;
end
end

function row = buildObstacleRow(points, frameId, sourceFile, clusterId, numPoints)
minPt = min(points, [], 1);
maxPt = max(points, [], 1);
centroid = mean(points, 1);
dims = maxPt - minPt;

[yaw, lengthXY, widthXY] = orientedBox2D(points(:, 1:2));
radius = 0.5 * hypot(lengthXY, widthXY);
range = hypot(centroid(1), centroid(2));
bearing = atan2(centroid(2), centroid(1));
frontClearance = min(points(:, 2));
lateralClearance = min(abs(points(:, 1)));

row = struct( ...
    "frameId", frameId, ...
    "sourceFile", sourceFile, ...
    "obstacleId", clusterId, ...
    "numPoints", numPoints, ...
    "centroidX", centroid(1), ...
    "centroidY", centroid(2), ...
    "centroidZ", centroid(3), ...
    "minX", minPt(1), ...
    "maxX", maxPt(1), ...
    "minY", minPt(2), ...
    "maxY", maxPt(2), ...
    "minZ", minPt(3), ...
    "maxZ", maxPt(3), ...
    "length", lengthXY, ...
    "width", widthXY, ...
    "height", dims(3), ...
    "yaw", yaw, ...
    "radius", radius, ...
    "range", range, ...
    "bearing", bearing, ...
    "frontClearance", frontClearance, ...
    "lateralClearance", lateralClearance);
end

function keep = passesObstacleFilters(row, opts)
planarMax = max(row.length, row.width);
planarMin = min(row.length, row.width);
aspectRatio = planarMax / max(planarMin, eps);

keep = ...
    row.height >= opts.MinObstacleHeight && ...
    row.height <= opts.MaxObstacleHeight && ...
    planarMax >= opts.MinObstaclePlanarSize && ...
    planarMax <= opts.MaxObstaclePlanarSize && ...
    aspectRatio <= opts.MaxAspectRatio;
end

function [yaw, lengthXY, widthXY] = orientedBox2D(xy)
if size(xy, 1) < 3 || rank(xy - mean(xy, 1)) < 2
    extents = max(xy, [], 1) - min(xy, [], 1);
    lengthXY = max(extents);
    widthXY = min(extents);
    yaw = 0;
    return;
end

xyCentered = xy - mean(xy, 1);
[~, ~, coeff] = svd(xyCentered, "econ");
score = xyCentered * coeff;
extents = max(score, [], 1) - min(score, [], 1);
lengthXY = max(extents(1), extents(2));
widthXY = min(extents(1), extents(2));

majorAxis = coeff(:, 1);
if extents(2) > extents(1)
    majorAxis = coeff(:, 2);
end
yaw = atan2(majorAxis(2), majorAxis(1));
end

function obstacleList = emptyObstacleTable()
obstacleList = table( ...
    zeros(0, 1), strings(0, 1), zeros(0, 1), zeros(0, 1), ...
    zeros(0, 1), zeros(0, 1), zeros(0, 1), ...
    zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), ...
    zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), ...
    zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), ...
    zeros(0, 1), zeros(0, 1), zeros(0, 1), ...
    'VariableNames', { ...
        'frameId', 'sourceFile', 'obstacleId', 'numPoints', ...
        'centroidX', 'centroidY', 'centroidZ', ...
        'minX', 'maxX', 'minY', 'maxY', 'minZ', 'maxZ', ...
        'length', 'width', 'height', 'yaw', 'radius', ...
        'range', 'bearing', 'frontClearance', 'lateralClearance'});
end
