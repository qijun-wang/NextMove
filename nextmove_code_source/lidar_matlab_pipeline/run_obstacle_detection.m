%% =====================================================================
%% NextMove: LiDAR obstacle clustering detector for preprocessed PCD frames
%% Geometry-only pipeline: no PointPillars, PointNet, YOLO, or training.
%% =====================================================================
clear; clc; close all;

%% 1. Input and output paths
workspaceRoot = fileparts(mfilename("fullpath"));
pcdFolder = fullfile(workspaceRoot, "pointcloud_preprocessed");
resultsFolder = fullfile(workspaceRoot, "obstacle_detection_results");

pcdFiles = dir(fullfile(pcdFolder, "*.pcd"));
if isempty(pcdFiles)
    error("No .pcd files were found in: %s", pcdFolder);
end

frameIds = arrayfun(@(file) parseFrameId(file.name), pcdFiles);
[~, order] = sort(frameIds);
pcdFiles = pcdFiles(order);
totalFrames = numel(pcdFiles);

if ~isfolder(resultsFolder)
    mkdir(resultsFolder);
end

%% 2. Detector parameters
detectorOptions = struct( ...
    "Roi", [-20 20 -35 35 -2.5 3.0], ...
    "EgoRoi", [-1.0 1.0 -2.5 1.8 -2.0 0.6], ...
    "RemoveEgoVehicle", true, ...
    "RemoveGround", false, ...
    "GroundMethod", "smrf", ...
    "DownsampleGridStep", 0.10, ...
    "Denoise", true, ...
    "ClusterDistance", 0.65, ...
    "MinClusterPoints", 8, ...
    "MaxClusterPoints", inf, ...
    "MinObstacleHeight", 0.10, ...
    "MaxObstacleHeight", 3.50, ...
    "MinObstaclePlanarSize", 0.10, ...
    "MaxObstaclePlanarSize", 12.0, ...
    "MaxAspectRatio", 12.0);

% Set to inf to process the full sequence. Use a smaller number for quick
% parameter tuning.
maxFramesToProcess = inf;
framesToProcess = min(totalFrames, maxFramesToProcess);

% Visualization is useful for tuning. Keep it off for fast batch export.
showLiveView = false;
viewEveryNFrames = 5;

fprintf("Loaded %d PCD frames from %s\n", totalFrames, pcdFolder);
fprintf("Processing %d frames for obstacle clustering...\n", framesToProcess);

%% 3. Frame-by-frame detection
allObstacleTables = cell(framesToProcess, 1);
frameSummaries = table( ...
    zeros(framesToProcess, 1), strings(framesToProcess, 1), ...
    zeros(framesToProcess, 1), zeros(framesToProcess, 1), ...
    'VariableNames', {'frameId', 'sourceFile', 'numObstacles', 'numPointsFiltered'});

if showLiveView
    figure("Name", "NextMove LiDAR Obstacle Clustering", "Color", "w");
end

for frameIdx = 1:framesToProcess
    sourceFile = string(pcdFiles(frameIdx).name);
    frameId = parseFrameId(sourceFile);
    ptCloud = pcread(fullfile(pcdFolder, sourceFile));

    [obstacleList, debug] = detectLidarObstacles( ...
        ptCloud, ...
        "FrameId", frameId, ...
        "SourceFile", sourceFile, ...
        "Roi", detectorOptions.Roi, ...
        "EgoRoi", detectorOptions.EgoRoi, ...
        "RemoveEgoVehicle", detectorOptions.RemoveEgoVehicle, ...
        "RemoveGround", detectorOptions.RemoveGround, ...
        "GroundMethod", detectorOptions.GroundMethod, ...
        "DownsampleGridStep", detectorOptions.DownsampleGridStep, ...
        "Denoise", detectorOptions.Denoise, ...
        "ClusterDistance", detectorOptions.ClusterDistance, ...
        "MinClusterPoints", detectorOptions.MinClusterPoints, ...
        "MaxClusterPoints", detectorOptions.MaxClusterPoints, ...
        "MinObstacleHeight", detectorOptions.MinObstacleHeight, ...
        "MaxObstacleHeight", detectorOptions.MaxObstacleHeight, ...
        "MinObstaclePlanarSize", detectorOptions.MinObstaclePlanarSize, ...
        "MaxObstaclePlanarSize", detectorOptions.MaxObstaclePlanarSize, ...
        "MaxAspectRatio", detectorOptions.MaxAspectRatio);

    allObstacleTables{frameIdx} = obstacleList;
    frameSummaries.frameId(frameIdx) = frameId;
    frameSummaries.sourceFile(frameIdx) = sourceFile;
    frameSummaries.numObstacles(frameIdx) = height(obstacleList);
    frameSummaries.numPointsFiltered(frameIdx) = debug.filteredPointCloud.Count;

    if showLiveView && (frameIdx == 1 || rem(frameIdx, viewEveryNFrames) == 0)
        visualizeObstacleFrame(debug.filteredPointCloud, obstacleList, frameIdx, framesToProcess);
    end

    if rem(frameIdx, 50) == 0 || frameIdx == framesToProcess
        fprintf("Frame %d/%d | obstacles: %d | filtered points: %d\n", ...
            frameIdx, framesToProcess, height(obstacleList), debug.filteredPointCloud.Count);
    end
end

%% 4. Save obstacle list for downstream controller modules
obstacleTable = vertcat(allObstacleTables{:});

timestamp = string(datetime("now", "Format", "yyyyMMdd_HHmmss"));
csvFile = fullfile(resultsFolder, "nextmove_obstacle_list_" + timestamp + ".csv");
summaryCsvFile = fullfile(resultsFolder, "nextmove_obstacle_summary_" + timestamp + ".csv");
matFile = fullfile(resultsFolder, "nextmove_obstacle_detection_" + timestamp + ".mat");

writetable(obstacleTable, csvFile);
writetable(frameSummaries, summaryCsvFile);
save(matFile, ...
    "obstacleTable", "allObstacleTables", "frameSummaries", ...
    "detectorOptions", "pcdFolder", "resultsFolder", "-v7.3");

fprintf("\nObstacle detection finished.\n");
fprintf("Obstacle list CSV: %s\n", csvFile);
fprintf("Frame summary CSV: %s\n", summaryCsvFile);
fprintf("MAT results: %s\n", matFile);

%% Local helper functions
function frameId = parseFrameId(fileName)
token = regexp(string(fileName), "^\d+", "match", "once");
if strlength(token) == 0
    frameId = NaN;
else
    frameId = str2double(token);
end
end

function visualizeObstacleFrame(ptCloud, obstacleList, frameIdx, totalFrames)
clf;
ax = axes;
pcshow(ptCloud, "Parent", ax);
hold(ax, "on");

for rowIdx = 1:height(obstacleList)
    drawObstacleBox(ax, obstacleList(rowIdx, :));
    plot3(ax, obstacleList.centroidX(rowIdx), obstacleList.centroidY(rowIdx), ...
        obstacleList.centroidZ(rowIdx), "ro", "MarkerSize", 7, "LineWidth", 1.5);
    text(ax, obstacleList.centroidX(rowIdx), obstacleList.centroidY(rowIdx), ...
        obstacleList.centroidZ(rowIdx) + 0.3, sprintf("%d", obstacleList.obstacleId(rowIdx)), ...
        "Color", "y", "FontWeight", "bold");
end

xlabel(ax, "X lateral (m)");
ylabel(ax, "Y longitudinal (m)");
zlabel(ax, "Z height (m)");
title(ax, sprintf("Obstacle clustering | frame %d/%d | obstacles: %d", ...
    frameIdx, totalFrames, height(obstacleList)));
view(ax, 2);
axis(ax, "equal");
grid(ax, "on");
hold(ax, "off");
drawnow limitrate;
end

function drawObstacleBox(ax, row)
x = [row.minX row.maxX row.maxX row.minX row.minX];
y = [row.minY row.minY row.maxY row.maxY row.minY];
z = repmat(row.minZ, size(x));
plot3(ax, x, y, z, "y-", "LineWidth", 1.2);
z = repmat(row.maxZ, size(x));
plot3(ax, x, y, z, "y-", "LineWidth", 1.2);

for k = 1:4
    plot3(ax, [x(k) x(k)], [y(k) y(k)], [row.minZ row.maxZ], "y-", "LineWidth", 1.2);
end
end
