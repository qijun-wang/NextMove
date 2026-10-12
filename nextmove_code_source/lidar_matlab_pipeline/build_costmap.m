%% =====================================================================
%% NextMove: fused risk-aware local cost map for planner / MPC control
%%
%% Inputs:
%%   1) pointcloud_preprocessed/*.pcd
%%   2) obstacle_tracking_results/nextmove_tracked_obstacle_list_*.csv
%%
%% Output:
%%   costmap_results/nextmove_planner_input_*.mat
%% =====================================================================
clear; clc; close all;

%% 1. Paths
workspaceRoot = fileparts(mfilename("fullpath"));
pcdFolder = fullfile(workspaceRoot, "pointcloud_preprocessed");
trackingFolder = fullfile(workspaceRoot, "obstacle_tracking_results");
resultsFolder = fullfile(workspaceRoot, "costmap_results");
vizFolder = fullfile(resultsFolder, "visualizations");
plannerInputFolder = fullfile(resultsFolder, "planner_inputs");

if ~isfolder(resultsFolder)
    mkdir(resultsFolder);
end
if ~isfolder(vizFolder)
    mkdir(vizFolder);
end
if ~isfolder(plannerInputFolder)
    mkdir(plannerInputFolder);
end

pcdFiles = dir(fullfile(pcdFolder, "*.pcd"));
if isempty(pcdFiles)
    error("No .pcd files were found in: %s", pcdFolder);
end

frameIds = arrayfun(@(file) parseFrameId(file.name), pcdFiles);
[~, order] = sort(frameIds);
pcdFiles = pcdFiles(order);
frameIds = frameIds(order);

trackingFiles = dir(fullfile(trackingFolder, "nextmove_tracked_obstacle_list_*.csv"));
if isempty(trackingFiles)
    error("No tracked obstacle list was found in: %s", trackingFolder);
end
[~, newestTrackingIdx] = max([trackingFiles.datenum]);
trackingCsv = fullfile(trackingFiles(newestTrackingIdx).folder, trackingFiles(newestTrackingIdx).name);
trackedTable = readtable(trackingCsv, 'TextType', 'string');

fprintf("Loaded %d PCD frames from %s\n", numel(pcdFiles), pcdFolder);
fprintf("Loaded tracked obstacle list: %s\n", trackingCsv);

%% 2. Cost map parameters
params = struct();
params.mapResolution = 10;              % cells / meter
params.mapWidth = 40;                   % X: -20 m to 20 m
params.mapHeight = 80;                  % Y: -40 m to 40 m
params.localOrigin = [-params.mapWidth / 2, -params.mapHeight / 2];
params.framePeriod = 0.1;

params.roi = [-20 20 -40 40 -2.5 3.0];
params.egoRoi = [-1.0 1.0 -2.5 1.8 -2.0 0.6];
params.staticHeightBand = [-1.2 2.2];

% Static point cloud layer: catches trees, walls, buildings, poles, and
% other non-ground structure that may not be represented by tracked bboxes.
params.staticCost = 0.50;
params.staticInflationRadius = 0.70;
params.staticDecayDistance = 0.45;
params.staticPointStride = 1;
params.collisionInflationRadius = 1.10;

% Dynamic / tracked obstacle layer.
params.lowRiskCost = 0.62;
params.mediumRiskCost = 0.82;
params.highRiskCost = 1.00;
params.lowRiskInflation = 0.80;
params.mediumRiskInflation = 1.20;
params.highRiskInflation = 1.80;
params.predictionHorizon = 1.0;
params.predictionStep = 0.25;
params.predictedCostScale = 0.72;

% Planner compatibility threshold for legacy cost-threshold consumers. The
% hard collision interface is collisionGrid, not costGrid >= threshold.
params.occupiedThreshold = 0.65;

maxFramesEnv = str2double(getenv("NEXTMOVE_MAX_COSTMAP_FRAMES"));
if isnan(maxFramesEnv) || maxFramesEnv <= 0
    maxFramesToProcess = inf;
else
    maxFramesToProcess = maxFramesEnv;
end
framesToProcess = min(numel(pcdFiles), maxFramesToProcess);
showLiveView = strcmp(getenv("NEXTMOVE_SHOW_COSTMAP"), "1");
viewEveryNFrames = 10;
saveVisualizationFrames = [1 500 1000 2000 3000 framesToProcess];
saveVisualizationFrames = unique(saveVisualizationFrames(saveVisualizationFrames >= 1));

plannerInputFiles = strings(framesToProcess, 1);
summaryRows = cell(framesToProcess, 1);

fig = [];
if showLiveView
    fig = figure("Name", "NextMove fused risk-aware cost map", "Color", "w");
end

%% 3. Build fused risk-aware cost maps
for frameIdx = 1:framesToProcess
    frameId = frameIds(frameIdx);
    pcdFile = fullfile(pcdFiles(frameIdx).folder, pcdFiles(frameIdx).name);
    ptCloud = pcread(pcdFile);
    frameTracks = trackedTable(trackedTable.frameId == frameId, :);

    staticLayer = buildStaticPointCloudLayer(ptCloud, params);
    dynamicLayer = buildTrackedRiskLayer(frameTracks, params);
    costGrid = max(staticLayer.costGrid, dynamicLayer.costGrid);
    collisionSeedGrid = staticLayer.seedGrid | dynamicLayer.seedGrid;
    collisionGrid = buildCollisionGrid(collisionSeedGrid, params);
    occupancyGrid = collisionGrid;

    localCostmap = occupancyMap(params.mapWidth, params.mapHeight, params.mapResolution);
    localCostmap.LocalOriginInWorld = params.localOrigin;
    occupiedXY = gridToWorld(find(occupancyGrid), size(costGrid), params);
    if ~isempty(occupiedXY)
        setOccupancy(localCostmap, occupiedXY, 1);
    end

    plannerInput = makePlannerInput( ...
        frameId, pcdFiles(frameIdx).name, costGrid, collisionGrid, ...
        localCostmap, staticLayer, dynamicLayer, frameTracks, params);

    plannerInputFile = fullfile(plannerInputFolder, sprintf("plannerInput_frame_%05d.mat", frameId));
    save(plannerInputFile, "plannerInput", "-v7.3");
    plannerInputFiles(frameIdx) = string(plannerInputFile);
    summaryRows{frameIdx} = makeSummaryRow(plannerInput);

    if showLiveView && (frameIdx == 1 || rem(frameIdx, viewEveryNFrames) == 0)
        visualizePlannerInput(fig, plannerInput);
    end

    if ismember(frameIdx, saveVisualizationFrames)
        outputPng = fullfile(vizFolder, sprintf("risk_aware_costmap_frame_%05d.png", frameId));
        savePlannerInputFigure(plannerInput, outputPng);
    end

    if rem(frameIdx, 50) == 0 || frameIdx == framesToProcess
        fprintf("Frame %d/%d | tracks: %d | high risk: %d | occupied cells: %d\n", ...
            frameIdx, framesToProcess, height(frameTracks), ...
            sum(frameTracks.riskLevel == "HIGH"), nnz(occupancyGrid));
    end
end

summaryTable = struct2table(vertcat(summaryRows{:}));

timestamp = string(datetime("now", "Format", "yyyyMMdd_HHmmss"));
matFile = fullfile(resultsFolder, "nextmove_planner_input_" + timestamp + ".mat");
summaryCsv = fullfile(resultsFolder, "nextmove_costmap_summary_" + timestamp + ".csv");

writetable(summaryTable, summaryCsv);
save(matFile, ...
    "plannerInputFiles", "summaryTable", "params", "trackingCsv", ...
    "pcdFolder", "plannerInputFolder", "-v7.3");

fprintf("\nFused risk-aware cost map finished.\n");
fprintf("Planner input MAT: %s\n", matFile);
fprintf("Summary CSV: %s\n", summaryCsv);

%% Local helper functions
function frameId = parseFrameId(fileName)
token = regexp(string(fileName), "^\d+", "match", "once");
if strlength(token) == 0
    frameId = NaN;
else
    frameId = str2double(token);
end
end

function layer = buildStaticPointCloudLayer(ptCloud, params)
xyz = reshape(ptCloud.Location, [], 3);
valid = all(isfinite(xyz), 2);

roi = params.roi;
insideRoi = ...
    xyz(:, 1) >= roi(1) & xyz(:, 1) <= roi(2) & ...
    xyz(:, 2) >= roi(3) & xyz(:, 2) <= roi(4) & ...
    xyz(:, 3) >= roi(5) & xyz(:, 3) <= roi(6);

ego = params.egoRoi;
insideEgo = ...
    xyz(:, 1) >= ego(1) & xyz(:, 1) <= ego(2) & ...
    xyz(:, 2) >= ego(3) & xyz(:, 2) <= ego(4) & ...
    xyz(:, 3) >= ego(5) & xyz(:, 3) <= ego(6);

heightBand = params.staticHeightBand;
insideHeightBand = xyz(:, 3) >= heightBand(1) & xyz(:, 3) <= heightBand(2);

valid = valid & insideRoi & ~insideEgo & insideHeightBand;
points = xyz(valid, 1:2);

if params.staticPointStride > 1 && size(points, 1) > params.staticPointStride
    points = points(1:params.staticPointStride:end, :);
end

seedGrid = false(gridSize(params));
idx = worldToGrid(points, params);
seedGrid(idx) = true;

costGrid = inflateSeedGrid( ...
    seedGrid, params.staticCost, ...
    params.staticInflationRadius, params.staticDecayDistance, params);

layer = struct( ...
    "name", "staticPointCloudLayer", ...
    "costGrid", single(costGrid), ...
    "seedGrid", seedGrid, ...
    "numPointsUsed", size(points, 1), ...
    "cost", params.staticCost, ...
    "inflationRadius", params.staticInflationRadius);
end

function layer = buildTrackedRiskLayer(frameTracks, params)
costGrid = zeros(gridSize(params));
riskSeedGrid = false(gridSize(params));

for rowIdx = 1:height(frameTracks)
    track = frameTracks(rowIdx, :);
    riskCost = costForRisk(track.riskLevel, params);
    inflationRadius = inflationForRisk(track.riskLevel, params);

    bboxSeed = bboxToSeedGrid(track.minX, track.maxX, track.minY, track.maxY, params);
    riskSeedGrid = riskSeedGrid | bboxSeed;
    bboxCost = inflateSeedGrid(bboxSeed, riskCost, inflationRadius, inflationRadius / 2, params);
    costGrid = max(costGrid, bboxCost);

    predictedGrid = buildPredictedTrackTube(track, params);
    if any(predictedGrid(:))
        predictedCost = inflateSeedGrid( ...
            predictedGrid, riskCost * params.predictedCostScale, ...
            inflationRadius * 0.65, max(inflationRadius * 0.35, 0.15), params);
        costGrid = max(costGrid, predictedCost);
    end
end

layer = struct( ...
    "name", "trackedRiskLayer", ...
    "costGrid", costGrid, ...
    "seedGrid", riskSeedGrid, ...
    "numTracks", height(frameTracks), ...
    "numHighRiskTracks", sum(frameTracks.riskLevel == "HIGH"), ...
    "numMediumRiskTracks", sum(frameTracks.riskLevel == "MEDIUM"), ...
    "numLowRiskTracks", sum(frameTracks.riskLevel == "LOW"));
end

function cost = costForRisk(riskLevel, params)
if riskLevel == "HIGH"
    cost = params.highRiskCost;
elseif riskLevel == "MEDIUM"
    cost = params.mediumRiskCost;
else
    cost = params.lowRiskCost;
end
end

function radius = inflationForRisk(riskLevel, params)
if riskLevel == "HIGH"
    radius = params.highRiskInflation;
elseif riskLevel == "MEDIUM"
    radius = params.mediumRiskInflation;
else
    radius = params.lowRiskInflation;
end
end

function predictedGrid = buildPredictedTrackTube(track, params)
predictedGrid = false(gridSize(params));

if track.speed < 0.2
    return;
end

times = params.predictionStep:params.predictionStep:params.predictionHorizon;
cx = track.x + track.vx .* times;
cy = track.y + track.vy .* times;
radius = max(track.radius, 0.35);

for idx = 1:numel(times)
    seed = circleToSeedGrid(cx(idx), cy(idx), radius, params);
    predictedGrid = predictedGrid | seed;
end
end

function seedGrid = bboxToSeedGrid(minX, maxX, minY, maxY, params)
[xCenters, yCenters] = gridCenters(params);
halfCell = 0.5 / params.mapResolution;
seedGrid = ...
    xCenters >= minX - halfCell & xCenters <= maxX + halfCell & ...
    yCenters >= minY - halfCell & yCenters <= maxY + halfCell;
end

function seedGrid = circleToSeedGrid(cx, cy, radius, params)
[xCenters, yCenters] = gridCenters(params);
seedGrid = hypot(xCenters - cx, yCenters - cy) <= radius;
end

function costGrid = inflateSeedGrid(seedGrid, seedCost, inflationRadius, decayDistance, params)
if ~any(seedGrid(:))
    costGrid = zeros(gridSize(params));
    return;
end

try
    distanceCells = bwdist(seedGrid);
catch
    distanceCells = fallbackDistanceTransform(seedGrid, params);
end

distanceMeters = distanceCells / params.mapResolution;
costGrid = seedCost .* exp(-distanceMeters ./ max(decayDistance, eps));
costGrid(distanceMeters > inflationRadius) = 0;
costGrid(seedGrid) = seedCost;
costGrid = min(costGrid, 1);
end

function collisionGrid = buildCollisionGrid(collisionSeedGrid, params)
if ~any(collisionSeedGrid(:))
    collisionGrid = false(gridSize(params));
    return;
end

try
    distanceCells = bwdist(collisionSeedGrid);
catch
    distanceCells = fallbackDistanceTransform(collisionSeedGrid, params);
end

distanceMeters = distanceCells / params.mapResolution;
collisionGrid = distanceMeters <= params.collisionInflationRadius;
end

function distanceCells = fallbackDistanceTransform(seedGrid, params)
[seedRows, seedCols] = find(seedGrid);
[rows, cols] = ndgrid(1:size(seedGrid, 1), 1:size(seedGrid, 2));
distanceCells = inf(size(seedGrid));

maxSeeds = min(numel(seedRows), 2500);
if numel(seedRows) > maxSeeds
    sampleIdx = round(linspace(1, numel(seedRows), maxSeeds));
    seedRows = seedRows(sampleIdx);
    seedCols = seedCols(sampleIdx);
end

for idx = 1:numel(seedRows)
    distanceCells = min(distanceCells, hypot(rows - seedRows(idx), cols - seedCols(idx)));
end

distanceCells(distanceCells > params.mapResolution * max(params.mapWidth, params.mapHeight)) = inf;
end

function plannerInput = makePlannerInput( ...
    frameId, sourceFile, costGrid, collisionGrid, localCostmap, ...
    staticLayer, dynamicLayer, frameTracks, params)

highRisk = frameTracks(frameTracks.riskLevel == "HIGH", :);
mediumRisk = frameTracks(frameTracks.riskLevel == "MEDIUM", :);
staticLayer.costGrid = single(staticLayer.costGrid);
dynamicLayer.costGrid = single(dynamicLayer.costGrid);

plannerInput = struct( ...
    "frameId", frameId, ...
    "sourceFile", string(sourceFile), ...
    "timestamp", (frameId - 1) * params.framePeriod, ...
    "costGrid", single(costGrid), ...
    "collisionGrid", collisionGrid, ...
    "occupancyGrid", collisionGrid, ...
    "occupancyMap", localCostmap, ...
    "mapResolution", params.mapResolution, ...
    "mapWidth", params.mapWidth, ...
    "mapHeight", params.mapHeight, ...
    "localOrigin", params.localOrigin, ...
    "trackedObstacles", frameTracks, ...
    "highRiskObstacles", highRisk, ...
    "mediumRiskObstacles", mediumRisk, ...
    "staticLayer", staticLayer, ...
    "dynamicLayer", dynamicLayer, ...
    "vehiclePoseLocal", [0 0 0], ...
    "safeCorridor", struct("halfWidth", 2.2, "forwardRange", 15.0), ...
    "occupiedThreshold", params.occupiedThreshold, ...
    "collisionInflationRadius", params.collisionInflationRadius);
end

function row = makeSummaryRow(plannerInput)
row = struct( ...
    "frameId", plannerInput.frameId, ...
    "sourceFile", plannerInput.sourceFile, ...
    "timestamp", plannerInput.timestamp, ...
    "numTracks", height(plannerInput.trackedObstacles), ...
    "numHighRiskTracks", height(plannerInput.highRiskObstacles), ...
    "numMediumRiskTracks", height(plannerInput.mediumRiskObstacles), ...
    "numStaticPointsUsed", plannerInput.staticLayer.numPointsUsed, ...
    "maxCost", max(plannerInput.costGrid(:)), ...
    "meanCost", mean(plannerInput.costGrid(:)), ...
    "collisionCells", nnz(plannerInput.collisionGrid), ...
    "occupiedCells", nnz(plannerInput.occupancyGrid));
end

function idx = worldToGrid(points, params)
if isempty(points)
    idx = [];
    return;
end

numRows = round(params.mapHeight * params.mapResolution);
numCols = round(params.mapWidth * params.mapResolution);

cols = floor((points(:, 1) - params.localOrigin(1)) * params.mapResolution) + 1;
rows = floor((points(:, 2) - params.localOrigin(2)) * params.mapResolution) + 1;

valid = rows >= 1 & rows <= numRows & cols >= 1 & cols <= numCols;
idx = sub2ind([numRows, numCols], rows(valid), cols(valid));
end

function xy = gridToWorld(linearIdx, gridShape, params)
if isempty(linearIdx)
    xy = zeros(0, 2);
    return;
end

[rows, cols] = ind2sub(gridShape, linearIdx);
x = params.localOrigin(1) + (cols - 0.5) / params.mapResolution;
y = params.localOrigin(2) + (rows - 0.5) / params.mapResolution;
xy = [x, y];
end

function [xCenters, yCenters] = gridCenters(params)
numRows = round(params.mapHeight * params.mapResolution);
numCols = round(params.mapWidth * params.mapResolution);
x = params.localOrigin(1) + ((1:numCols) - 0.5) / params.mapResolution;
y = params.localOrigin(2) + ((1:numRows)' - 0.5) / params.mapResolution;
[xCenters, yCenters] = meshgrid(x, y);
end

function shape = gridSize(params)
shape = [ ...
    round(params.mapHeight * params.mapResolution), ...
    round(params.mapWidth * params.mapResolution)];
end

function visualizePlannerInput(fig, plannerInput)
figure(fig);
clf(fig);
drawPlannerInput(gca, plannerInput);
drawnow limitrate;
end

function savePlannerInputFigure(plannerInput, outputPng)
fig = figure("Visible", "off", "Color", "w", "Position", [100 100 750 1000]);
drawPlannerInput(gca, plannerInput);
exportgraphics(fig, outputPng, "Resolution", 180);
close(fig);
end

function drawPlannerInput(ax, plannerInput)
xLimits = [plannerInput.localOrigin(1), plannerInput.localOrigin(1) + plannerInput.mapWidth];
yLimits = [plannerInput.localOrigin(2), plannerInput.localOrigin(2) + plannerInput.mapHeight];

imagesc(ax, xLimits, yLimits, plannerInput.costGrid);
set(ax, "YDir", "normal");
axis(ax, "equal");
xlim(ax, xLimits);
ylim(ax, yLimits);
colormap(ax, turbo(256));
clim(ax, [0 1]);
colorbar(ax);
hold(ax, "on");

tracks = plannerInput.trackedObstacles;
for rowIdx = 1:height(tracks)
    track = tracks(rowIdx, :);
    color = riskColor(track.riskLevel);
    rectangle(ax, "Position", ...
        [track.minX, track.minY, track.maxX - track.minX, track.maxY - track.minY], ...
        "EdgeColor", color, "LineWidth", 1.2);
    plot(ax, track.x, track.y, "o", ...
        "MarkerSize", 4, "MarkerFaceColor", color, "MarkerEdgeColor", "k");
    if track.speed > 0.2
        quiver(ax, track.x, track.y, track.vx, track.vy, 0.35, ...
            "Color", color, "LineWidth", 1.1, "MaxHeadSize", 1.5);
    end
end

plot(ax, 0, 0, "ks", "MarkerFaceColor", "k", "MarkerSize", 8);
rectangle(ax, "Position", [-1.0 -2.5 2.0 4.3], ...
    "EdgeColor", [0 0.35 0.9], "LineWidth", 1.4, "LineStyle", "--");

title(ax, sprintf("Fused risk-aware local cost map | frame %05d | tracks %d", ...
    plannerInput.frameId, height(tracks)));
xlabel(ax, "X lateral (m)");
ylabel(ax, "Y longitudinal (m)");
grid(ax, "on");
box(ax, "on");
hold(ax, "off");

    function color = riskColor(riskLevel)
        if riskLevel == "HIGH"
            color = [0.88 0.06 0.08];
        elseif riskLevel == "MEDIUM"
            color = [0.95 0.55 0.05];
        else
            color = [0.10 0.55 0.85];
        end
    end
end
