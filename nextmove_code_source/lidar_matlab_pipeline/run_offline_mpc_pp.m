%% =====================================================================
%% NextMove: single-frame offline MPC and Pure Pursuit tracking comparison
%%
%% Purpose:
%%   Use MATLAB Pure Pursuit and finite-horizon MPC controllers to track
%%   the same local A* path generated on one fused risk-aware cost map.
%%
%% This is a single-frame offline demo, not a closed-loop validation.
%% =====================================================================
clear; clc; close all;

workspaceRoot = fileparts(mfilename("fullpath"));
costmapFolder = fullfile(workspaceRoot, "costmap_results");
plannerInputFolder = fullfile(costmapFolder, "planner_inputs");
resultsFolder = fullfile(costmapFolder, "pure_pursuit_offline_demo");

if ~isfolder(resultsFolder)
    mkdir(resultsFolder);
end

frameId = readFrameIdFromEnv("NEXTMOVE_PP_FRAME", 1);
goalXY = readXYFromEnv("NEXTMOVE_PP_GOAL_X", "NEXTMOVE_PP_GOAL_Y", [-15, -30]);
startXY = readXYFromEnv("NEXTMOVE_PP_START_X", "NEXTMOVE_PP_START_Y", [0, 0]);

plannerInputFile = fullfile(plannerInputFolder, sprintf("plannerInput_frame_%05d.mat", frameId));
if ~isfile(plannerInputFile)
    error("Planner input for frame %05d was not found: %s", frameId, plannerInputFile);
end

loaded = load(plannerInputFile, "plannerInput");
plannerInput = loaded.plannerInput;

fprintf("Loaded planner input: %s\n", plannerInputFile);
fprintf("Pure Pursuit target frame %05d | goal [%.2f %.2f]\n", ...
    plannerInput.frameId, goalXY(1), goalXY(2));

%% 1. Build Navigation Toolbox binary map from hard collisionGrid
if ~isfield(plannerInput, "collisionGrid")
    error("plannerInput does not contain collisionGrid. Regenerate costmap with the latest build_costmap.m.");
end

collisionGrid = logical(plannerInput.collisionGrid);
mapForPlanner = binaryOccupancyMap(flipud(collisionGrid), plannerInput.mapResolution);
mapForPlanner.LocalOriginInWorld = plannerInput.localOrigin;
trackingPlanningInflationRadius = readScalarEnv("NEXTMOVE_PP_PLANNING_INFLATION", 1.20);
if trackingPlanningInflationRadius > 0
    inflate(mapForPlanner, trackingPlanningInflationRadius);
end
plannerOccupancyGrid = occupancyMatrix(mapForPlanner) > 0;

mapForValidation = binaryOccupancyMap(flipud(collisionGrid), plannerInput.mapResolution);
mapForValidation.LocalOriginInWorld = plannerInput.localOrigin;

%% 2. Plan local path with MATLAB plannerAStarGrid
planner = plannerAStarGrid(mapForPlanner, ...
    "DiagonalSearch", "off", ...
    "TieBreaker", "on");

startGrid = nearestFreeGrid(world2grid(mapForPlanner, startXY), plannerOccupancyGrid, 25);
goalGrid = nearestFreeGrid(world2grid(mapForPlanner, goalXY), plannerOccupancyGrid, 25);
startXYFree = grid2world(mapForPlanner, startGrid);
goalXYFree = grid2world(mapForPlanner, goalGrid);

pathGrid = plan(planner, startGrid, goalGrid);
if isempty(pathGrid)
    error("plannerAStarGrid did not find a path from [%.2f %.2f] to [%.2f %.2f].", ...
        startXY(1), startXY(2), goalXY(1), goalXY(2));
end

pathXY = grid2world(mapForPlanner, pathGrid);
astarCollisionReport = validateXYPath(pathXY, mapForValidation, plannerInput, 0.20);
if ~astarCollisionReport.collisionFree
    error("A* path is not collision-free. gridHits=%d, bboxHits=%d", ...
        astarCollisionReport.numGridCollisionSamples, astarCollisionReport.numBboxCollisionSamples);
end

fprintf("A* waypoints: %d | length %.2f m\n", size(pathXY, 1), computePathLength(pathXY));

%% 3. Track path with MATLAB Pure Pursuit controller
pp = controllerPurePursuit;
pp.Waypoints = pathXY;
pp.DesiredLinearVelocity = readScalarEnv("NEXTMOVE_PP_LINEAR_VELOCITY", 0.35);
pp.MaxAngularVelocity = readScalarEnv("NEXTMOVE_PP_MAX_ANGULAR_VELOCITY", 6.0);
pp.LookaheadDistance = readScalarEnv("NEXTMOVE_PP_LOOKAHEAD_DISTANCE", 0.10);

dt = readScalarEnv("NEXTMOVE_PP_DT", 0.10);
defaultMaxSimTime = max(160.0, 1.30 * computePathLength(pathXY) / pp.DesiredLinearVelocity);
maxSimTime = readScalarEnv("NEXTMOVE_PP_MAX_TIME", defaultMaxSimTime);
goalTolerance = readScalarEnv("NEXTMOVE_PP_GOAL_TOLERANCE", 0.60);
wheelbase = readScalarEnv("NEXTMOVE_PP_WHEELBASE", 2.59);
maxSteeringAngle = readScalarEnv("NEXTMOVE_PP_MAX_STEERING_ANGLE", deg2rad(70));

initialHeading = initialHeadingFromPath(pathXY);
pose = [pathXY(1, 1), pathXY(1, 2), initialHeading];

numSteps = ceil(maxSimTime / dt) + 1;
poseHistory = zeros(numSteps, 3);
controlHistory = zeros(numSteps, 4);
lookaheadHistory = nan(numSteps, 2);
timeHistory = zeros(numSteps, 1);

reachedGoal = false;
usedSteps = 0;

for stepIdx = 1:numSteps
    currentTime = (stepIdx - 1) * dt;
    poseHistory(stepIdx, :) = pose;
    timeHistory(stepIdx) = currentTime;

    distanceToGoal = norm(pose(1:2) - pathXY(end, :));
    if distanceToGoal <= goalTolerance
        reachedGoal = true;
        usedSteps = stepIdx;
        break;
    end

    [v, omega, lookaheadPoint] = pp(pose);
    steeringAngle = angularVelocityToSteering(v, omega, wheelbase, maxSteeringAngle);
    controlHistory(stepIdx, :) = [v, omega, steeringAngle, wheelbase];
    lookaheadHistory(stepIdx, :) = lookaheadPoint;

    pose = integrateBicycleModel(pose, v, steeringAngle, wheelbase, dt);
    usedSteps = stepIdx;
end

poseHistory = poseHistory(1:usedSteps, :);
controlHistory = controlHistory(1:usedSteps, :);
lookaheadHistory = lookaheadHistory(1:usedSteps, :);
timeHistory = timeHistory(1:usedSteps, :);

trackingCollisionReport = validateXYPath(poseHistory(:, 1:2), mapForValidation, plannerInput, 0.20);
finalGoalError = norm(poseHistory(end, 1:2) - pathXY(end, :));
trackingPathLength = computePathLength(poseHistory(:, 1:2));

purePursuitOutput = struct( ...
    "frameId", plannerInput.frameId, ...
    "plannerInputFile", string(plannerInputFile), ...
    "startXY", startXYFree, ...
    "goalXY", goalXYFree, ...
    "pathXY", pathXY, ...
    "poseHistory", poseHistory, ...
    "controlHistory", controlHistory, ...
    "lookaheadHistory", lookaheadHistory, ...
    "timeHistory", timeHistory, ...
    "dt", dt, ...
    "vehicleModel", "kinematicBicycle", ...
    "wheelbase", wheelbase, ...
    "maxSteeringAngle", maxSteeringAngle, ...
    "trackingPlanningInflationRadius", trackingPlanningInflationRadius, ...
    "lookaheadDistance", pp.LookaheadDistance, ...
    "desiredLinearVelocity", pp.DesiredLinearVelocity, ...
    "maxAngularVelocity", pp.MaxAngularVelocity, ...
    "reachedGoal", reachedGoal, ...
    "finalGoalError", finalGoalError, ...
    "trackingPathLength", trackingPathLength, ...
    "astarCollisionReport", astarCollisionReport, ...
    "trackingCollisionReport", trackingCollisionReport, ...
    "controllerName", "controllerPurePursuit");

fprintf("Pure Pursuit reached goal: %d | final error %.2f m | sim time %.2f s\n", ...
    reachedGoal, finalGoalError, timeHistory(end));
fprintf("Bicycle model wheelbase %.2f m | max steering %.1f deg\n", ...
    wheelbase, rad2deg(maxSteeringAngle));
fprintf("Tracking collision-free: %d | gridHits=%d | bboxHits=%d\n", ...
    trackingCollisionReport.collisionFree, ...
    trackingCollisionReport.numGridCollisionSamples, ...
    trackingCollisionReport.numBboxCollisionSamples);

%% 4. Track the same A* path with a finite-horizon MPC controller
runMpcComparison = readBoolEnv("NEXTMOVE_RUN_MPC", true);
mpcOutput = [];
comparisonOutput = [];
trajectoryDifference = [];

if runMpcComparison
    mpcOptions = struct( ...
        "dt", dt, ...
        "maxSimTime", maxSimTime, ...
        "goalTolerance", goalTolerance, ...
        "wheelbase", wheelbase, ...
        "maxSteeringAngle", maxSteeringAngle, ...
        "desiredLinearVelocity", readScalarEnv("NEXTMOVE_MPC_LINEAR_VELOCITY", pp.DesiredLinearVelocity), ...
        "horizonSteps", round(readScalarEnv("NEXTMOVE_MPC_HORIZON_STEPS", 4)), ...
        "positionWeight", readScalarEnv("NEXTMOVE_MPC_POSITION_WEIGHT", 45.0), ...
        "headingWeight", readScalarEnv("NEXTMOVE_MPC_HEADING_WEIGHT", 4.0), ...
        "steeringWeight", readScalarEnv("NEXTMOVE_MPC_STEERING_WEIGHT", 0.08), ...
        "steeringRateWeight", readScalarEnv("NEXTMOVE_MPC_STEERING_RATE_WEIGHT", 0.80), ...
        "collisionWeight", readScalarEnv("NEXTMOVE_MPC_COLLISION_WEIGHT", 5000.0), ...
        "referencePreviewDistance", readScalarEnv("NEXTMOVE_MPC_REFERENCE_PREVIEW", 0.30), ...
        "beamWidth", round(readScalarEnv("NEXTMOVE_MPC_BEAM_WIDTH", 4)), ...
        "numSteeringSamples", round(readScalarEnv("NEXTMOVE_MPC_STEERING_SAMPLES", 5)), ...
        "referenceJumpFactor", readScalarEnv("NEXTMOVE_MPC_REFERENCE_JUMP_FACTOR", 2.0), ...
        "useFmincon", readBoolEnv("NEXTMOVE_MPC_USE_FMINCON", false));

    mpcOutput = runFiniteHorizonMpcTracking( ...
        pathXY, poseHistory(1, :), mpcOptions, mapForValidation, plannerInput);

    fprintf("MPC reached goal: %d | final error %.2f m | sim time %.2f s\n", ...
        mpcOutput.reachedGoal, mpcOutput.finalGoalError, mpcOutput.timeHistory(end));
    fprintf("MPC collision-free: %d | gridHits=%d | bboxHits=%d\n", ...
        mpcOutput.trackingCollisionReport.collisionFree, ...
        mpcOutput.trackingCollisionReport.numGridCollisionSamples, ...
        mpcOutput.trackingCollisionReport.numBboxCollisionSamples);

    trajectoryDifference = compareControllerTrajectories(purePursuitOutput, mpcOutput);
    fprintf("PP vs independent MPC trajectory difference: max %.3f m | mean %.3f m | RMSE %.3f m\n", ...
        trajectoryDifference.maxDistance, ...
        trajectoryDifference.meanDistance, ...
        trajectoryDifference.rmseDistance);

    comparisonOutput = struct( ...
        "frameId", plannerInput.frameId, ...
        "plannerInputFile", string(plannerInputFile), ...
        "startXY", startXYFree, ...
        "goalXY", goalXYFree, ...
        "pathXY", pathXY, ...
        "purePursuit", purePursuitOutput, ...
        "mpc", mpcOutput, ...
        "trajectoryDifference", trajectoryDifference, ...
        "vehicleModel", "kinematicBicycle", ...
        "commonInput", struct( ...
            "plannerInputFile", string(plannerInputFile), ...
            "goalXY", goalXYFree, ...
            "pathXY", pathXY, ...
            "collisionGridSource", "plannerInput.collisionGrid"));
else
    fprintf("MPC comparison skipped. Set NEXTMOVE_RUN_MPC=1 to enable it.\n");
end

%% 5. Save output and visualization
timestamp = string(datetime("now", "Format", "yyyyMMdd_HHmmss"));
outputMat = fullfile(resultsFolder, "nextmove_pure_pursuit_offline_" + timestamp + ".mat");
outputPng = fullfile(resultsFolder, sprintf("pure_pursuit_frame_%05d_%s.png", ...
    plannerInput.frameId, timestamp));
outputGif = fullfile(resultsFolder, sprintf("pure_pursuit_frame_%05d_%s.gif", ...
    plannerInput.frameId, timestamp));
comparisonPng = fullfile(resultsFolder, sprintf("controller_comparison_frame_%05d_%s.png", ...
    plannerInput.frameId, timestamp));
comparisonGif = fullfile(resultsFolder, sprintf("controller_comparison_frame_%05d_%s.gif", ...
    plannerInput.frameId, timestamp));

save(outputMat, "purePursuitOutput", "mpcOutput", "comparisonOutput", ...
    "trajectoryDifference", "plannerInput", "-v7.3");

if runMpcComparison
    visualizeControllerComparison(plannerInput, comparisonOutput, collisionGrid, comparisonPng);
else
    visualizePurePursuitOffline(plannerInput, purePursuitOutput, collisionGrid, outputPng);
end

writePurePursuitSummary(resultsFolder, timestamp, purePursuitOutput, mpcOutput, trajectoryDifference);

writeAnimation = readBoolEnv("NEXTMOVE_WRITE_GIF", true);
if writeAnimation
    if runMpcComparison
        animateControllerComparison(plannerInput, comparisonOutput, collisionGrid, comparisonGif);
    else
        animatePurePursuitOffline(plannerInput, purePursuitOutput, collisionGrid, outputGif);
    end
else
    fprintf("GIF animation skipped. Set NEXTMOVE_WRITE_GIF=1 to enable it.\n");
end

fprintf("Pure Pursuit output MAT: %s\n", outputMat);
if runMpcComparison
    fprintf("Comparison PNG: %s\n", comparisonPng);
    if writeAnimation
        fprintf("Comparison GIF: %s\n", comparisonGif);
    end
else
    fprintf("Pure Pursuit PNG: %s\n", outputPng);
    if writeAnimation
        fprintf("Pure Pursuit GIF: %s\n", outputGif);
    end
end
disp("Single-frame offline Pure Pursuit vs independent MPC tracking comparison finished.");

%% Local helper functions
function frameId = readFrameIdFromEnv(envName, defaultValue)
value = str2double(getenv(envName));
if isnan(value) || value <= 0
    frameId = defaultValue;
else
    frameId = round(value);
end
end

function xy = readXYFromEnv(xEnvName, yEnvName, defaultXY)
xValue = str2double(getenv(xEnvName));
yValue = str2double(getenv(yEnvName));
if isnan(xValue) || isnan(yValue)
    xy = defaultXY;
else
    xy = [xValue, yValue];
end
end

function value = readScalarEnv(envName, defaultValue)
envValue = str2double(getenv(envName));
if isnan(envValue)
    value = defaultValue;
else
    value = envValue;
end
end

function value = readBoolEnv(envName, defaultValue)
envValue = lower(strtrim(string(getenv(envName))));
if strlength(envValue) == 0
    value = defaultValue;
else
    value = any(envValue == ["1", "true", "yes", "on"]);
end
end

function freeGrid = nearestFreeGrid(grid, occupancyGrid, searchRadiusCells)
numRows = size(occupancyGrid, 1);
numCols = size(occupancyGrid, 2);
row = grid(1);
col = grid(2);

if row >= 1 && row <= numRows && col >= 1 && col <= numCols && ~occupancyGrid(row, col)
    freeGrid = grid;
    return;
end

for radius = 1:searchRadiusCells
    rowMin = max(1, row - radius);
    rowMax = min(numRows, row + radius);
    colMin = max(1, col - radius);
    colMax = min(numCols, col + radius);
    [rr, cc] = ndgrid(rowMin:rowMax, colMin:colMax);
    ringMask = abs(rr - row) == radius | abs(cc - col) == radius;
    candidates = [rr(ringMask), cc(ringMask)];
    freeMask = ~occupancyGrid(sub2ind(size(occupancyGrid), candidates(:, 1), candidates(:, 2)));
    candidates = candidates(freeMask, :);

    if ~isempty(candidates)
        [~, bestIdx] = min(sum((candidates - grid).^2, 2));
        freeGrid = candidates(bestIdx, :);
        return;
    end
end

error("No free grid cell was found near [%d %d].", row, col);
end

function heading = initialHeadingFromPath(pathXY)
if size(pathXY, 1) < 2
    heading = 0;
else
    delta = pathXY(2, :) - pathXY(1, :);
    heading = atan2(delta(2), delta(1));
end
end

function steeringAngle = angularVelocityToSteering(v, omega, wheelbase, maxSteeringAngle)
if abs(v) < 1e-3
    steeringAngle = 0;
else
    steeringAngle = atan(wheelbase * omega / v);
end
steeringAngle = min(max(steeringAngle, -maxSteeringAngle), maxSteeringAngle);
end

function poseNext = integrateBicycleModel(pose, v, steeringAngle, wheelbase, dt)
theta = pose(3);
poseNext = pose;
poseNext(1) = pose(1) + v * cos(theta) * dt;
poseNext(2) = pose(2) + v * sin(theta) * dt;
poseNext(3) = wrapToPiLocal(theta + (v / wheelbase) * tan(steeringAngle) * dt);
end

function angle = wrapToPiLocal(angle)
angle = mod(angle + pi, 2 * pi) - pi;
end

function report = validateXYPath(pathXY, mapForPlanner, plannerInput, bboxMargin)
densePathXY = densifyPath(pathXY, 0.02);
gridHits = checkOccupancy(mapForPlanner, densePathXY) > 0;
bboxHits = pathHitsTrackedBboxes(densePathXY, plannerInput.trackedObstacles, bboxMargin);

report = struct( ...
    "collisionFree", ~any(gridHits) && ~any(bboxHits), ...
    "numGridCollisionSamples", nnz(gridHits), ...
    "numBboxCollisionSamples", nnz(bboxHits), ...
    "bboxMargin", bboxMargin);
end

function writePurePursuitSummary(resultsFolder, timestamp, purePursuitOutput, mpcOutput, trajectoryDifference)
summaryFile = fullfile(resultsFolder, "pure_pursuit_summary_" + timestamp + ".txt");
fid = fopen(summaryFile, "w");
if fid < 0
    warning("Could not write summary file: %s", summaryFile);
    return;
end
cleanupObj = onCleanup(@() fclose(fid));

fprintf(fid, "NextMove Pure Pursuit offline validation\n");
fprintf(fid, "Frame: %05d\n", purePursuitOutput.frameId);
fprintf(fid, "Controller: %s\n", purePursuitOutput.controllerName);
fprintf(fid, "Reached goal: %d\n", purePursuitOutput.reachedGoal);
fprintf(fid, "Final goal error: %.3f m\n", purePursuitOutput.finalGoalError);
fprintf(fid, "Tracking path length: %.3f m\n", purePursuitOutput.trackingPathLength);
fprintf(fid, "Simulation time: %.3f s\n", purePursuitOutput.timeHistory(end));
fprintf(fid, "Collision-free: %d\n", purePursuitOutput.trackingCollisionReport.collisionFree);
fprintf(fid, "Grid collision samples: %d\n", ...
    purePursuitOutput.trackingCollisionReport.numGridCollisionSamples);
fprintf(fid, "BBox collision samples: %d\n", ...
    purePursuitOutput.trackingCollisionReport.numBboxCollisionSamples);
fprintf(fid, "Planning inflation radius: %.3f m\n", ...
    purePursuitOutput.trackingPlanningInflationRadius);
fprintf(fid, "Lookahead distance: %.3f m\n", purePursuitOutput.lookaheadDistance);
fprintf(fid, "Desired linear velocity: %.3f m/s\n", ...
    purePursuitOutput.desiredLinearVelocity);

if ~isempty(mpcOutput)
    fprintf(fid, "\nMPC comparison\n");
    fprintf(fid, "Reached goal: %d\n", mpcOutput.reachedGoal);
    fprintf(fid, "Final goal error: %.3f m\n", mpcOutput.finalGoalError);
    fprintf(fid, "Tracking path length: %.3f m\n", mpcOutput.trackingPathLength);
    fprintf(fid, "Simulation time: %.3f s\n", mpcOutput.timeHistory(end));
    fprintf(fid, "Collision-free: %d\n", mpcOutput.trackingCollisionReport.collisionFree);
    fprintf(fid, "Grid collision samples: %d\n", ...
        mpcOutput.trackingCollisionReport.numGridCollisionSamples);
    fprintf(fid, "BBox collision samples: %d\n", ...
        mpcOutput.trackingCollisionReport.numBboxCollisionSamples);

    if ~isempty(trajectoryDifference)
        fprintf(fid, "\nTrajectory difference: Pure Pursuit vs independent MPC\n");
        fprintf(fid, "Max distance: %.3f m\n", trajectoryDifference.maxDistance);
        fprintf(fid, "Mean distance: %.3f m\n", trajectoryDifference.meanDistance);
        fprintf(fid, "RMSE distance: %.3f m\n", trajectoryDifference.rmseDistance);
    end
end

fprintf("Summary TXT: %s\n", summaryFile);
end

function densePath = densifyPath(pathXY, sampleStep)
if size(pathXY, 1) < 2
    densePath = pathXY;
    return;
end

parts = cell(size(pathXY, 1) - 1, 1);
for idx = 1:size(pathXY, 1) - 1
    p0 = pathXY(idx, :);
    p1 = pathXY(idx + 1, :);
    segmentLength = norm(p1 - p0);
    n = max(2, ceil(segmentLength / sampleStep) + 1);
    t = linspace(0, 1, n)';
    segment = p0 + (p1 - p0) .* t;
    if idx > 1
        segment = segment(2:end, :);
    end
    parts{idx} = segment;
end
densePath = vertcat(parts{:});
end

function hits = pathHitsTrackedBboxes(pathXY, tracks, margin)
hits = false(size(pathXY, 1), 1);
for rowIdx = 1:height(tracks)
    track = tracks(rowIdx, :);
    inBox = ...
        pathXY(:, 1) >= track.minX - margin & ...
        pathXY(:, 1) <= track.maxX + margin & ...
        pathXY(:, 2) >= track.minY - margin & ...
        pathXY(:, 2) <= track.maxY + margin;
    hits = hits | inBox;
end
end

function lengthMeters = computePathLength(pathXY)
if size(pathXY, 1) < 2
    lengthMeters = 0;
else
    deltas = diff(pathXY, 1, 1);
    lengthMeters = sum(hypot(deltas(:, 1), deltas(:, 2)));
end
end

function diffStats = compareControllerTrajectories(purePursuitOutput, mpcOutput)
endTime = min(purePursuitOutput.timeHistory(end), mpcOutput.timeHistory(end));
dt = min(purePursuitOutput.dt, mpcOutput.dt);
commonTime = (0:dt:endTime)';

ppX = interp1(purePursuitOutput.timeHistory, purePursuitOutput.poseHistory(:, 1), commonTime, "linear", "extrap");
ppY = interp1(purePursuitOutput.timeHistory, purePursuitOutput.poseHistory(:, 2), commonTime, "linear", "extrap");
mpcX = interp1(mpcOutput.timeHistory, mpcOutput.poseHistory(:, 1), commonTime, "linear", "extrap");
mpcY = interp1(mpcOutput.timeHistory, mpcOutput.poseHistory(:, 2), commonTime, "linear", "extrap");

distances = hypot(ppX - mpcX, ppY - mpcY);
diffStats = struct( ...
    "commonTime", commonTime, ...
    "distanceHistory", distances, ...
    "maxDistance", max(distances), ...
    "meanDistance", mean(distances), ...
    "rmseDistance", sqrt(mean(distances.^2)));
end

function output = runFiniteHorizonMpcTracking(pathXY, initialPose, opts, mapForValidation, plannerInput)
dt = opts.dt;
maxSimTime = opts.maxSimTime;
goalTolerance = opts.goalTolerance;
wheelbase = opts.wheelbase;
maxSteeringAngle = opts.maxSteeringAngle;
v = opts.desiredLinearVelocity;

numSteps = ceil(maxSimTime / dt) + 1;
poseHistory = zeros(numSteps, 3);
controlHistory = zeros(numSteps, 4);
referenceHistory = nan(numSteps, 2);
timeHistory = zeros(numSteps, 1);

pathArc = cumulativePathDistance(pathXY);
pose = initialPose;
lastSteering = 0;
referenceS = 0;
warmStartSequence = zeros(max(2, opts.horizonSteps), 1);
reachedGoal = false;
usedSteps = 0;
usesFmincon = opts.useFmincon && exist("fmincon", "file") == 2;

for stepIdx = 1:numSteps
    currentTime = (stepIdx - 1) * dt;
    poseHistory(stepIdx, :) = pose;
    timeHistory(stepIdx) = currentTime;

    distanceToGoal = norm(pose(1:2) - pathXY(end, :));
    if distanceToGoal <= goalTolerance
        reachedGoal = true;
        usedSteps = stepIdx;
        break;
    end

    nearestS = nearestPathArcLength(pathXY, pathArc, pose(1:2));
    maxReferenceJump = max(opts.referenceJumpFactor * v * dt, 0.25);
    referenceS = min(max(referenceS, nearestS), referenceS + maxReferenceJump);
    referenceS = min(referenceS, pathArc(end));
    referenceHistory(stepIdx, :) = interpolatePathByArc(pathXY, pathArc, ...
        min(referenceS + opts.referencePreviewDistance, pathArc(end)));

    [steeringAngle, warmStartSequence] = solveMpcSteering( ...
        pose, lastSteering, referenceS, warmStartSequence, pathXY, pathArc, opts, mapForValidation, plannerInput, usesFmincon);
    omega = v / wheelbase * tan(steeringAngle);
    controlHistory(stepIdx, :) = [v, omega, steeringAngle, wheelbase];

    pose = integrateBicycleModel(pose, v, steeringAngle, wheelbase, dt);
    lastSteering = steeringAngle;
    usedSteps = stepIdx;
end

poseHistory = poseHistory(1:usedSteps, :);
controlHistory = controlHistory(1:usedSteps, :);
referenceHistory = referenceHistory(1:usedSteps, :);
timeHistory = timeHistory(1:usedSteps, :);

trackingCollisionReport = validateXYPath(poseHistory(:, 1:2), mapForValidation, plannerInput, 0.20);
finalGoalError = norm(poseHistory(end, 1:2) - pathXY(end, :));
trackingPathLength = computePathLength(poseHistory(:, 1:2));

output = struct( ...
    "frameId", plannerInput.frameId, ...
    "startXY", pathXY(1, :), ...
    "goalXY", pathXY(end, :), ...
    "pathXY", pathXY, ...
    "poseHistory", poseHistory, ...
    "controlHistory", controlHistory, ...
    "referenceHistory", referenceHistory, ...
    "timeHistory", timeHistory, ...
    "dt", dt, ...
    "vehicleModel", "kinematicBicycle", ...
    "wheelbase", wheelbase, ...
    "maxSteeringAngle", maxSteeringAngle, ...
    "desiredLinearVelocity", v, ...
    "horizonSteps", opts.horizonSteps, ...
    "reachedGoal", reachedGoal, ...
    "finalGoalError", finalGoalError, ...
    "trackingPathLength", trackingPathLength, ...
    "trackingCollisionReport", trackingCollisionReport, ...
    "usesFmincon", usesFmincon, ...
    "beamWidth", opts.beamWidth, ...
    "numSteeringSamples", opts.numSteeringSamples, ...
    "controllerName", "independentFiniteHorizonMPC");
end

function [steeringAngle, warmStartSequence] = solveMpcSteering( ...
    pose, lastSteering, referenceS, warmStartSequence, pathXY, pathArc, opts, mapForValidation, plannerInput, usesFmincon)

n = max(2, opts.horizonSteps);
lb = -opts.maxSteeringAngle * ones(n, 1);
ub = opts.maxSteeringAngle * ones(n, 1);
x0 = normalizeWarmStart(warmStartSequence, lastSteering, n, opts.maxSteeringAngle);

if usesFmincon
    try
        options = optimoptions("fmincon", ...
            "Display", "none", ...
            "Algorithm", "sqp", ...
            "MaxIterations", 10, ...
            "MaxFunctionEvaluations", 160, ...
            "StepTolerance", 1e-3, ...
            "OptimalityTolerance", 1e-2);
        solution = fmincon(@(u) mpcObjective( ...
            pose, lastSteering, referenceS, u, pathXY, pathArc, opts, mapForValidation, plannerInput), ...
            x0, [], [], [], [], lb, ub, [], options);
        steeringAngle = solution(1);
        warmStartSequence = shiftWarmStart(solution, opts.maxSteeringAngle);
        return;
    catch
        % Fall through to independent beam search if fmincon fails.
    end
end

[steeringAngle, warmStartSequence] = solveMpcSteeringBeamSearch( ...
    pose, lastSteering, referenceS, x0, pathXY, pathArc, opts, mapForValidation, plannerInput);
end

function x0 = normalizeWarmStart(warmStartSequence, lastSteering, n, maxSteeringAngle)
if numel(warmStartSequence) ~= n
    x0 = lastSteering * ones(n, 1);
else
    x0 = warmStartSequence(:);
end
x0 = min(max(x0, -maxSteeringAngle), maxSteeringAngle);
end

function warmStartSequence = shiftWarmStart(solution, maxSteeringAngle)
solution = min(max(solution(:), -maxSteeringAngle), maxSteeringAngle);
if numel(solution) < 2
    warmStartSequence = solution;
else
    warmStartSequence = [solution(2:end); solution(end)];
end
end

function [steeringAngle, warmStartSequence] = solveMpcSteeringBeamSearch( ...
    pose, lastSteering, referenceS, warmStartSequence, pathXY, pathArc, opts, mapForValidation, plannerInput)

n = max(2, opts.horizonSteps);
beamWidth = max(1, opts.beamWidth);
numSteeringSamples = max(3, opts.numSteeringSamples);
steeringSamples = linspace(-opts.maxSteeringAngle, opts.maxSteeringAngle, numSteeringSamples);
steeringSamples = unique([steeringSamples, 0, warmStartSequence(1)], "stable");

beamPoses = pose;
beamSequences = zeros(1, 0);
beamCosts = 0;
beamPreviousSteering = lastSteering;

for depthIdx = 1:n
    numBeam = size(beamPoses, 1);
    numSamples = numel(steeringSamples);
    candidatePoses = zeros(numBeam * numSamples, 3);
    candidateSequences = zeros(numBeam * numSamples, depthIdx);
    candidateCosts = zeros(numBeam * numSamples, 1);
    candidatePreviousSteering = zeros(numBeam * numSamples, 1);
    insertIdx = 0;

    for beamIdx = 1:numBeam
        for sampleIdx = 1:numSamples
            insertIdx = insertIdx + 1;
            steering = steeringSamples(sampleIdx);
            predPose = integrateBicycleModel( ...
                beamPoses(beamIdx, :), opts.desiredLinearVelocity, steering, opts.wheelbase, opts.dt);

            if depthIdx == 1
                sequencePrefix = steering;
            else
                sequencePrefix = [beamSequences(beamIdx, :), steering];
            end

            stepCost = mpcStepCost( ...
                predPose, steering, beamPreviousSteering(beamIdx), referenceS, depthIdx, ...
                pathXY, pathArc, opts, mapForValidation, plannerInput);

            candidatePoses(insertIdx, :) = predPose;
            candidateSequences(insertIdx, :) = sequencePrefix;
            candidateCosts(insertIdx) = beamCosts(beamIdx) + stepCost;
            candidatePreviousSteering(insertIdx) = steering;
        end
    end

    [~, order] = sort(candidateCosts, "ascend");
    keepCount = min(beamWidth, numel(order));
    keepIdx = order(1:keepCount);
    beamPoses = candidatePoses(keepIdx, :);
    beamSequences = candidateSequences(keepIdx, :);
    beamCosts = candidateCosts(keepIdx);
    beamPreviousSteering = candidatePreviousSteering(keepIdx);
end

[~, bestIdx] = min(beamCosts);
bestSequence = beamSequences(bestIdx, :)';
warmStartSequence = shiftWarmStart(bestSequence, opts.maxSteeringAngle);
steeringAngle = bestSequence(1);
end

function cost = mpcObjective(pose, lastSteering, referenceS, steeringSequence, pathXY, pathArc, opts, mapForValidation, plannerInput)
predPose = pose;
v = opts.desiredLinearVelocity;
dt = opts.dt;
cost = 0;
previousSteering = lastSteering;
startS = referenceS;

for k = 1:numel(steeringSequence)
    steering = min(max(steeringSequence(k), -opts.maxSteeringAngle), opts.maxSteeringAngle);
    predPose = integrateBicycleModel(predPose, v, steering, opts.wheelbase, dt);

    cost = cost + mpcStepCost( ...
        predPose, steering, previousSteering, startS, k, pathXY, pathArc, opts, mapForValidation, plannerInput);

    previousSteering = steering;
end

goalError = norm(predPose(1:2) - pathXY(end, :));
cost = cost + 0.5 * opts.positionWeight * goalError^2;
end

function cost = mpcStepCost(predPose, steering, previousSteering, referenceS, stepIdx, pathXY, pathArc, opts, mapForValidation, plannerInput)
refS = min(referenceS + stepIdx * opts.desiredLinearVelocity * opts.dt + opts.referencePreviewDistance, pathArc(end));
refXY = interpolatePathByArc(pathXY, pathArc, refS);
refHeading = interpolateHeadingByArc(pathXY, pathArc, refS);

posError = norm(predPose(1:2) - refXY);
headingError = wrapToPiLocal(predPose(3) - refHeading);
steeringRate = steering - previousSteering;
collisionPenalty = predictedCollisionPenalty(predPose(1:2), mapForValidation, plannerInput);

cost = ...
    opts.positionWeight * posError^2 + ...
    opts.headingWeight * headingError^2 + ...
    opts.steeringWeight * steering^2 + ...
    opts.steeringRateWeight * steeringRate^2 + ...
    opts.collisionWeight * collisionPenalty;
end

function penalty = predictedCollisionPenalty(xy, mapForValidation, plannerInput)
if checkOccupancy(mapForValidation, xy) > 0
    penalty = 1;
elseif any(pathHitsTrackedBboxes(xy, plannerInput.trackedObstacles, 0.20))
    penalty = 1;
else
    penalty = 0;
end
end

function pathArc = cumulativePathDistance(pathXY)
if size(pathXY, 1) < 2
    pathArc = 0;
    return;
end
deltas = diff(pathXY, 1, 1);
pathArc = [0; cumsum(hypot(deltas(:, 1), deltas(:, 2)))];
end

function s = nearestPathArcLength(pathXY, pathArc, xy)
[~, idx] = min(sum((pathXY - xy).^2, 2));
s = pathArc(idx);
end

function xy = interpolatePathByArc(pathXY, pathArc, s)
if isscalar(pathArc)
    xy = pathXY(1, :);
else
    s = min(max(s, pathArc(1)), pathArc(end));
    x = interp1(pathArc, pathXY(:, 1), s, "linear", "extrap");
    y = interp1(pathArc, pathXY(:, 2), s, "linear", "extrap");
    xy = [x, y];
end
end

function heading = interpolateHeadingByArc(pathXY, pathArc, s)
if size(pathXY, 1) < 2
    heading = 0;
    return;
end
s = min(max(s, pathArc(1)), pathArc(end));
idx = find(pathArc <= s, 1, "last");
idx = min(max(idx, 1), size(pathXY, 1) - 1);
delta = pathXY(idx + 1, :) - pathXY(idx, :);
heading = atan2(delta(2), delta(1));
end

function visualizeControllerComparison(plannerInput, comparisonOutput, collisionGrid, outputPng)
fig = figure("Visible", "off", "Color", "w", "Position", [100 100 1500 820]);
layout = tiledlayout(fig, 1, 2, "TileSpacing", "compact", "Padding", "compact");
axPP = nexttile(layout, 1);
axMPC = nexttile(layout, 2);
drawSingleControllerStatic(axPP, plannerInput, comparisonOutput, collisionGrid, ...
    comparisonOutput.purePursuit, "Pure Pursuit", [0.05 0.65 0.10]);
drawSingleControllerStatic(axMPC, plannerInput, comparisonOutput, collisionGrid, ...
    comparisonOutput.mpc, "MPC", [0.78 0.05 0.78]);
title(layout, sprintf("Controller comparison | frame %05d | PP err %.2f m | MPC err %.2f m", ...
    plannerInput.frameId, comparisonOutput.purePursuit.finalGoalError, comparisonOutput.mpc.finalGoalError));
if isfield(comparisonOutput, "trajectoryDifference")
    title(layout, sprintf( ...
        "Controller comparison | frame %05d | PP err %.2f m | MPC err %.2f m | diff max %.2f m mean %.2f m RMSE %.2f m", ...
        plannerInput.frameId, comparisonOutput.purePursuit.finalGoalError, comparisonOutput.mpc.finalGoalError, ...
        comparisonOutput.trajectoryDifference.maxDistance, ...
        comparisonOutput.trajectoryDifference.meanDistance, ...
        comparisonOutput.trajectoryDifference.rmseDistance));
end
exportgraphics(fig, outputPng, "Resolution", 180);
close(fig);
end

function drawSingleControllerStatic(ax, plannerInput, comparisonOutput, collisionGrid, output, controllerLabel, trajectoryColor)
xLimits = [plannerInput.localOrigin(1), plannerInput.localOrigin(1) + plannerInput.mapWidth];
yLimits = [plannerInput.localOrigin(2), plannerInput.localOrigin(2) + plannerInput.mapHeight];

imagesc(ax, xLimits, yLimits, double(plannerInput.costGrid));
set(ax, "YDir", "normal");
axis(ax, "equal");
xlim(ax, xLimits);
ylim(ax, yLimits);
colormap(ax, turbo(256));
clim(ax, [0 1]);
colorbar(ax);
hold(ax, "on");

drawMapOverlays(ax, plannerInput, collisionGrid, xLimits, yLimits);

pathXY = comparisonOutput.pathXY;
plot(ax, pathXY(:, 1), pathXY(:, 2), "c--", "LineWidth", 2.0);
plot(ax, output.poseHistory(:, 1), output.poseHistory(:, 2), ...
    "-", "Color", trajectoryColor, "LineWidth", 2.8);

drawVehicleFootprint(ax, output.poseHistory(end, :), 4.10, 1.80, trajectoryColor);

plot(ax, comparisonOutput.startXY(1), comparisonOutput.startXY(2), ...
    "go", "MarkerFaceColor", "g", "MarkerSize", 8);
plot(ax, comparisonOutput.goalXY(1), comparisonOutput.goalXY(2), ...
    "rx", "LineWidth", 2.2, "MarkerSize", 10);

legend(ax, ["Collision boundary", "A* path", controllerLabel], ...
    "Location", "southoutside");
title(ax, sprintf( ...
    "%s | reached %d | err %.2f m | collision-free %d", ...
    controllerLabel, output.reachedGoal, output.finalGoalError, ...
    output.trackingCollisionReport.collisionFree));
xlabel(ax, "X lateral (m)");
ylabel(ax, "Y longitudinal (m)");
grid(ax, "on");
box(ax, "on");
hold(ax, "off");
end

function animateControllerComparison(plannerInput, comparisonOutput, collisionGrid, outputGif)
frameDelay = readScalarEnv("NEXTMOVE_PP_ANIMATION_FRAME_DELAY", 0.08);
maxAnimationFrames = readScalarEnv("NEXTMOVE_PP_ANIMATION_MAX_FRAMES", 120);

numFrames = max( ...
    size(comparisonOutput.purePursuit.poseHistory, 1), ...
    size(comparisonOutput.mpc.poseHistory, 1));
frameIds = round(linspace(1, numFrames, min(maxAnimationFrames, numFrames)));
frameIds = unique(frameIds, "stable");

fig = figure("Visible", "off", "Color", "w", "Position", [100 100 1500 820]);
layout = tiledlayout(fig, 1, 2, "TileSpacing", "compact", "Padding", "compact");
axPP = nexttile(layout, 1);
axMPC = nexttile(layout, 2);

for frameIdx = 1:numel(frameIds)
    k = frameIds(frameIdx);
    drawControllerComparisonFrame(axPP, axMPC, layout, plannerInput, comparisonOutput, collisionGrid, k);

    frame = getframe(fig);
    [imageIndexed, colorMap] = rgb2ind(frame2im(frame), 256);

    if frameIdx == 1
        imwrite(imageIndexed, colorMap, outputGif, ...
            "gif", "LoopCount", inf, "DelayTime", frameDelay);
    else
        imwrite(imageIndexed, colorMap, outputGif, ...
            "gif", "WriteMode", "append", "DelayTime", frameDelay);
    end
end

close(fig);
end

function drawControllerComparisonFrame(axPP, axMPC, layout, plannerInput, comparisonOutput, collisionGrid, k)
cla(axPP);
cla(axMPC);

pp = comparisonOutput.purePursuit;
mpc = comparisonOutput.mpc;
ppK = min(k, size(pp.poseHistory, 1));
mpcK = min(k, size(mpc.poseHistory, 1));

drawSingleControllerFrame(axPP, plannerInput, comparisonOutput, collisionGrid, ...
    pp, ppK, "Pure Pursuit", [0.05 0.65 0.10]);
drawSingleControllerFrame(axMPC, plannerInput, comparisonOutput, collisionGrid, ...
    mpc, mpcK, "MPC", [0.78 0.05 0.78]);
if isfield(comparisonOutput, "trajectoryDifference")
    title(layout, sprintf("Controller comparison | frame %05d | t %.2f s | diff RMSE %.2f m", ...
        plannerInput.frameId, min(k * pp.dt, max(pp.timeHistory(end), mpc.timeHistory(end))), ...
        comparisonOutput.trajectoryDifference.rmseDistance));
else
    title(layout, sprintf("Controller comparison | frame %05d | t %.2f s", ...
        plannerInput.frameId, min(k * pp.dt, max(pp.timeHistory(end), mpc.timeHistory(end)))));
end
drawnow;
end

function drawSingleControllerFrame(ax, plannerInput, comparisonOutput, collisionGrid, kOutput, k, controllerLabel, trajectoryColor)
xLimits = [plannerInput.localOrigin(1), plannerInput.localOrigin(1) + plannerInput.mapWidth];
yLimits = [plannerInput.localOrigin(2), plannerInput.localOrigin(2) + plannerInput.mapHeight];

imagesc(ax, xLimits, yLimits, double(plannerInput.costGrid));
set(ax, "YDir", "normal");
axis(ax, "equal");
xlim(ax, xLimits);
ylim(ax, yLimits);
colormap(ax, turbo(256));
clim(ax, [0 1]);
hold(ax, "on");

drawMapOverlays(ax, plannerInput, collisionGrid, xLimits, yLimits);

pathXY = comparisonOutput.pathXY;
plot(ax, pathXY(:, 1), pathXY(:, 2), "c--", "LineWidth", 1.8);
plot(ax, kOutput.poseHistory(1:k, 1), kOutput.poseHistory(1:k, 2), ...
    "-", "Color", trajectoryColor, "LineWidth", 2.6);
drawVehicleFootprint(ax, kOutput.poseHistory(k, :), 4.10, 1.80, trajectoryColor);

plot(ax, comparisonOutput.startXY(1), comparisonOutput.startXY(2), ...
    "go", "MarkerFaceColor", "g", "MarkerSize", 8);
plot(ax, comparisonOutput.goalXY(1), comparisonOutput.goalXY(2), ...
    "rx", "LineWidth", 2.2, "MarkerSize", 10);

title(ax, sprintf("%s | reached %d | collision-free %d", ...
    controllerLabel, kOutput.reachedGoal, kOutput.trackingCollisionReport.collisionFree));
xlabel(ax, "X lateral (m)");
ylabel(ax, "Y longitudinal (m)");
grid(ax, "on");
box(ax, "on");
hold(ax, "off");
end

function drawMapOverlays(ax, plannerInput, collisionGrid, xLimits, yLimits)
contourX = linspace(xLimits(1), xLimits(2), size(collisionGrid, 2));
contourY = linspace(yLimits(1), yLimits(2), size(collisionGrid, 1));
contour(ax, contourX, contourY, double(collisionGrid), [1 1], ...
    "Color", [0.02 0.02 0.02], "LineWidth", 0.7);

tracks = plannerInput.trackedObstacles;
for rowIdx = 1:height(tracks)
    track = tracks(rowIdx, :);
    color = riskColor(track.riskLevel);
    rectangle(ax, "Position", ...
        [track.minX, track.minY, track.maxX - track.minX, track.maxY - track.minY], ...
        "EdgeColor", color, "LineWidth", 1.0);
end

rectangle(ax, "Position", [-1.0 -2.5 2.0 4.3], ...
    "EdgeColor", [0 0.35 0.9], "LineWidth", 1.4, "LineStyle", "--");
end

function visualizePurePursuitOffline(plannerInput, output, collisionGrid, outputPng)
fig = figure("Visible", "off", "Color", "w", "Position", [100 100 900 1050]);
ax = axes(fig);

xLimits = [plannerInput.localOrigin(1), plannerInput.localOrigin(1) + plannerInput.mapWidth];
yLimits = [plannerInput.localOrigin(2), plannerInput.localOrigin(2) + plannerInput.mapHeight];

imagesc(ax, xLimits, yLimits, double(plannerInput.costGrid));
set(ax, "YDir", "normal");
axis(ax, "equal");
xlim(ax, xLimits);
ylim(ax, yLimits);
colormap(ax, turbo(256));
clim(ax, [0 1]);
colorbar(ax);
hold(ax, "on");

contourX = linspace(xLimits(1), xLimits(2), size(collisionGrid, 2));
contourY = linspace(yLimits(1), yLimits(2), size(collisionGrid, 1));
contour(ax, contourX, contourY, double(collisionGrid), [1 1], ...
    "Color", [0.02 0.02 0.02], "LineWidth", 0.7);

tracks = plannerInput.trackedObstacles;
for rowIdx = 1:height(tracks)
    track = tracks(rowIdx, :);
    color = riskColor(track.riskLevel);
    rectangle(ax, "Position", ...
        [track.minX, track.minY, track.maxX - track.minX, track.maxY - track.minY], ...
        "EdgeColor", color, "LineWidth", 1.0);
end

plot(ax, output.pathXY(:, 1), output.pathXY(:, 2), ...
    "w--", "LineWidth", 3.2);
plot(ax, output.pathXY(:, 1), output.pathXY(:, 2), ...
    "c--", "LineWidth", 1.8);
plot(ax, output.poseHistory(:, 1), output.poseHistory(:, 2), ...
    "w-", "LineWidth", 4.0);
plot(ax, output.poseHistory(:, 1), output.poseHistory(:, 2), ...
    "g-", "LineWidth", 2.4);

plot(ax, output.startXY(1), output.startXY(2), ...
    "go", "MarkerFaceColor", "g", "MarkerSize", 8);
plot(ax, output.goalXY(1), output.goalXY(2), ...
    "rx", "LineWidth", 2.2, "MarkerSize", 10);

poseStride = max(1, floor(size(output.poseHistory, 1) / 20));
quiver(ax, output.poseHistory(1:poseStride:end, 1), ...
    output.poseHistory(1:poseStride:end, 2), ...
    cos(output.poseHistory(1:poseStride:end, 3)), ...
    sin(output.poseHistory(1:poseStride:end, 3)), ...
    0.55, "Color", [0.1 0.9 0.1], "LineWidth", 0.9);

rectangle(ax, "Position", [-1.0 -2.5 2.0 4.3], ...
    "EdgeColor", [0 0.35 0.9], "LineWidth", 1.4, "LineStyle", "--");

legend(ax, ["Collision boundary", "A* path", "", "Pure Pursuit trajectory"], ...
    "Location", "southoutside");
title(ax, sprintf("Offline Pure Pursuit tracking | frame %05d | reached: %d | collision-free: %d", ...
    plannerInput.frameId, output.reachedGoal, output.trackingCollisionReport.collisionFree));
xlabel(ax, "X lateral (m)");
ylabel(ax, "Y longitudinal (m)");
grid(ax, "on");
box(ax, "on");
hold(ax, "off");

exportgraphics(fig, outputPng, "Resolution", 180);
close(fig);
end

function animatePurePursuitOffline(plannerInput, output, collisionGrid, outputGif)
frameDelay = readScalarEnv("NEXTMOVE_PP_ANIMATION_FRAME_DELAY", 0.08);
maxAnimationFrames = readScalarEnv("NEXTMOVE_PP_ANIMATION_MAX_FRAMES", 120);
vehicleLength = readScalarEnv("NEXTMOVE_PP_VEHICLE_LENGTH", 4.10);
vehicleWidth = readScalarEnv("NEXTMOVE_PP_VEHICLE_WIDTH", 1.80);

poseHistory = output.poseHistory;
frameIds = round(linspace(1, size(poseHistory, 1), ...
    min(maxAnimationFrames, size(poseHistory, 1))));
frameIds = unique(frameIds, "stable");

fig = figure("Visible", "off", "Color", "w", "Position", [100 100 900 1050]);
ax = axes(fig);

for frameIdx = 1:numel(frameIds)
    k = frameIds(frameIdx);
    drawPurePursuitFrame(ax, plannerInput, output, collisionGrid, k, vehicleLength, vehicleWidth);

    frame = getframe(fig);
    [imageIndexed, colorMap] = rgb2ind(frame2im(frame), 256);

    if frameIdx == 1
        imwrite(imageIndexed, colorMap, outputGif, ...
            "gif", "LoopCount", inf, "DelayTime", frameDelay);
    else
        imwrite(imageIndexed, colorMap, outputGif, ...
            "gif", "WriteMode", "append", "DelayTime", frameDelay);
    end
end

close(fig);
end

function drawPurePursuitFrame(ax, plannerInput, output, collisionGrid, k, vehicleLength, vehicleWidth)
cla(ax);

xLimits = [plannerInput.localOrigin(1), plannerInput.localOrigin(1) + plannerInput.mapWidth];
yLimits = [plannerInput.localOrigin(2), plannerInput.localOrigin(2) + plannerInput.mapHeight];

imagesc(ax, xLimits, yLimits, double(plannerInput.costGrid));
set(ax, "YDir", "normal");
axis(ax, "equal");
xlim(ax, xLimits);
ylim(ax, yLimits);
colormap(ax, turbo(256));
clim(ax, [0 1]);
hold(ax, "on");

contourX = linspace(xLimits(1), xLimits(2), size(collisionGrid, 2));
contourY = linspace(yLimits(1), yLimits(2), size(collisionGrid, 1));
contour(ax, contourX, contourY, double(collisionGrid), [1 1], ...
    "Color", [0.02 0.02 0.02], "LineWidth", 0.7);

tracks = plannerInput.trackedObstacles;
for rowIdx = 1:height(tracks)
    track = tracks(rowIdx, :);
    color = riskColor(track.riskLevel);
    rectangle(ax, "Position", ...
        [track.minX, track.minY, track.maxX - track.minX, track.maxY - track.minY], ...
        "EdgeColor", color, "LineWidth", 1.0);
end

plot(ax, output.pathXY(:, 1), output.pathXY(:, 2), ...
    "c--", "LineWidth", 1.8);
plot(ax, output.poseHistory(1:k, 1), output.poseHistory(1:k, 2), ...
    "w-", "LineWidth", 4.0);
plot(ax, output.poseHistory(1:k, 1), output.poseHistory(1:k, 2), ...
    "g-", "LineWidth", 2.4);

pose = output.poseHistory(k, :);
drawVehicleFootprint(ax, pose, vehicleLength, vehicleWidth);

if k <= size(output.lookaheadHistory, 1) && all(isfinite(output.lookaheadHistory(k, :)))
    lookaheadPoint = output.lookaheadHistory(k, :);
    plot(ax, lookaheadPoint(1), lookaheadPoint(2), ...
        "mo", "MarkerFaceColor", "m", "MarkerSize", 5);
    plot(ax, [pose(1), lookaheadPoint(1)], [pose(2), lookaheadPoint(2)], ...
        "m:", "LineWidth", 1.1);
end

plot(ax, output.startXY(1), output.startXY(2), ...
    "go", "MarkerFaceColor", "g", "MarkerSize", 8);
plot(ax, output.goalXY(1), output.goalXY(2), ...
    "rx", "LineWidth", 2.2, "MarkerSize", 10);

rectangle(ax, "Position", [-1.0 -2.5 2.0 4.3], ...
    "EdgeColor", [0 0.35 0.9], "LineWidth", 1.4, "LineStyle", "--");

title(ax, sprintf( ...
    "Pure Pursuit bicycle tracking | frame %05d | t %.2f s | reached %d | collision-free %d", ...
    output.frameId, output.timeHistory(k), output.reachedGoal, ...
    output.trackingCollisionReport.collisionFree));
xlabel(ax, "X lateral (m)");
ylabel(ax, "Y longitudinal (m)");
grid(ax, "on");
box(ax, "on");
hold(ax, "off");
drawnow;
end

function drawVehicleFootprint(ax, pose, vehicleLength, vehicleWidth, color)
if nargin < 5
    color = [0.1 0.95 0.1];
end

x = pose(1);
y = pose(2);
theta = pose(3);

cornersLocal = [
     vehicleLength / 2,  vehicleWidth / 2
     vehicleLength / 2, -vehicleWidth / 2
    -vehicleLength / 2, -vehicleWidth / 2
    -vehicleLength / 2,  vehicleWidth / 2
     vehicleLength / 2,  vehicleWidth / 2
];

R = [cos(theta), -sin(theta); sin(theta), cos(theta)];
cornersWorld = cornersLocal * R' + [x, y];
plot(ax, cornersWorld(:, 1), cornersWorld(:, 2), ...
    "-", "Color", color, "LineWidth", 1.8);
plot(ax, x, y, "ko", "MarkerFaceColor", "k", "MarkerSize", 4);

frontPoint = [vehicleLength / 2, 0] * R' + [x, y];
plot(ax, [x, frontPoint(1)], [y, frontPoint(2)], ...
    "-", "Color", color, "LineWidth", 1.4);
end

function color = riskColor(riskLevel)
if riskLevel == "HIGH"
    color = [0.88 0.06 0.08];
elseif riskLevel == "MEDIUM"
    color = [0.95 0.55 0.05];
else
    color = [0.10 0.55 0.85];
end
end
