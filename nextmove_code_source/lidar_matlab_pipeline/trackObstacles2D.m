function [trackedTable, trackHistory] = trackObstacles2D(obstacleTable, opts)
% trackObstacles2D Track 2-D obstacle detections with NN association + KF.
%
% The state is [x; y; vx; vy] in the lidar/vehicle XY plane. Detections are
% associated to predicted tracks by nearest-neighbor Mahalanobis distance.

arguments
    obstacleTable table
    opts.FramePeriod (1,1) double = 0.1
    opts.ProcessNoise (1,1) double = 3.0
    opts.MeasurementNoise (1,1) double = 0.45
    opts.AssociationGate (1,1) double = 7.0
    opts.MaxAssociationDistance (1,1) double = 2.5
    opts.MaxMissedFrames (1,1) double = 8
    opts.MinHitsToConfirm (1,1) double = 2
    opts.MaxSpeed (1,1) double = 25.0
    opts.RiskHighDistance (1,1) double = 6.0
    opts.RiskMediumDistance (1,1) double = 12.0
    opts.RiskCorridorHalfWidth (1,1) double = 2.2
    opts.RiskHighTTC (1,1) double = 2.5
    opts.RiskMediumTTC (1,1) double = 5.0
end

if isempty(obstacleTable)
    trackedTable = emptyTrackedTable();
    trackHistory = struct([]);
    return;
end

obstacleTable = sortrows(obstacleTable, ["frameId", "obstacleId"]);
frameIds = unique(obstacleTable.frameId, "stable");

tracks = initializeTrackStruct();
trackHistory = initializeTrackStruct();
nextTrackId = 1;
outputRows = cell(0, 1);

for frameIdx = 1:numel(frameIds)
    frameId = frameIds(frameIdx);
    frameDetections = obstacleTable(obstacleTable.frameId == frameId, :);

    [tracks, predictedStates, predictedCovariances] = predictTracks(tracks, opts);
    [assignments, unassignedTracks, unassignedDetections] = associateDetections( ...
        tracks, predictedStates, predictedCovariances, frameDetections, opts);

    for idx = 1:size(assignments, 1)
        trackIdx = assignments(idx, 1);
        detectionIdx = assignments(idx, 2);
        [tracks(trackIdx), outputRow] = updateAssignedTrack( ...
            tracks(trackIdx), frameDetections(detectionIdx, :), frameId, opts);
        outputRows{end + 1, 1} = outputRow; %#ok<AGROW>
    end

    for idx = unassignedTracks(:)'
        tracks(idx).age = tracks(idx).age + 1;
        tracks(idx).missedFrames = tracks(idx).missedFrames + 1;
    end

    for detectionIdx = unassignedDetections(:)'
        [newTrack, outputRow] = createTrack( ...
            nextTrackId, frameDetections(detectionIdx, :), frameId, opts);
        tracks(end + 1) = newTrack; %#ok<AGROW>
        outputRows{end + 1, 1} = outputRow; %#ok<AGROW>
        nextTrackId = nextTrackId + 1;
    end

    activeMask = [tracks.missedFrames] <= opts.MaxMissedFrames;
    trackHistory = appendTrackHistory(trackHistory, tracks(~activeMask));
    tracks = tracks(activeMask);
end

trackHistory = appendTrackHistory(trackHistory, tracks);

if isempty(outputRows)
    trackedTable = emptyTrackedTable();
else
    trackedTable = struct2table(vertcat(outputRows{:}));
    trackedTable = sortrows(trackedTable, ["frameId", "trackId"]);
end
end

function tracks = initializeTrackStruct()
tracks = struct( ...
    "trackId", {}, ...
    "state", {}, ...
    "covariance", {}, ...
    "age", {}, ...
    "totalHits", {}, ...
    "consecutiveHits", {}, ...
    "missedFrames", {}, ...
    "lastFrameId", {}, ...
    "lastDetectionId", {}, ...
    "lastBbox", {}, ...
    "lastRiskLevel", {});
end

function [tracks, predictedStates, predictedCovariances] = predictTracks(tracks, opts)
numTracks = numel(tracks);
predictedStates = cell(numTracks, 1);
predictedCovariances = cell(numTracks, 1);

dt = opts.FramePeriod;
F = [1 0 dt 0; 0 1 0 dt; 0 0 1 0; 0 0 0 1];
Q = constantVelocityProcessNoise(dt, opts.ProcessNoise);

for idx = 1:numTracks
    tracks(idx).state = F * tracks(idx).state;
    tracks(idx).covariance = F * tracks(idx).covariance * F' + Q;
    tracks(idx).state(3:4) = clampVelocity(tracks(idx).state(3:4), opts.MaxSpeed);
    predictedStates{idx} = tracks(idx).state;
    predictedCovariances{idx} = tracks(idx).covariance;
end
end

function Q = constantVelocityProcessNoise(dt, sigmaA)
q = sigmaA^2;
Q = q * [ ...
    dt^4 / 4, 0, dt^3 / 2, 0; ...
    0, dt^4 / 4, 0, dt^3 / 2; ...
    dt^3 / 2, 0, dt^2, 0; ...
    0, dt^3 / 2, 0, dt^2];
end

function [assignments, unassignedTracks, unassignedDetections] = associateDetections( ...
    tracks, predictedStates, predictedCovariances, detections, opts)

numTracks = numel(tracks);
numDetections = height(detections);
assignments = zeros(0, 2);
unassignedTracks = 1:numTracks;
unassignedDetections = 1:numDetections;

if numTracks == 0 || numDetections == 0
    return;
end

cost = inf(numTracks, numDetections);
H = [1 0 0 0; 0 1 0 0];
R = eye(2) * opts.MeasurementNoise^2;

for trackIdx = 1:numTracks
    zPred = H * predictedStates{trackIdx};
    S = H * predictedCovariances{trackIdx} * H' + R;

    for detectionIdx = 1:numDetections
        z = [detections.centroidX(detectionIdx); detections.centroidY(detectionIdx)];
        residual = z - zPred;
        mahalanobisDistance = sqrt(residual' / S * residual);
        euclideanDistance = norm(residual);

        if mahalanobisDistance <= opts.AssociationGate && ...
                euclideanDistance <= opts.MaxAssociationDistance
            cost(trackIdx, detectionIdx) = mahalanobisDistance;
        end
    end
end

while true
    [bestValue, linearIdx] = min(cost(:));
    if ~isfinite(bestValue)
        break;
    end

    [trackIdx, detectionIdx] = ind2sub(size(cost), linearIdx);
    assignments(end + 1, :) = [trackIdx, detectionIdx]; %#ok<AGROW>
    cost(trackIdx, :) = inf;
    cost(:, detectionIdx) = inf;
end

unassignedTracks = setdiff(1:numTracks, assignments(:, 1));
unassignedDetections = setdiff(1:numDetections, assignments(:, 2));
end

function [track, outputRow] = updateAssignedTrack(track, detection, frameId, opts)
dt = opts.FramePeriod;
H = [1 0 0 0; 0 1 0 0];
R = eye(2) * opts.MeasurementNoise^2;
z = [detection.centroidX; detection.centroidY];

S = H * track.covariance * H' + R;
K = track.covariance * H' / S;
innovation = z - H * track.state;
track.state = track.state + K * innovation;
track.covariance = (eye(4) - K * H) * track.covariance;
track.state(3:4) = clampVelocity(track.state(3:4), opts.MaxSpeed);

track.age = track.age + 1;
track.totalHits = track.totalHits + 1;
track.consecutiveHits = track.consecutiveHits + 1;
track.missedFrames = 0;
track.lastFrameId = frameId;
track.lastDetectionId = detection.obstacleId;
track.lastBbox = bboxFromDetection(detection);
track.lastRiskLevel = estimateRiskLevel(track.state, detection, opts);

outputRow = buildOutputRow(track, detection, frameId, dt, opts);
end

function [track, outputRow] = createTrack(trackId, detection, frameId, opts)
state = [detection.centroidX; detection.centroidY; 0; 0];
covariance = diag([opts.MeasurementNoise^2, opts.MeasurementNoise^2, 10, 10]);

track = struct( ...
    "trackId", trackId, ...
    "state", state, ...
    "covariance", covariance, ...
    "age", 1, ...
    "totalHits", 1, ...
    "consecutiveHits", 1, ...
    "missedFrames", 0, ...
    "lastFrameId", frameId, ...
    "lastDetectionId", detection.obstacleId, ...
    "lastBbox", bboxFromDetection(detection), ...
    "lastRiskLevel", estimateRiskLevel(state, detection, opts));

outputRow = buildOutputRow(track, detection, frameId, opts.FramePeriod, opts);
end

function outputRow = buildOutputRow(track, detection, frameId, dt, opts)
state = track.state;
speed = hypot(state(3), state(4));
range = hypot(state(1), state(2));
bearing = atan2(state(2), state(1));
riskLevel = estimateRiskLevel(state, detection, opts);
ttc = estimateTTC(state, detection);

outputRow = struct( ...
    "frameId", frameId, ...
    "sourceFile", string(detection.sourceFile), ...
    "trackId", track.trackId, ...
    "detectionId", detection.obstacleId, ...
    "isConfirmed", track.totalHits >= opts.MinHitsToConfirm, ...
    "age", track.age, ...
    "totalHits", track.totalHits, ...
    "missedFrames", track.missedFrames, ...
    "x", state(1), ...
    "y", state(2), ...
    "vx", state(3), ...
    "vy", state(4), ...
    "speed", speed, ...
    "range", range, ...
    "bearing", bearing, ...
    "minX", detection.minX, ...
    "maxX", detection.maxX, ...
    "minY", detection.minY, ...
    "maxY", detection.maxY, ...
    "minZ", detection.minZ, ...
    "maxZ", detection.maxZ, ...
    "length", detection.length, ...
    "width", detection.width, ...
    "height", detection.height, ...
    "yaw", detection.yaw, ...
    "radius", detection.radius, ...
    "frontClearance", detection.frontClearance, ...
    "lateralClearance", detection.lateralClearance, ...
    "ttc", ttc, ...
    "riskLevel", riskLevel, ...
    "dt", dt);
end

function riskLevel = estimateRiskLevel(state, detection, opts)
range = hypot(state(1), state(2));
insideCorridor = abs(state(1)) <= opts.RiskCorridorHalfWidth || ...
    detection.lateralClearance <= opts.RiskCorridorHalfWidth;
inFront = detection.maxY >= 0;
ttc = estimateTTC(state, detection);

if insideCorridor && inFront && ...
        (range <= opts.RiskHighDistance || ttc <= opts.RiskHighTTC)
    riskLevel = "HIGH";
elseif inFront && ...
        (range <= opts.RiskMediumDistance || ttc <= opts.RiskMediumTTC)
    riskLevel = "MEDIUM";
else
    riskLevel = "LOW";
end
end

function ttc = estimateTTC(state, detection)
closingSpeed = -state(4);
if detection.maxY < 0 || closingSpeed <= 0.05
    ttc = inf;
else
    ttc = max(detection.frontClearance, 0) / closingSpeed;
end
end

function bbox = bboxFromDetection(detection)
bbox = [ ...
    detection.minX, detection.maxX, ...
    detection.minY, detection.maxY, ...
    detection.minZ, detection.maxZ];
end

function velocity = clampVelocity(velocity, maxSpeed)
speed = hypot(velocity(1), velocity(2));
if speed > maxSpeed
    velocity = velocity / speed * maxSpeed;
end
end

function trackHistory = appendTrackHistory(trackHistory, tracksToAppend)
if isempty(tracksToAppend)
    return;
end

if isempty(trackHistory)
    trackHistory = tracksToAppend;
else
    trackHistory(end + 1:end + numel(tracksToAppend)) = tracksToAppend;
end
end

function trackedTable = emptyTrackedTable()
trackedTable = table( ...
    zeros(0, 1), strings(0, 1), zeros(0, 1), zeros(0, 1), ...
    false(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), ...
    zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), ...
    zeros(0, 1), zeros(0, 1), zeros(0, 1), ...
    zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), ...
    zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), ...
    zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), ...
    zeros(0, 1), zeros(0, 1), strings(0, 1), zeros(0, 1), ...
    'VariableNames', { ...
        'frameId', 'sourceFile', 'trackId', 'detectionId', ...
        'isConfirmed', 'age', 'totalHits', 'missedFrames', ...
        'x', 'y', 'vx', 'vy', 'speed', 'range', 'bearing', ...
        'minX', 'maxX', 'minY', 'maxY', 'minZ', 'maxZ', ...
        'length', 'width', 'height', 'yaw', 'radius', ...
        'frontClearance', 'lateralClearance', 'ttc', ...
        'riskLevel', 'dt'});
end
