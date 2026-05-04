%% =========================
%% Full Resistor Prediction
%% Logic: Detect Band Type -> Predict Value
%% =========================
clear; clc; close all;

%% -------------------------
%% Paths to Models
%% -------------------------
bandTypeFolder = 'C:\AI\resistor_model_bandtype';
fourBandFolder = 'C:\AI\4_Band_final';
fiveBandFolder = 'C:\AI\5_Band_final';

%% -------------------------
%% 1. Load Band Type Model
%% -------------------------
fprintf('Loading band type model...\n');
try
    load(fullfile(bandTypeFolder, 'model_bandtype.mat'), 'trainedNet');
    bandTypeNet = trainedNet;
    inputSize = [224 224];
    fprintf('✓ Band type model loaded\n');
catch
    error('Could not load Band Type model. Check path: %s', bandTypeFolder);
end

%% -------------------------
%% 2. Load 4-Band Value Model
%% -------------------------
fprintf('Loading 4-Band value model...\n');
if isfile(fullfile(fourBandFolder, 'model_4band_values.mat'))
    load(fullfile(fourBandFolder, 'model_4band_values.mat'), 'trainedNet');
    net4Band = trainedNet;
    fprintf('✓ 4-Band model loaded\n');
else
    net4Band = [];
    warning('4-Band model not found at: %s', fourBandFolder);
end

%% -------------------------
%% 3. Load 5-Band Value Model
%% -------------------------
fprintf('Loading 5-Band value model...\n');
if isfile(fullfile(fiveBandFolder, 'model_5band_values.mat'))
    load(fullfile(fiveBandFolder, 'model_5band_values.mat'), 'trainedNet');
    net5Band = trainedNet;
    fprintf('✓ 5-Band model loaded\n');
else
    net5Band = [];
    warning('5-Band model not found!');
end

%% -------------------------
%% 4. Select Image
%% -------------------------
[fileName, pathName] = uigetfile({'*.jpg;*.png;*.jpeg;*.bmp','Image Files'}, ...
    'Select a Resistor Image');
if isequal(fileName, 0), return; end

imgRaw     = imread(fullfile(pathName, fileName));
imgResized = imresize(imgRaw, inputSize);

%% -------------------------
%% 5. Step 1: Detect Band Type
%% -------------------------
[bandPred, bandScores] = classify(bandTypeNet, imgResized);
bandType       = string(bandPred);
bandConfidence = max(bandScores) * 100;

fprintf('\n--- ANALYSIS START ---\n');
fprintf('Structure: %s (%.1f%% confidence)\n', bandType, bandConfidence);

%% -------------------------
%% 6. Step 2: Predict Value
%% -------------------------
targetNet = [];
if contains(bandType, '4', 'IgnoreCase', true)
    targetNet = net4Band;
elseif contains(bandType, '5', 'IgnoreCase', true)
    targetNet = net5Band;
end

if isempty(targetNet)
    error('Selected model for %s is not loaded!', bandType);
end

[valPred, valScores] = classify(targetNet, imgResized);
valueConfidence = max(valScores) * 100;
valueLabel      = string(valPred); % Labels are already clean e.g. "1k", "33k"

% Display Top 3
[sortedScores, idx] = sort(valScores, 'descend');
classNames = targetNet.Layers(end).Classes;
fprintf('Top Predictions:\n');
for i = 1:min(3, numel(classNames))
    fprintf('  %d. %s (%.1f%%)\n', i, string(classNames(idx(i))), sortedScores(i)*100);
end

%% -------------------------
%% 7. Display Result
%% -------------------------
figure('Name', 'Resistor AI Classifier', 'NumberTitle', 'off');
imshow(imgRaw);

displayColor = [0 0.5 0];
if valueConfidence < 70, displayColor = [0.8 0 0]; end

title(sprintf('Type: %s | Value: %s\nConfidence: %.1f%%', ...
    bandType, valueLabel, valueConfidence), ...
    'FontSize', 14, 'Color', displayColor);

fprintf('\n--- FINAL RESULT: %s ---\n', valueLabel);