%% =========================
%% Full Resistor Prediction
%% Uses: Band Type -> 4-Band or 5-Band Value Model
%% =========================
clear; clc; close all;

%% -------------------------
%% Paths to all 3 models
%% -------------------------
bandTypeFolder = 'C:\AI\resistor_model_bandtype';     % UPDATE IF NEEDED
fourBandFolder = 'C:\AI\resistor_model_4band_only';   % UPDATE IF NEEDED
fiveBandFolder = 'C:\AI\resistor_model_5band_only';   % UPDATE IF NEEDED

%% -------------------------
%% Load Band Type Model
%% -------------------------
fprintf('Loading band type model...\n');
if ~isfile(fullfile(bandTypeFolder, 'model_bandtype.mat'))
    error('Band type model not found at: %s', bandTypeFolder);
end
load(fullfile(bandTypeFolder, 'model_bandtype.mat'), 'trainedNet');
bandTypeNet = trainedNet;

if isfile(fullfile(bandTypeFolder, 'config_bandtype.mat'))
    load(fullfile(bandTypeFolder, 'config_bandtype.mat'), 'config');
    inputSize = config.inputSize(1:2);
else
    inputSize = [224 224];
    warning('Band type config not found, using default 224x224.');
end

%% -------------------------
%% Load 4-Band Value Model
%% -------------------------
fprintf('Loading 4-Band value model...\n');
if isfile(fullfile(fourBandFolder, 'model_4band_values.mat'))
    load(fullfile(fourBandFolder, 'model_4band_values.mat'), 'trainedNet');
    net4Band = trainedNet;
    fprintf('✓ 4-Band model loaded\n');
else
    net4Band = [];
    warning('4-Band model not found!');
end

%% -------------------------
%% Load 5-Band Value Model
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
%% Select Image
%% -------------------------
[fileName, pathName] = uigetfile({'*.jpg;*.png;*.jpeg;*.bmp','Image Files'}, ...
    'Select a Resistor Image');

if isequal(fileName, 0)
    disp('No file selected. Exiting.');
    return;
end

imgPath    = fullfile(pathName, fileName);
img        = imread(imgPath);
imgResized = imresize(img, inputSize);

%% -------------------------
%% Step 1: Detect Band Type
%% -------------------------
[bandPred, bandScores] = classify(bandTypeNet, imgResized);
bandType       = string(bandPred);
bandConfidence = max(bandScores);

fprintf('\n--- Step 1: Band Type ---\n');
fprintf('Detected: %s  (%.1f%% confidence)\n', bandType, bandConfidence * 100);

%% -------------------------
%% Step 2: Predict Value
%% -------------------------
fprintf('\n--- Step 2: Resistor Value ---\n');

switch bandType
    case '4Band'
        if isempty(net4Band)
            error('4-Band value model not loaded!');
        end
        [valPred, valScores] = classify(net4Band, imgResized);

    case '5Band'
        if isempty(net5Band)
            error('5-Band value model not loaded!');
        end
        [valPred, valScores] = classify(net5Band, imgResized);

    otherwise
        error('Unknown band type predicted: %s', bandType);
end

valueConfidence = max(valScores);

% Strip prefix e.g. "4B-10K" -> "10K"
valueLabel = regexprep(string(valPred), '^[45][Bb]-', '');

fprintf('Resistor Value: %s  (%.1f%% confidence)\n', valueLabel, valueConfidence * 100);

%% -------------------------
%% Display Result
%% -------------------------
figure;
imshow(img);
title(sprintf('%s  |  %s  (%.1f%%)', bandType, valueLabel, valueConfidence * 100), ...
    'FontSize', 16);