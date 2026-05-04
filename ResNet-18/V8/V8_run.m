%% =========================
%% 4-Band Resistor Prediction Script
%% =========================
clear; clc; close all;

%% -------------------------
%% Path to saved model
%% -------------------------
modelFolder = 'C:\AI\resistor_model_4band_only'; % <--- update if needed
modelFile   = fullfile(modelFolder, 'model_4band_values.mat');
configFile  = fullfile(modelFolder, 'config_4band_values.mat');

%% -------------------------
%% Load Model
%% -------------------------
fprintf('Loading 4-Band value model...\n');
if ~isfile(modelFile)
    error('Model file not found: %s', modelFile);
end
load(modelFile, 'trainedNet');

if isfile(configFile)
    load(configFile, 'config');
    inputSize = config.inputSize(1:2);
else
    inputSize = [224 224];
    warning('Config file not found, using default input size 224x224.');
end

%% -------------------------
%% Select Resistor Image
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
%% Predict & Strip Prefix
%% -------------------------
[valPred, scores]  = classify(trainedNet, imgResized);
valueConfidence    = max(scores);

% Strip "4B-" or "4b-" prefix -> e.g. "4B-10K" becomes "10K"
valueLabel = regexprep(string(valPred), '^[45][Bb]-', '');

%% -------------------------
%% Display Result
%% -------------------------
fprintf('\nResistor Value: %s  (%.1f%% confidence)\n', valueLabel, valueConfidence*100);

figure;
imshow(img);
title(sprintf('%s  (%.1f%%)', valueLabel, valueConfidence*100), 'FontSize', 16);