%% 4-Band Resistor Value Classifier
% Trains ONLY on 4-Band resistors to identify their specific resistance values
% Folder names should be the resistor values e.g. "4B-10K", "4B-220R" etc.

clear; clc; close all;

%% ========================================
%%  CONFIGURATION
%% ========================================

datasetPath = 'C:\AI\data1'; % UPDATE THIS PATH

[datasetDir, ~, ~] = fileparts(datasetPath);
if isempty(datasetDir)
    datasetDir = pwd;
end

modelSavePath = fullfile(datasetDir, 'resistor_model_4band_only');
if ~exist(modelSavePath, 'dir')
    mkdir(modelSavePath);
end

inputSize = [224 224 3];

%% ========================================
%%  GPU SETUP
%% ========================================

parallel.gpu.enableCUDAForwardCompatibility(true);

if canUseGPU
    gpuInfo = gpuDevice;
    fprintf('✓ GPU: %s\n', gpuInfo.Name);
    reset(gpuInfo);
else
    error('GPU required for this training!');
end

%% ========================================
%%  LOAD & FILTER 4-BAND IMAGES ONLY
%% ========================================

fprintf('\n=== Loading Full Dataset ===\n');
imds_full = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

originalLabels = imds_full.Labels;
fprintf('Total images loaded: %d\n', numel(imds_full.Files));

%% Keep only 4-Band entries
fprintf('\n=== Filtering to 4-Band Only ===\n');

is4Band = false(numel(originalLabels), 1);
for i = 1:numel(originalLabels)
    labelStr = char(originalLabels(i));
    if startsWith(labelStr, '4B-', 'IgnoreCase', true) || startsWith(labelStr, '4b-', 'IgnoreCase', true)
        is4Band(i) = true;
    end
end

imds_4band = subset(imds_full, is4Band);

if numel(imds_4band.Files) == 0
    error('No 4-Band images found! Check your folder names start with "4B-"');
end

fprintf('4-Band images found: %d\n', numel(imds_4band.Files));
fprintf('4-Band classes found:\n');
disp(countEachLabel(imds_4band));

%% ========================================
%%  SPLIT DATASET
%% ========================================

[imdsTrain, imdsVal] = splitEachLabel(imds_4band, 0.85, 'randomized');

numClasses = numel(categories(imdsTrain.Labels));
fprintf('Number of value classes: %d\n', numClasses);

if numClasses < 2
    error('Need at least 2 resistor value classes to train!');
end

%% ========================================
%%  AUGMENTATION
%% ========================================

augmenter = imageDataAugmenter( ...
    'RandRotation',      [-20, 20], ...
    'RandXTranslation',  [-30 30], ...
    'RandYTranslation',  [-30 30], ...
    'RandXScale',        [0.8 1.2], ...
    'RandYScale',        [0.8 1.2], ...
    'RandXReflection',   true);

augTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', augmenter);
augVal = augmentedImageDatastore(inputSize(1:2), imdsVal);

%% ========================================
%%  BUILD NETWORK
%% ========================================

fprintf('\n=== Building 4-Band Value Network ===\n');

net = resnet18;
lgraph = layerGraph(net);

newLayers = [
    dropoutLayer(0.5, 'Name', 'dropout_4b')
    fullyConnectedLayer(numClasses, 'Name', 'fc_4band', ...
        'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'softmax_4b')
    classificationLayer('Name', 'output_4b')];

lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'pool5', 'dropout_4b');

%% ========================================
%%  TRAINING OPTIONS
%% ========================================

options = trainingOptions('adam', ...
    'InitialLearnRate',      0.0005, ...
    'LearnRateSchedule',     'piecewise', ...
    'LearnRateDropFactor',   0.3, ...
    'LearnRateDropPeriod',   15, ...
    'MaxEpochs',             40, ...
    'MiniBatchSize',         32, ...
    'ValidationData',        augVal, ...
    'ValidationFrequency',   20, ...
    'ValidationPatience',    Inf, ...
    'Shuffle',               'every-epoch', ...
    'Verbose',               true, ...
    'Plots',                 'training-progress', ...
    'ExecutionEnvironment',  'gpu');

%% ========================================
%%  TRAIN
%% ========================================

fprintf('\n=== Training 4-Band Value Classifier ===\n');
fprintf('Classes: %d resistor values\n', numClasses);
fprintf('Training images: %d\n', numel(imdsTrain.Files));
fprintf('Validation images: %d\n\n', numel(imdsVal.Files));

tic;
[trainedNet, info] = trainNetwork(augTrain, lgraph, options);
trainingTime = toc;

%% ========================================
%%  EVALUATE
%% ========================================

YPred = classify(trainedNet, augVal, 'ExecutionEnvironment', 'gpu');
YVal  = imdsVal.Labels;
accuracy = sum(YPred == YVal) / numel(YVal);

fprintf('\n========================================\n');
fprintf('TRAINING COMPLETE!\n');
fprintf('Value Classification Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Training Time: %.2f minutes\n', trainingTime / 60);
fprintf('========================================\n');

%% ========================================
%%  SAVE
%% ========================================

save(fullfile(modelSavePath, 'model_4band_values.mat'), ...
    'trainedNet', 'info', '-v7.3');

config.accuracy  = accuracy;
config.classes   = categories(imdsTrain.Labels);
config.inputSize = inputSize;

save(fullfile(modelSavePath, 'config_4band_values.mat'), 'config');

fprintf('\n✓ Model saved to: %s\n', modelSavePath);
fprintf('✓ Classes saved: %s\n', strjoin(config.classes, ', '));