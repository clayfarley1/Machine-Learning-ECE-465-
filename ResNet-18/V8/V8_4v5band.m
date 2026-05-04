%% Band Type Identifier - Trains to detect 4-Band vs 5-Band
clear; clc; close all;

%% ========================================
%%  CONFIGURATION
%% ========================================

datasetPath = 'C:\AI\data1'; % UPDATE THIS PATH

[datasetDir, ~, ~] = fileparts(datasetPath);
if isempty(datasetDir)
    datasetDir = pwd;
end

modelSavePath = fullfile(datasetDir, 'resistor_model_bandtype');
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
%%  LOAD DATASET & REMAP LABELS
%% ========================================

fprintf('\n=== Loading Full Dataset ===\n');
imds_full = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

originalLabels = imds_full.Labels;
fprintf('Total images loaded: %d\n', numel(imds_full.Files));

%% Remap all folder labels to just "4Band" or "5Band"
fprintf('\n=== Remapping Labels to Band Type ===\n');

newLabelsCell = cell(numel(originalLabels), 1);
validMask = false(numel(originalLabels), 1);

for i = 1:numel(originalLabels)
    labelStr = char(originalLabels(i));
    if startsWith(labelStr, '4B-', 'IgnoreCase', true)
        newLabelsCell{i} = '4Band';
        validMask(i) = true;
    elseif startsWith(labelStr, '5B-', 'IgnoreCase', true)
        newLabelsCell{i} = '5Band';
        validMask(i) = true;
    end
end

%% Keep only 4-Band and 5-Band images
imds_filtered = subset(imds_full, validMask);
imds_filtered.Labels = categorical(newLabelsCell(validMask));

if numel(imds_filtered.Files) == 0
    error('No 4B- or 5B- labeled images found! Check your folder names.');
end

fprintf('Images after filtering: %d\n', numel(imds_filtered.Files));
fprintf('Band type counts:\n');
disp(countEachLabel(imds_filtered));

%% ========================================
%%  SPLIT DATASET
%% ========================================

[imdsTrain, imdsVal] = splitEachLabel(imds_filtered, 0.85, 'randomized');

numClasses = numel(categories(imdsTrain.Labels));
fprintf('Classes: %d\n', numClasses);

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

fprintf('\n=== Building Band Type Network ===\n');

net    = resnet18;
lgraph = layerGraph(net);

newLayers = [
    dropoutLayer(0.3, 'Name', 'dropout_bt')
    fullyConnectedLayer(numClasses, 'Name', 'fc_bandtype', ...
        'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'softmax_bt')
    classificationLayer('Name', 'output_bt')];

lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'pool5', 'dropout_bt');

%% ========================================
%%  TRAINING OPTIONS
%% ========================================

options = trainingOptions('adam', ...
    'InitialLearnRate',      0.001, ...
    'LearnRateSchedule',     'piecewise', ...
    'LearnRateDropFactor',   0.3, ...
    'LearnRateDropPeriod',   10, ...
    'MaxEpochs',             20, ...
    'MiniBatchSize',         32, ...
    'ValidationData',        augVal, ...
    'ValidationFrequency',   10, ...
    'ValidationPatience',    Inf, ...
    'Shuffle',               'every-epoch', ...
    'Verbose',               true, ...
    'Plots',                 'training-progress', ...
    'ExecutionEnvironment',  'gpu');

%% ========================================
%%  TRAIN
%% ========================================

fprintf('\n=== Training Band Type Classifier ===\n');
fprintf('Training images: %d\n', numel(imdsTrain.Files));
fprintf('Validation images: %d\n\n', numel(imdsVal.Files));

tic;
[trainedNet, info] = trainNetwork(augTrain, lgraph, options);
trainingTime = toc;

%% ========================================
%%  EVALUATE
%% ========================================

YPred    = classify(trainedNet, augVal, 'ExecutionEnvironment', 'gpu');
YVal     = imdsVal.Labels;
accuracy = sum(YPred == YVal) / numel(YVal);

fprintf('\n========================================\n');
fprintf('BAND TYPE TRAINING COMPLETE!\n');
fprintf('Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Training Time: %.2f minutes\n', trainingTime / 60);
fprintf('========================================\n');

%% ========================================
%%  SAVE
%% ========================================

save(fullfile(modelSavePath, 'model_bandtype.mat'), ...
    'trainedNet', 'info', '-v7.3');

config.accuracy  = accuracy;
config.classes   = categories(imdsTrain.Labels);
config.inputSize = inputSize;

save(fullfile(modelSavePath, 'config_bandtype.mat'), 'config');

fprintf('\n✓ Model saved to: %s\n', modelSavePath);
fprintf('✓ Classes: %s\n', strjoin(config.classes, ', '));