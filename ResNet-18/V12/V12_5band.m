%% 5-Band Resistor Value Classifier
clear; clc; close all;

datasetPath = 'C:\AI\data1'; % UPDATE THIS PATH

[datasetDir, ~, ~] = fileparts(datasetPath);
if isempty(datasetDir), datasetDir = pwd; end

modelSavePath = fullfile(datasetDir, 'resistor_model_5band_only');
if ~exist(modelSavePath, 'dir'), mkdir(modelSavePath); end

inputSize = [224 224 3];

%% GPU
parallel.gpu.enableCUDAForwardCompatibility(true);
if canUseGPU
    gpuInfo = gpuDevice;
    fprintf('✓ GPU: %s\n', gpuInfo.Name);
    reset(gpuInfo);
else
    error('GPU required!');
end

%% Load & Filter
fprintf('\n=== Loading Dataset ===\n');
imds_full = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

originalLabels = imds_full.Labels;
fprintf('Total images: %d\n', numel(imds_full.Files));

% Keep only 5B- folders
is5Band = startsWith(string(originalLabels), '5B-', 'IgnoreCase', true);
imds_5band = subset(imds_full, is5Band);

if numel(imds_5band.Files) == 0
    error('No 5-Band images found! Make sure folders are prefixed with "5B-"');
end

% Strip the 5B- prefix AND any trailing -XX suffix e.g. "5B-4R7-T5" -> "4R7"
cleanLabels = regexprep(string(imds_5band.Labels), '^5[Bb]-', '');  % remove 5B-
cleanLabels = regexprep(cleanLabels, '-\w+$', '');                   % remove -T5 etc.
imds_5band.Labels = categorical(cleanLabels);

fprintf('5-Band images: %d\n', numel(imds_5band.Files));
disp(countEachLabel(imds_5band));

%% Split
[imdsTrain, imdsVal] = splitEachLabel(imds_5band, 0.85, 'randomized');
numClasses = numel(categories(imdsTrain.Labels));
fprintf('Value classes: %d\n', numClasses);

if numClasses < 2
    error('Need at least 2 classes!');
end

%% Augmentation Pipeline
spatialAugmenter = imageDataAugmenter( ...
    'RandRotation',     [-2, 2], ...
    'RandXTranslation', [-3 3], ...
    'RandYTranslation', [-3 3], ...
    'RandXScale',       [0.99 1.01], ...
    'RandYScale',       [0.99 1.01], ...
    'RandXReflection',  true);

augTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', spatialAugmenter);
augVal   = augmentedImageDatastore(inputSize(1:2), imdsVal);

%% Build Network
fprintf('\n=== Building Network ===\n');

net    = resnet18;
lgraph = layerGraph(net);

newLayers = [
    dropoutLayer(0.5, 'Name', 'dropout_5b')
    fullyConnectedLayer(numClasses, 'Name', 'fc_5band', ...
        'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10, ...
        'WeightL2Factor', 0.01)
    softmaxLayer('Name', 'softmax_5b')
    classificationLayer('Name', 'output_5b')];

lgraph = removeLayers(lgraph, {'fc1000','prob','ClassificationLayer_predictions'});
lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'pool5', 'dropout_5b');

%% Training Options
options = trainingOptions('adam', ...
    'InitialLearnRate',     0.0003, ...
    'LearnRateSchedule',    'piecewise', ...
    'LearnRateDropFactor',  0.3, ...
    'LearnRateDropPeriod',  10, ...
    'MaxEpochs',            70, ...
    'MiniBatchSize',        16, ...
    'ValidationData',       augVal, ...
    'ValidationFrequency',  20, ...
    'ValidationPatience',   Inf, ...
    'L2Regularization',     0.001, ...
    'Shuffle',              'every-epoch', ...
    'Verbose',              true, ...
    'Plots',                'training-progress', ...
    'ExecutionEnvironment', 'gpu');

%% Train
fprintf('\n=== Training 5-Band Value Classifier ===\n');
fprintf('Training images:   %d\n', numel(imdsTrain.Files));
fprintf('Validation images: %d\n\n', numel(imdsVal.Files));

tic;
[trainedNet, info] = trainNetwork(augTrain, lgraph, options);
trainingTime = toc;

%% Evaluate
YPred    = classify(trainedNet, augVal, 'ExecutionEnvironment', 'gpu');
YVal     = imdsVal.Labels;
accuracy = sum(YPred == YVal) / numel(YVal);

fprintf('\n========================================\n');
fprintf('5-BAND TRAINING COMPLETE!\n');
fprintf('Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Time: %.2f minutes\n', trainingTime / 60);
fprintf('========================================\n');

%% Save
save(fullfile(modelSavePath, 'model_5band_values.mat'), 'trainedNet', 'info', '-v7.3');
config.accuracy  = accuracy;
config.classes   = categories(imdsTrain.Labels);
config.inputSize = inputSize;
save(fullfile(modelSavePath, 'config_5band_values.mat'), 'config');
fprintf('✓ Saved to: %s\n', modelSavePath);