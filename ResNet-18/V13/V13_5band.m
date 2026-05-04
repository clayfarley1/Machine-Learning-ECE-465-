%% 5-Band Resistor Value Classifier (FAST GPU VERSION)

clear; clc; close all;

datasetPath = 'C:\AI\Data'; % UPDATE THIS PATH

[datasetDir, ~, ~] = fileparts(datasetPath);
if isempty(datasetDir), datasetDir = pwd; end

modelSavePath = fullfile(datasetDir, '5_Band_final');
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

%% Parallel Pool (important for throughput)

if isempty(gcp('nocreate'))
    parpool('threads'); % lightweight, fast startup
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
    error('No 5-Band images found!');
end

% === NEW: Limit to 100 images per label ===
labels = categories(imds_5band.Labels);
keepIdx = [];

for i = 1:numel(labels)
    idx = find(imds_5band.Labels == labels{i});
    
    if numel(idx) > 130
        idx = idx(randperm(numel(idx), 100));
    end
    
    keepIdx = [keepIdx; idx];
end

imds_5band = subset(imds_5band, keepIdx);
fprintf('Limited to max 100 images per class.\n');
% ==================================================

% Clean labels
cleanLabels = regexprep(string(imds_5band.Labels), '^5[Bb]-', '');
cleanLabels = regexprep(cleanLabels, '-\w+$', '');
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

%% FAST Data Pipeline (no augmentation, optimized IO)

augTrain = augmentedImageDatastore( ...
    inputSize(1:2), imdsTrain, ...
    'ColorPreprocessing', 'gray2rgb');

augVal = augmentedImageDatastore( ...
    inputSize(1:2), imdsVal, ...
    'ColorPreprocessing', 'gray2rgb');

augTrain.MiniBatchSize = 64;
augVal.MiniBatchSize   = 64;

%% Build Network

fprintf('\n=== Building Network ===\n');

net    = resnet18;
lgraph = layerGraph(net);

newLayers = [
    dropoutLayer(0.5, 'Name', 'dropout_5b')
    fullyConnectedLayer(numClasses, 'Name', 'fc_5band', ...
        'WeightLearnRateFactor', 10, ...
        'BiasLearnRateFactor', 10, ...
        'WeightL2Factor', 0.01)
    softmaxLayer('Name', 'softmax_5b')
    classificationLayer('Name', 'output_5b')
];

lgraph = removeLayers(lgraph, {'fc1000','prob','ClassificationLayer_predictions'});
lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'pool5', 'dropout_5b');

%% Training Options (optimized for throughput)

options = trainingOptions('adam', ...
    'InitialLearnRate',     0.0003, ...
    'LearnRateSchedule',    'piecewise', ...
    'LearnRateDropFactor',  0.3, ...
    'LearnRateDropPeriod',  10, ...
    'MaxEpochs',            60, ...
    'MiniBatchSize',        64, ...
    'ValidationData',       augVal, ...
    'ValidationFrequency',  50, ...
    'ValidationPatience',   Inf, ...
    'L2Regularization',     0.001, ...
    'Shuffle',              'every-epoch', ...
    'Verbose',              true, ...
    'Plots',                'training-progress', ...
    'ExecutionEnvironment', 'gpu', ...
    'DispatchInBackground', true);

%% Train

fprintf('\n=== Training 5-Band Value Classifier ===\n');
fprintf('Training images:   %d\n', numel(imdsTrain.Files));
fprintf('Validation images: %d\n\n', numel(imdsVal.Files));

tic;
[trainedNet, info] = trainNetwork(augTrain, lgraph, options);
trainingTime = toc;

%% Evaluate

YPred = classify(trainedNet, augVal, ...
    'ExecutionEnvironment', 'gpu');

YVal = imdsVal.Labels;

accuracy = sum(YPred == YVal) / numel(YVal);

fprintf('\n========================================\n');
fprintf('5-BAND TRAINING COMPLETE!\n');
fprintf('Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Time: %.2f minutes\n', trainingTime / 60);
fprintf('========================================\n');

%% Save

save(fullfile(modelSavePath, 'model_5band_values.mat'), ...
    'trainedNet', 'info', '-v7.3');

config.accuracy  = accuracy;
config.classes   = categories(imdsTrain.Labels);
config.inputSize = inputSize;

save(fullfile(modelSavePath, 'config_5band_values.mat'), 'config');

fprintf('✓ Saved to: %s\n', modelSavePath);