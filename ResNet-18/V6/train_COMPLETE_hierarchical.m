%% COMPLETE Resistor Identification System - All Stages
% This script trains the complete hierarchical system:
% Stage 1: Band type (4-Band vs 5-Band vs Standard)
% Stage 2: Specific values within each band type
% All in one script!

clear; clc; close all;

%% ========================================
%%  CONFIGURATION
%% ========================================

datasetPath = 'C:\AI\full data'; % UPDATE THIS PATH

[datasetDir, ~, ~] = fileparts(datasetPath);
if isempty(datasetDir)
    datasetDir = pwd;
end
modelSavePath = fullfile(datasetDir, 'resistor_model_hierarchical');
if ~exist(modelSavePath, 'dir')
    mkdir(modelSavePath);
end

%% GPU Setup
parallel.gpu.enableCUDAForwardCompatibility(true);

if canUseGPU
    gpuInfo = gpuDevice;
    fprintf('✓ GPU: %s\n', gpuInfo.Name);
    reset(gpuInfo);
else
    error('GPU required for this training!');
end

%% ========================================
%%  STAGE 1: BAND TYPE CLASSIFICATION
%% ========================================

fprintf('\n');
fprintf('########################################\n');
fprintf('#                                      #\n');
fprintf('#   STAGE 1: BAND TYPE CLASSIFIER      #\n');
fprintf('#                                      #\n');
fprintf('########################################\n\n');

%% Load full dataset
fprintf('=== Loading Full Dataset ===\n');
imds_full = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

originalLabels = imds_full.Labels;
fprintf('Total images: %d\n', numel(imds_full.Files));
fprintf('Total classes: %d\n', numel(categories(originalLabels)));

%% Create hierarchical categories
fprintf('\n=== Creating Hierarchical Categories ===\n');

newLabelsCell = cell(numel(originalLabels), 1);

for i = 1:numel(originalLabels)
    labelStr = char(originalLabels(i));
    
    if startsWith(labelStr, '4B-', 'IgnoreCase', true) || startsWith(labelStr, '4b-', 'IgnoreCase', true)
        newLabelsCell{i} = '4Band';
    elseif startsWith(labelStr, '5B-', 'IgnoreCase', true) || startsWith(labelStr, '5b-', 'IgnoreCase', true)
        newLabelsCell{i} = '5Band';
    else
        newLabelsCell{i} = 'Standard';
    end
end

imds_stage1 = imds_full;
imds_stage1.Labels = categorical(newLabelsCell);

labelCounts_stage1 = countEachLabel(imds_stage1);
numClasses_stage1 = height(labelCounts_stage1);

fprintf('Stage 1 categories:\n');
disp(labelCounts_stage1);

%% Split and augment
[imdsTrain_s1, imdsVal_s1] = splitEachLabel(imds_stage1, 0.85, 'randomized');

augmenter = imageDataAugmenter( ...
    'RandRotation', [-20, 20], ...
    'RandXTranslation', [-30 30], ...
    'RandYTranslation', [-30 30], ...
    'RandXScale', [0.8 1.2], ...
    'RandYScale', [0.8 1.2], ...
    'RandXReflection', true);

inputSize = [224 224 3];

augTrain_s1 = augmentedImageDatastore(inputSize(1:2), imdsTrain_s1, ...
    'DataAugmentation', augmenter);
augVal_s1 = augmentedImageDatastore(inputSize(1:2), imdsVal_s1);

%% Build Stage 1 network
fprintf('\n=== Building Stage 1 Network ===\n');

net_s1 = resnet18;
lgraph_s1 = layerGraph(net_s1);

layers_s1 = [
    fullyConnectedLayer(numClasses_stage1, 'Name', 'fc_stage1', ...
        'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'softmax_s1')
    classificationLayer('Name', 'output_s1')];

lgraph_s1 = replaceLayer(lgraph_s1, 'fc1000', layers_s1(1));
lgraph_s1 = replaceLayer(lgraph_s1, 'prob', layers_s1(2));
lgraph_s1 = replaceLayer(lgraph_s1, 'ClassificationLayer_predictions', layers_s1(3));

%% Train Stage 1
options_s1 = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 32, ...
    'ValidationData', augVal_s1, ...
    'ValidationFrequency', 10, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'gpu');

fprintf('\n=== Training Stage 1 ===\n');
fprintf('Expected: 75-95%% accuracy\n\n');

tic;
[trainedNet_s1, info_s1] = trainNetwork(augTrain_s1, lgraph_s1, options_s1);
time_s1 = toc;

%% Evaluate Stage 1
YPred_s1 = classify(trainedNet_s1, augVal_s1, 'ExecutionEnvironment', 'gpu');
YVal_s1 = imdsVal_s1.Labels;
accuracy_s1 = sum(YPred_s1 == YVal_s1) / numel(YVal_s1);

fprintf('\n========================================\n');
fprintf('STAGE 1 COMPLETE!\n');
fprintf('Accuracy: %.2f%%\n', accuracy_s1 * 100);
fprintf('Time: %.2f minutes\n', time_s1/60);
fprintf('========================================\n');

% Save Stage 1
save(fullfile(modelSavePath, 'stage1_band_type.mat'), 'trainedNet_s1', 'info_s1', '-v7.3');
config_s1.accuracy = accuracy_s1;
config_s1.classes = categories(imdsTrain_s1.Labels);
save(fullfile(modelSavePath, 'stage1_config.mat'), 'config_s1');

pause(2);

%% ========================================
%%  STAGE 2a: 4-BAND VALUE CLASSIFICATION
%% ========================================

fprintf('\n');
fprintf('########################################\n');
fprintf('#                                      #\n');
fprintf('#   STAGE 2a: 4-BAND VALUES            #\n');
fprintf('#                                      #\n');
fprintf('########################################\n\n');

%% Filter to 4-Band only
fprintf('=== Filtering 4-Band Resistors ===\n');

is4Band = false(numel(originalLabels), 1);
for i = 1:numel(originalLabels)
    labelStr = char(originalLabels(i));
    if startsWith(labelStr, '4B-', 'IgnoreCase', true) || startsWith(labelStr, '4b-', 'IgnoreCase', true)
        is4Band(i) = true;
    end
end

imds_4band = subset(imds_full, is4Band);

if numel(imds_4band.Files) == 0
    fprintf('⚠ No 4-Band resistors found! Skipping Stage 2a.\n');
    trainedNet_s2a = [];
else
    labelCounts_4b = countEachLabel(imds_4band);
    numClasses_4b = height(labelCounts_4b);
    
    fprintf('4-Band images: %d\n', numel(imds_4band.Files));
    fprintf('4-Band classes: %d\n', numClasses_4b);
    
    %% Split and augment
    [imdsTrain_4b, imdsVal_4b] = splitEachLabel(imds_4band, 0.85, 'randomized');
    
    augTrain_4b = augmentedImageDatastore(inputSize(1:2), imdsTrain_4b, ...
        'DataAugmentation', augmenter);
    augVal_4b = augmentedImageDatastore(inputSize(1:2), imdsVal_4b);
    
 %% Build 4-Band network
fprintf('\n=== Building 4-Band Network ===\n');

net_4b = resnet18;
lgraph_4b = layerGraph(net_4b);

% 1. Define the NEW layer stack (Dropout -> FC -> Softmax -> Output)
newLayers_4b = [
    dropoutLayer(0.5, 'Name', 'dropout_4b')
    fullyConnectedLayer(numClasses_4b, 'Name', 'fc_4band', ...
        'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'softmax_4b')
    classificationLayer('Name', 'output_4b')];

% 2. REMOVE the old final layers
% In ResNet-18, the features come from 'pool5'. We remove everything after it.
lgraph_4b = removeLayers(lgraph_4b, {'fc1000', 'prob', 'ClassificationLayer_predictions'});

% 3. ADD the new layer stack to the graph
lgraph_4b = addLayers(lgraph_4b, newLayers_4b);

% 4. CONNECT 'pool5' (the last kept layer) to 'dropout_4b' (the start of your new layers)
lgraph_4b = connectLayers(lgraph_4b, 'pool5', 'dropout_4b');

% Verify
fprintf('Network modified: pool5 -> dropout_4b -> fc_4band (%d classes) -> softmax -> output\n', numClasses_4b);


    %% Train 4-Band
    options_4b = trainingOptions('adam', ...
        'InitialLearnRate', 0.0005, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.3, ...
        'LearnRateDropPeriod', 15, ...
        'MaxEpochs', 40, ...
        'MiniBatchSize', 32, ...
        'ValidationData', augVal_4b, ...
        'ValidationFrequency', 20, ...
        'ValidationPatience', Inf, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', true, ...
        'Plots', 'training-progress', ...
        'ExecutionEnvironment', 'gpu');
    
    fprintf('\n=== Training 4-Band Classifier ===\n');
    fprintf('Expected: 50-75%% accuracy\n\n');
    
    tic;
    [trainedNet_s2a, info_s2a] = trainNetwork(augTrain_4b, lgraph_4b, options_4b);
    time_s2a = toc;
    
    %% Evaluate 4-Band
    YPred_4b = classify(trainedNet_s2a, augVal_4b, 'ExecutionEnvironment', 'gpu');
    YVal_4b = imdsVal_4b.Labels;
    accuracy_s2a = sum(YPred_4b == YVal_4b) / numel(YVal_4b);
    
    % Top-5 accuracy
    [~, scores_4b] = classify(trainedNet_s2a, augVal_4b, 'ExecutionEnvironment', 'gpu');
    [~, sortedIdx_4b] = sort(scores_4b, 2, 'descend');
    classNames_4b = categories(imdsTrain_4b.Labels);
    top5_4b = 0;
    for i = 1:numel(YVal_4b)
        classIdx = find(strcmp(classNames_4b, char(YVal_4b(i))));
        if any(sortedIdx_4b(i, 1:5) == classIdx)
            top5_4b = top5_4b + 1;
        end
    end
    top5acc_s2a = top5_4b / numel(YVal_4b) * 100;
    
    fprintf('\n========================================\n');
    fprintf('STAGE 2a (4-BAND) COMPLETE!\n');
    fprintf('Top-1 Accuracy: %.2f%%\n', accuracy_s2a * 100);
    fprintf('Top-5 Accuracy: %.2f%%\n', top5acc_s2a);
    fprintf('Time: %.2f minutes\n', time_s2a/60);
    fprintf('========================================\n');
    
    % Save 4-Band model
    save(fullfile(modelSavePath, 'stage2a_4band.mat'), 'trainedNet_s2a', 'info_s2a', '-v7.3');
    config_s2a.accuracy = accuracy_s2a;
    config_s2a.top5accuracy = top5acc_s2a;
    config_s2a.classes = categories(imdsTrain_4b.Labels);
    save(fullfile(modelSavePath, 'stage2a_config.mat'), 'config_s2a');
end

pause(2);

%% ========================================
%%  STAGE 2b: 5-BAND VALUE CLASSIFICATION
%% ========================================

fprintf('\n');
fprintf('########################################\n');
fprintf('#                                      #\n');
fprintf('#   STAGE 2b: 5-BAND VALUES            #\n');
fprintf('#                                      #\n');
fprintf('########################################\n\n');

%% Filter to 5-Band only
fprintf('=== Filtering 5-Band Resistors ===\n');

is5Band = false(numel(originalLabels), 1);
for i = 1:numel(originalLabels)
    labelStr = char(originalLabels(i));
    if startsWith(labelStr, '5B-', 'IgnoreCase', true) || startsWith(labelStr, '5b-', 'IgnoreCase', true)
        is5Band(i) = true;
    end
end

imds_5band = subset(imds_full, is5Band);

if numel(imds_5band.Files) == 0
    fprintf('⚠ No 5-Band resistors found! Skipping Stage 2b.\n');
    trainedNet_s2b = [];
else
    labelCounts_5b = countEachLabel(imds_5band);
    numClasses_5b = height(labelCounts_5b);
    
    fprintf('5-Band images: %d\n', numel(imds_5band.Files));
    fprintf('5-Band classes: %d\n', numClasses_5b);
    
    %% Split and augment
    [imdsTrain_5b, imdsVal_5b] = splitEachLabel(imds_5band, 0.85, 'randomized');
    
    augTrain_5b = augmentedImageDatastore(inputSize(1:2), imdsTrain_5b, ...
        'DataAugmentation', augmenter);
    augVal_5b = augmentedImageDatastore(inputSize(1:2), imdsVal_5b);

    %% Build 5-Band network
    fprintf('\n=== Building 5-Band Network ===\n');
    
    net_5b = resnet18;
    lgraph_5b = layerGraph(net_5b);
    
    % 1. Define the NEW layer stack
    newLayers_5b = [
        dropoutLayer(0.5, 'Name', 'dropout_5b')
        fullyConnectedLayer(numClasses_5b, 'Name', 'fc_5band', ...
            'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
        softmaxLayer('Name', 'softmax_5b')
        classificationLayer('Name', 'output_5b')];
    
    % 2. REMOVE the old final layers
    lgraph_5b = removeLayers(lgraph_5b, {'fc1000', 'prob', 'ClassificationLayer_predictions'});
    
    % 3. ADD the new layer stack
    lgraph_5b = addLayers(lgraph_5b, newLayers_5b);
    
    % 4. CONNECT the feature extractor ('pool5') to your new stack
    lgraph_5b = connectLayers(lgraph_5b, 'pool5', 'dropout_5b');
    
    %% Train 5-Band
    options_5b = trainingOptions('adam', ...
        'InitialLearnRate', 0.0005, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.3, ...
        'LearnRateDropPeriod', 15, ...
        'MaxEpochs', 40, ...
        'MiniBatchSize', 32, ...
        'ValidationData', augVal_5b, ...
        'ValidationFrequency', 20, ...
        'ValidationPatience', Inf, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', true, ...
        'Plots', 'training-progress', ...
        'ExecutionEnvironment', 'gpu');
    
    fprintf('\n=== Training 5-Band Classifier ===\n');
    fprintf('Expected: 50-75%% accuracy\n\n');
    
    tic;
    [trainedNet_s2b, info_s2b] = trainNetwork(augTrain_5b, lgraph_5b, options_5b);
    time_s2b = toc;
    
    %% Evaluate 5-Band
    YPred_5b = classify(trainedNet_s2b, augVal_5b, 'ExecutionEnvironment', 'gpu');
    YVal_5b = imdsVal_5b.Labels;
    accuracy_s2b = sum(YPred_5b == YVal_5b) / numel(YVal_5b);
    
    % Top-5 accuracy
    [~, scores_5b] = classify(trainedNet_s2b, augVal_5b, 'ExecutionEnvironment', 'gpu');
    [~, sortedIdx_5b] = sort(scores_5b, 2, 'descend');
    classNames_5b = categories(imdsTrain_5b.Labels);
    top5_5b = 0;
    for i = 1:numel(YVal_5b)
        classIdx = find(strcmp(classNames_5b, char(YVal_5b(i))));
        if any(sortedIdx_5b(i, 1:5) == classIdx)
            top5_5b = top5_5b + 1;
        end
    end
    top5acc_s2b = top5_5b / numel(YVal_5b) * 100;
    
    fprintf('\n========================================\n');
    fprintf('STAGE 2b (5-BAND) COMPLETE!\n');
    fprintf('Top-1 Accuracy: %.2f%%\n', accuracy_s2b * 100);
    fprintf('Top-5 Accuracy: %.2f%%\n', top5acc_s2b);
    fprintf('Time: %.2f minutes\n', time_s2b/60);
    fprintf('========================================\n');
    
    % Save 5-Band model
    save(fullfile(modelSavePath, 'stage2b_5band.mat'), 'trainedNet_s2b', 'info_s2b', '-v7.3');
    config_s2b.accuracy = accuracy_s2b;
    config_s2b.top5accuracy = top5acc_s2b;
    config_s2b.classes = categories(imdsTrain_5b.Labels);
    save(fullfile(modelSavePath, 'stage2b_config.mat'), 'config_s2b');
end

pause(2);

%% ========================================
%%  FINAL SUMMARY
%% ========================================

fprintf('\n\n');
fprintf('########################################\n');
fprintf('#                                      #\n');
fprintf('#      TRAINING COMPLETE!              #\n');
fprintf('#                                      #\n');
fprintf('########################################\n\n');

fprintf('Models saved to: %s\n\n', modelSavePath);

fprintf('========================================\n');
fprintf('STAGE 1 (Band Type):\n');
fprintf('  Accuracy: %.1f%%\n', accuracy_s1 * 100);
fprintf('  Time: %.1f min\n', time_s1/60);
fprintf('========================================\n\n');

if exist('accuracy_s2a', 'var')
    fprintf('========================================\n');
    fprintf('STAGE 2a (4-Band Values):\n');
    fprintf('  Top-1: %.1f%%\n', accuracy_s2a * 100);
    fprintf('  Top-5: %.1f%%\n', top5acc_s2a);
    fprintf('  Time: %.1f min\n', time_s2a/60);
    fprintf('========================================\n\n');
end

if exist('accuracy_s2b', 'var')
    fprintf('========================================\n');
    fprintf('STAGE 2b (5-Band Values):\n');
    fprintf('  Top-1: %.1f%%\n', accuracy_s2b * 100);
    fprintf('  Top-5: %.1f%%\n', top5acc_s2b);
    fprintf('  Time: %.1f min\n', time_s2b/60);
    fprintf('========================================\n\n');
end

totalTime = time_s1;
if exist('time_s2a', 'var'), totalTime = totalTime + time_s2a; end
if exist('time_s2b', 'var'), totalTime = totalTime + time_s2b; end

fprintf('========================================\n');
fprintf('TOTAL TRAINING TIME: %.1f minutes\n', totalTime/60);
fprintf('========================================\n\n');

fprintf('Next step: Use predict_hierarchical.m to test!\n\n');


