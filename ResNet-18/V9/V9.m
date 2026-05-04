%% 4-Band Resistor Value Classifier - HIGH ACCURACY VERSION
clear; clc; close all;

datasetPath = 'C:\AI\data1'; % UPDATE THIS PATH
[datasetDir, ~, ~] = fileparts(datasetPath);
if isempty(datasetDir), datasetDir = pwd; end
modelSavePath = fullfile(datasetDir, 'resistor_model_high_acc');
if ~exist(modelSavePath, 'dir'), mkdir(modelSavePath); end

% Increased input size for better band detail
inputSize = [224 224 3];

%% GPU Setup
parallel.gpu.enableCUDAForwardCompatibility(true);
if canUseGPU
    gpuInfo = gpuDevice;
    fprintf('✓ GPU: %s\n', gpuInfo.Name);
    reset(gpuInfo);
else
    error('GPU required for training.');
end

%% 1. Load, Filter, and Advanced Label Cleaning
fprintf('\n=== Loading & Balancing Dataset ===\n');
imds_full = imageDatastore(datasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Filter 4-Band only
is4Band = startsWith(cellstr(imds_full.Labels), '4B-', 'IgnoreCase', true);
imds_4band = subset(imds_full, is4Band);

% Clean Labels: 4B-1K2-T5 -> 1K | 4B-27R-T5 -> 27
rawLabels = cellstr(imds_4band.Labels);
cleanedLabels = regexp(rawLabels, '(?<=-)(.*?)(?=-)', 'match', 'once');
cleanedLabels = regexprep(cleanedLabels, '[rR]', ''); % Remove R
cleanedLabels = regexprep(cleanedLabels, '([kK])\d+', '$1'); % 1K2 -> 1K
imds_4band.Labels = categorical(cleanedLabels);

% --- ACCURACY BOOST: Class Balancing ---
% This ensures the model doesn't favor the most common resistor type
minSetCount = min(countEachLabel(imds_4band).Count); 
% Use up to 200 images per class, or the max available
imds_balanced = splitEachLabel(imds_4band, min(200, minSetCount), 'randomize'); 

[imdsTrain, imdsVal] = splitEachLabel(imds_balanced, 0.8, 'randomized');
numClasses = numel(categories(imdsTrain.Labels));

%% 2. Advanced Augmentation (L*a*b* & Contrast Stretching)
% We use a transform function to apply color-science based logic
augTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', imageDataAugmenter('RandXReflection', true, 'RandRotation', [-90 90]));
augTrain = transform(augTrain, @highAccuracyAugment);

augVal = augmentedImageDatastore(inputSize(1:2), imdsVal);

%% 3. Build Deeper Network (ResNet-50)
fprintf('\n=== Building ResNet-50 Network ===\n');
net = resnet50; 
lgraph = layerGraph(net);

% Add Dropout to prevent overfitting on specific backgrounds
newLayers = [
    dropoutLayer(0.4, 'Name', 'high_acc_dropout')
    fullyConnectedLayer(numClasses, 'Name', 'fc_new', 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
    softmaxLayer('Name', 'softmax_new')
    classificationLayer('Name', 'output_new')];

lgraph = removeLayers(lgraph, {'fc1000','fc1000_softmax','ClassificationLayer_fc1000'});
lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'avg_pool', 'high_acc_dropout');

%% 4. High-Accuracy Training Options
options = trainingOptions('adam', ...
    'InitialLearnRate',     1e-4, ... % Slower learning for deeper net
    'LearnRateSchedule',    'piecewise', ...
    'LearnRateDropFactor',  0.5, ...
    'LearnRateDropPeriod',  10, ...
    'MaxEpochs',            40, ...
    'MiniBatchSize',        16, ... % Small batch for better generalization
    'ValidationData',       augVal, ...
    'ValidationFrequency',  30, ...
    'Shuffle',              'every-epoch', ...
    'Plots',                'training-progress', ...
    'ExecutionEnvironment', 'gpu');

%% 5. Train
[trainedNet, info] = trainNetwork(augTrain, lgraph, options);

%% 6. Save Result
save(fullfile(modelSavePath, 'high_acc_resistor_model.mat'), 'trainedNet', 'info');
fprintf('✓ Training Complete. Model saved.\n');

%% ========================================
%% ACCURACY ENHANCEMENT FUNCTIONS
%% ========================================
function dataOut = highAccuracyAugment(dataIn)
    img = single(dataIn.input{1});
    if max(img(:)) > 1, img = img / 255; end

    % --- 1. Contrast Stretching (Makes bands pop) ---
    img = imadjust(img, stretchlim(img), []);

    % --- 2. L*a*b* Color Shift ---
    % Helps model recognize "Brown" or "Red" even in yellowish light
    if rand > 0.5
        lab = rgb2lab(double(img));
        lab(:,:,2) = lab(:,:,2) + (rand-0.5)*5; % Adjust Red-Green axis
        lab(:,:,3) = lab(:,:,3) + (rand-0.5)*5; % Adjust Blue-Yellow axis
        img = single(lab2rgb(lab));
    end

    % --- 3. Random Sharpness/Blur ---
    if rand > 0.5
        img = imsharpen(img, 'Radius', 1, 'Amount', 1);
    else
        img = imgaussfilt(img, 0.5 + rand);
    end

    dataOut = dataIn;
    dataOut.input{1} = max(0, min(1, img));
end