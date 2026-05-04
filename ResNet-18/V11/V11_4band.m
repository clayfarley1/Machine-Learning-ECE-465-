%% 4-Band Resistor Value Classifier — Custom CNN
%  Architecture: 4x Conv blocks (BN + ReLU + MaxPool) → Dropout → FC → Softmax
clear; clc; close all;

datasetPath = 'C:\AI\data1'; % UPDATE THIS PATH

[datasetDir, ~, ~] = fileparts(datasetPath);
if isempty(datasetDir), datasetDir = pwd; end

modelSavePath = fullfile(datasetDir, 'resistor_model_4band_customCNN');
if ~exist(modelSavePath, 'dir'), mkdir(modelSavePath); end

inputSize = [224 224 3];

%% GPU Check
parallel.gpu.enableCUDAForwardCompatibility(true);
if canUseGPU
    gpuInfo = gpuDevice;
    fprintf('✓ GPU: %s\n', gpuInfo.Name);
    reset(gpuInfo);
else
    error('GPU required! Enable GPU or change ExecutionEnvironment to "cpu".');
end

%% Load & Filter Dataset
fprintf('\n=== Loading Dataset ===\n');
imds_full = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource',       'foldernames');

originalLabels = imds_full.Labels;
fprintf('Total images found: %d\n', numel(imds_full.Files));

% Keep only folders prefixed with "4B-"
is4Band   = startsWith(string(originalLabels), '4B-', 'IgnoreCase', true);
imds_4band = subset(imds_full, is4Band);

if numel(imds_4band.Files) == 0
    error('No 4-Band images found! Ensure folders are prefixed with "4B-".');
end

% Strip "4B-" prefix and any trailing variant suffix e.g. "4B-4R7-T5" → "4R7"
cleanLabels = regexprep(string(imds_4band.Labels), '^4[Bb]-', '');
cleanLabels = regexprep(cleanLabels, '-\w+$', '');
imds_4band.Labels = categorical(cleanLabels);

fprintf('4-Band images kept: %d\n', numel(imds_4band.Files));
disp(countEachLabel(imds_4band));

%% Train / Validation Split
[imdsTrain, imdsVal] = splitEachLabel(imds_4band, 0.85, 'randomized');
numClasses = numel(categories(imdsTrain.Labels));
fprintf('Value classes: %d\n', numClasses);

if numClasses < 2
    error('Need at least 2 classes to train!');
end

%% Augmentation Pipeline
spatialAugmenter = imageDataAugmenter( ...
    'RandRotation',     [-2, 2], ...
    'RandXTranslation', [-3, 3], ...
    'RandYTranslation', [-3, 3], ...
    'RandXScale',       [0.99, 1.10], ...
    'RandYScale',       [0.99, 1.10], ...
    'RandXReflection',  true);

augTrainBase = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', spatialAugmenter);
augTrain = transform(augTrainBase, @augmentResistor);
augVal   = augmentedImageDatastore(inputSize(1:2), imdsVal);

%% =====================================================
%%  BUILD CUSTOM CNN
%%  Block 1: 32 filters  — learns edges & colour bands
%%  Block 2: 64 filters  — learns stripe combinations
%%  Block 3: 128 filters — learns band patterns
%%  Block 4: 256 filters — high-level resistor features
%%  Head: Global avg pool → Dropout → FC → Softmax
%% =====================================================
fprintf('\n=== Building Custom CNN ===\n');

layers = [

    %% Input
    imageInputLayer(inputSize, 'Name', 'input', ...
        'Normalization', 'zscore')

    %% ── Block 1 ── 32 filters, 3×3
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1_1')
    batchNormalizationLayer('Name', 'bn1_1')
    reluLayer('Name', 'relu1_1')
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1_2')
    batchNormalizationLayer('Name', 'bn1_2')
    reluLayer('Name', 'relu1_2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')   % → 112×112×32
    dropoutLayer(0.15, 'Name', 'drop1')

    %% ── Block 2 ── 64 filters
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2_1')
    batchNormalizationLayer('Name', 'bn2_1')
    reluLayer('Name', 'relu2_1')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2_2')
    batchNormalizationLayer('Name', 'bn2_2')
    reluLayer('Name', 'relu2_2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')   % → 56×56×64
    dropoutLayer(0.20, 'Name', 'drop2')

    %% ── Block 3 ── 128 filters
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3_1')
    batchNormalizationLayer('Name', 'bn3_1')
    reluLayer('Name', 'relu3_1')
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3_2')
    batchNormalizationLayer('Name', 'bn3_2')
    reluLayer('Name', 'relu3_2')
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3_3')
    batchNormalizationLayer('Name', 'bn3_3')
    reluLayer('Name', 'relu3_3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3')   % → 28×28×128
    dropoutLayer(0.25, 'Name', 'drop3')

    %% ── Block 4 ── 256 filters
    convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv4_1')
    batchNormalizationLayer('Name', 'bn4_1')
    reluLayer('Name', 'relu4_1')
    convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv4_2')
    batchNormalizationLayer('Name', 'bn4_2')
    reluLayer('Name', 'relu4_2')
    convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv4_3')
    batchNormalizationLayer('Name', 'bn4_3')
    reluLayer('Name', 'relu4_3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool4')   % → 14×14×256
    dropoutLayer(0.30, 'Name', 'drop4')

    %% ── Classification Head ──
    % Global average pooling collapses spatial dims → 1×1×256
    globalAveragePooling2dLayer('Name', 'gap')

    fullyConnectedLayer(512, 'Name', 'fc1', ...
        'WeightL2Factor', 0.01)
    batchNormalizationLayer('Name', 'bn_fc1')
    reluLayer('Name', 'relu_fc1')
    dropoutLayer(0.50, 'Name', 'drop_fc')

    fullyConnectedLayer(numClasses, 'Name', 'fc_out', ...
        'WeightLearnRateFactor', 10, ...
        'BiasLearnRateFactor',   10, ...
        'WeightL2Factor',        0.01)
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

% Quick sanity check — will error if graph is broken
analyzeNetwork(layers);

%% Training Options
%  LR schedule: start at 5e-4, drop ×0.2 at epoch 20 and 35
options = trainingOptions('adam', ...
    'InitialLearnRate',     5e-4, ...
    'LearnRateSchedule',    'piecewise', ...
    'LearnRateDropFactor',  0.2, ...
    'LearnRateDropPeriod',  20, ...
    'MaxEpochs',            60, ...
    'MiniBatchSize',        32, ...
    'ValidationData',       augVal, ...
    'ValidationFrequency',  25, ...
    'ValidationPatience',   Inf, ...
    'L2Regularization',     1e-4, ...
    'Shuffle',              'every-epoch', ...
    'Verbose',              true, ...
    'Plots',                'training-progress', ...
    'ExecutionEnvironment', 'gpu');

%% Train
fprintf('\n=== Training Custom CNN ===\n');
fprintf('Training images:   %d\n', numel(imdsTrain.Files));
fprintf('Validation images: %d\n', numel(imdsVal.Files));
fprintf('Classes:           %d\n\n', numClasses);

tic;
[trainedNet, info] = trainNetwork(augTrain, layers, options);
trainingTime = toc;

%% Evaluate
YPred    = classify(trainedNet, augVal, 'ExecutionEnvironment', 'gpu');
YVal     = imdsVal.Labels;
accuracy = mean(YPred == YVal);

fprintf('\n========================================\n');
fprintf('CUSTOM CNN TRAINING COMPLETE!\n');
fprintf('Validation Accuracy : %.2f%%\n', accuracy * 100);
fprintf('Training Time       : %.2f minutes\n', trainingTime / 60);
fprintf('========================================\n');

%% Confusion Matrix (optional — comment out for large class counts)
if numClasses <= 50
    figure('Name', 'Confusion Matrix');
    confusionchart(YVal, YPred, ...
        'Title',            '4-Band Resistor — Custom CNN', ...
        'RowSummary',       'row-normalized', ...
        'ColumnSummary',    'column-normalized');
end

%% Save
save(fullfile(modelSavePath, 'model_customCNN_4band.mat'), ...
    'trainedNet', 'info', '-v7.3');

config.accuracy    = accuracy;
config.classes     = categories(imdsTrain.Labels);
config.inputSize   = inputSize;
config.architecture = 'CustomCNN_4xConvBlock_GAP';
save(fullfile(modelSavePath, 'config_customCNN_4band.mat'), 'config');

fprintf('✓ Model saved to: %s\n', modelSavePath);


%% ========================================================
%%  AUGMENTATION HELPER  (pixel-level — runs after spatial)
%% ========================================================
function dataOut = augmentResistor(dataIn)
    img = single(dataIn.input{1});

    % Normalise to [0,1] if needed
    if max(img(:)) > 1
        img = img / 255;
    end

    % ── Brightness jitter ──
    img = img + single((rand - 0.5) * 0.4);

    % ── Contrast jitter ──
    img = (img - 0.5) * single(0.6 + rand * 0.8) + 0.5;

    % ── Gaussian blur (simulates defocus / distance) ──
    if rand > 0.5
        sigma  = rand * 2.0;
        ks     = 2 * ceil(2 * sigma) + 1;
        [x, y] = meshgrid(-(ks-1)/2 : (ks-1)/2, -(ks-1)/2 : (ks-1)/2);
        kernel = exp(-(x.^2 + y.^2) / (2 * sigma^2));
        kernel = double(kernel / sum(kernel(:)));
        for c = 1:size(img, 3)
            img(:,:,c) = single(imfilter(double(img(:,:,c)), kernel, 'replicate'));
        end
    end

    % ── Additive Gaussian noise ──
    if rand > 0.4
        img = img + single(randn(size(img))) * single(rand * 0.06);
    end

    % ── Simulated low-resolution capture ──
    if rand > 0.5
        factor = 0.7 + rand * 0.3;
        img = imresize(imresize(img, factor), [size(img,1) size(img,2)]);
    end

    % ── Hue / saturation shift ──
    if rand > 0.5 && size(img, 3) == 3
        hsv        = rgb2hsv(double(img));
        hsv(:,:,1) = mod(hsv(:,:,1) + (rand - 0.5) * 0.1, 1);
        hsv(:,:,2) = max(0, min(1, hsv(:,:,2) * (0.8 + rand * 0.4)));
        img        = single(hsv2rgb(hsv));
    end

    dataOut          = dataIn;
    dataOut.input{1} = max(single(0), min(single(1), img));
end