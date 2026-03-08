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
    'RandRotation',     [-25, 25], ...
    'RandXTranslation', [-35 35], ...
    'RandYTranslation', [-35 35], ...
    'RandXScale',       [0.75 1.25], ...
    'RandYScale',       [0.75 1.25], ...
    'RandXReflection',  true);

augTrainBase = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', spatialAugmenter);
augTrain     = transform(augTrainBase, @augmentResistor);
augVal       = augmentedImageDatastore(inputSize(1:2), imdsVal);

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
    'LearnRateDropFactor',  0.2, ...
    'LearnRateDropPeriod',  15, ...
    'MaxEpochs',            50, ...
    'MiniBatchSize',        32, ...
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


%% ========================================
%% AUGMENTATION FUNCTION - must be at end
%% ========================================

function dataOut = augmentResistor(dataIn)
    img = single(dataIn.input{1});

    if max(img(:)) > 1
        img = img / 255;
    end

    % Brightness
    img = img + single((rand - 0.5) * 0.4);

    % Contrast
    img = (img - 0.5) * single(0.6 + rand * 0.8) + 0.5;

    % Blur
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

    % Noise
    if rand > 0.4
        img = img + single(randn(size(img))) * single(rand * 0.06);
    end

    % Downscale/upscale
    if rand > 0.5
        factor = 0.7 + rand * 0.3;
        img    = imresize(imresize(img, factor), [size(img,1) size(img,2)]);
    end

    % Hue/saturation shift
    if rand > 0.5 && size(img,3) == 3
        hsv        = rgb2hsv(double(img));
        hsv(:,:,1) = mod(hsv(:,:,1) + (rand-0.5)*0.1, 1);
        hsv(:,:,2) = max(0, min(1, hsv(:,:,2) * (0.8 + rand*0.4)));
        img        = single(hsv2rgb(hsv));
    end

    dataOut          = dataIn;
    dataOut.input{1} = max(single(0), min(single(1), img));
end