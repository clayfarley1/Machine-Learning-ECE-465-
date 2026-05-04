%% Resistor Identification Model Training Script
% This script trains a deep learning model to identify resistor values
% from images of color-coded resistors

clear; clc; close all;

%% 1. Setup and Configuration
% Define your dataset path
datasetPath = 'path/to/your/resistor/images'; % UPDATE THIS PATH

% Image preprocessing parameters
inputSize = [224 224 3]; % Standard size for pretrained networks
augmentationIntensity = 0.3;

% Training parameters
initialLearningRate = 0.001;
maxEpochs = 20;
miniBatchSize = 32;
validationFrequency = 10;

%% 2. Load and Prepare Dataset
% Assuming your images are organized in folders by resistor value
% e.g., datasetPath/10ohm/, datasetPath/100ohm/, etc.

imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Display dataset statistics
numImages = numel(imds.Files);
labelCounts = countEachLabel(imds);
disp('Dataset Summary:');
disp(labelCounts);

% Split data into training (80%) and validation (20%)
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');

fprintf('Training images: %d\n', numel(imdsTrain.Files));
fprintf('Validation images: %d\n', numel(imdsValidation.Files));

%% 3. Data Augmentation
% Create augmented image datastores for training
imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-20, 20], ...
    'RandXTranslation', [-10 10], ...
    'RandYTranslation', [-10 10], ...
    'RandXScale', [0.9 1.1], ...
    'RandYScale', [0.9 1.1], ...
    'RandXReflection', true);

augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', imageAugmenter, ...
    'ColorPreprocessing', 'gray2rgb');

augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation, ...
    'ColorPreprocessing', 'gray2rgb');

%% 4. Define Network Architecture
% Using transfer learning with ResNet-18 (you can also use ResNet-50, GoogLeNet, etc.)

% Load pretrained network
net = resnet18;

% Get input size
inputSize = net.Layers(1).InputSize;

% Replace final layers for your classification task
numClasses = numel(categories(imdsTrain.Labels));

lgraph = layerGraph(net);

% Find and replace final layers
newLayers = [
    fullyConnectedLayer(numClasses, 'Name', 'fc_new', ...
        'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'softmax_new')
    classificationLayer('Name', 'classoutput_new')];

lgraph = replaceLayer(lgraph, 'fc1000', newLayers(1));
lgraph = replaceLayer(lgraph, 'prob', newLayers(2));
lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', newLayers(3));

%% 5. Training Options
options = trainingOptions('adam', ...
    'InitialLearnRate', initialLearningRate, ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', validationFrequency, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto'); % Will use GPU if available

%% 6. Train the Network
fprintf('\nStarting training...\n');
[trainedNet, info] = trainNetwork(augimdsTrain, lgraph, options);

%% 7. Evaluate Model Performance
% Predict on validation set
YPred = classify(trainedNet, augimdsValidation);
YValidation = imdsValidation.Labels;

% Calculate accuracy
accuracy = sum(YPred == YValidation) / numel(YValidation);
fprintf('\nValidation Accuracy: %.2f%%\n', accuracy * 100);

% Display confusion matrix
figure;
confusionchart(YValidation, YPred);
title(sprintf('Validation Accuracy: %.2f%%', accuracy * 100));

%% 8. Save the Trained Model
save('resistor_identification_model.mat', 'trainedNet', 'info');
fprintf('\nModel saved as: resistor_identification_model.mat\n');

% Save training configuration for reference
config.inputSize = inputSize;
config.numClasses = numClasses;
config.classNames = categories(imdsTrain.Labels);
config.accuracy = accuracy;
save('model_config.mat', 'config');

fprintf('\nTraining complete!\n');
