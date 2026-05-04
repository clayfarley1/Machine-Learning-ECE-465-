%% Resistor Identification - COLOR OPTIMIZED
% Emphasizes color information for resistor band identification
% All fixes included: GPU, save location, class weights, no early stopping

clear; clc; close all;

%% 1. CUDA Forward Compatibility
fprintf('========================================\n');
fprintf('    GPU SETUP\n');
fprintf('========================================\n');

parallel.gpu.enableCUDAForwardCompatibility(true);
fprintf('✓ CUDA forward compatibility enabled\n');

if canUseGPU
    gpuInfo = gpuDevice;
    fprintf('✓ GPU: %s\n', gpuInfo.Name);
    fprintf('  Memory: %.2f GB\n', gpuInfo.TotalMemory / 1e9);
    reset(gpuInfo);
else
    error('GPU not available!');
end

%% 2. Dataset Configuration
fprintf('\n========================================\n');
fprintf('    DATASET CONFIGURATION\n');
fprintf('========================================\n');

datasetPath = 'C:\AI\full data'; % UPDATE THIS PATH

[datasetDir, ~, ~] = fileparts(datasetPath);
if isempty(datasetDir)
    datasetDir = pwd;
end

modelSavePath = fullfile(datasetDir, 'resistor_model');
if ~exist(modelSavePath, 'dir')
    mkdir(modelSavePath);
end

fprintf('Dataset Path: %s\n', datasetPath);
fprintf('Model Save Path: %s\n', modelSavePath);

%% 3. Load Dataset and Verify Color
fprintf('\n=== Loading Dataset ===\n');

imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'ReadFcn', @readAndVerifyColor);  % Custom function to ensure color

labelCounts = countEachLabel(imds);
numClasses = height(labelCounts);
totalImages = numel(imds.Files);

fprintf('Total Images: %d\n', totalImages);
fprintf('Total Classes: %d\n', numClasses);

% Check a sample image to verify it's in color
fprintf('\n=== Verifying Color Images ===\n');
sampleImg = readimage(imds, 1);
if size(sampleImg, 3) == 3
    fprintf('✓ Images are in COLOR (RGB)\n');
    fprintf('  Image size: %dx%dx%d\n', size(sampleImg, 1), size(sampleImg, 2), size(sampleImg, 3));
else
    warning('Images appear to be grayscale! Color bands may not be detected properly.');
end

%% 4. Class Weights (Use ALL images)
fprintf('\n=== Class Weighting Strategy ===\n');
fprintf('Using ALL images with weighted loss\n');
fprintf('(Color information preserved for all images!)\n\n');

totalSamples = sum(labelCounts.Count);
classWeights = zeros(numClasses, 1);

for i = 1:numClasses
    classWeights(i) = totalSamples / (numClasses * labelCounts.Count(i));
end

classWeights = classWeights / sum(classWeights) * numClasses;

fprintf('Weight range: %.3f to %.3f\n', min(classWeights), max(classWeights));

%% 5. Split Data
fprintf('\n=== Splitting Data ===\n');

[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.85, 'randomized');

numTrain = numel(imdsTrain.Files);
numVal = numel(imdsValidation.Files);

fprintf('Training: %d images (%.1f per class)\n', numTrain, numTrain/numClasses);
fprintf('Validation: %d images (%.1f per class)\n', numVal, numVal/numClasses);

%% 6. COLOR-PRESERVING Data Augmentation
fprintf('\n=== Configuring COLOR-PRESERVING Augmentation ===\n');
fprintf('Key: Color information is CRITICAL for resistor bands!\n\n');

% Standard geometric augmentation (preserves colors)
imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-20, 20], ...
    'RandXTranslation', [-30 30], ...
    'RandYTranslation', [-30 30], ...
    'RandXScale', [0.8 1.2], ...
    'RandYScale', [0.8 1.2], ...
    'RandXReflection', true, ...
    'RandXShear', [-15 15], ...
    'RandYShear', [-15 15]);

inputSize = [224 224 3];  % 3 channels for RGB color!

% Create augmented datastores with COLOR PRESERVATION
augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', imageAugmenter, ...
    'OutputSizeMode', 'centercrop');
    % NOTE: No ColorPreprocessing - keeps original RGB colors!

augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation, ...
    'OutputSizeMode', 'centercrop');
    % NOTE: No ColorPreprocessing - keeps original RGB colors!

fprintf('✓ Augmentation configured\n');
fprintf('✓ Color channels preserved (RGB)\n');
fprintf('✓ No color conversion applied\n');

% Display sample augmented images to verify color
fprintf('\n=== Displaying Sample Training Images ===\n');
figure('Name', 'Sample Training Images - Verify Colors', 'Position', [100 100 1000 700]);
for i = 1:min(9, numTrain)
    subplot(3, 3, i);
    img = readimage(imdsTrain, i);
    imshow(img);
    title(string(imdsTrain.Labels(i)), 'Interpreter', 'none');
end
sgtitle('Sample Images - Verify Color Bands Are Visible');
fprintf('✓ Sample images displayed - verify colors are visible!\n');

%% 7. Network Architecture
fprintf('\n=== Building Network ===\n');

try
    net = resnet50;
    fprintf('✓ Using ResNet-50 (pre-trained on COLOR images)\n');
    fcLayerName = 'fc1000';
catch
    net = resnet18;
    fprintf('✓ Using ResNet-18 (pre-trained on COLOR images)\n');
    fcLayerName = 'fc1000';
end

fprintf('  Network input: 224x224x3 (RGB color)\n');

lgraph = layerGraph(net);

% Add layers
newLayers = [
    dropoutLayer(0.5, 'Name', 'dropout_resistor')
    fullyConnectedLayer(numClasses, 'Name', 'fc_resistor', ...
        'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'softmax_resistor')
    classificationLayer('Name', 'classoutput_resistor')];

lgraph = replaceLayer(lgraph, fcLayerName, newLayers(1));
lgraph = replaceLayer(lgraph, 'prob', newLayers(2));

try
    lgraph = replaceLayer(lgraph, 'prob_softmax', newLayers(3));
catch
end

lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', newLayers(4));

%% 8. Apply Class Weights
fprintf('\n=== Applying Class Weights ===\n');

classNames = categories(imdsTrain.Labels);
classWeightsVector = zeros(numClasses, 1);

for i = 1:numClasses
    idx = find(strcmp(labelCounts.Label, classNames{i}));
    classWeightsVector(i) = classWeights(idx);
end

weightedClassLayer = classificationLayer('Name', 'classoutput_weighted', ...
    'Classes', classNames, ...
    'ClassWeights', classWeightsVector);

lgraph = replaceLayer(lgraph, 'classoutput_resistor', weightedClassLayer);

fprintf('✓ Class weights applied\n');

%% 9. Training Options
fprintf('\n=== Training Configuration ===\n');

initialLearningRate = 0.0003;
maxEpochs = 60;
miniBatchSize = 32;
validationFrequency = 30;
validationPatience = Inf;  % Never stop early!

options = trainingOptions('adam', ...
    'InitialLearnRate', initialLearningRate, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.3, ...
    'LearnRateDropPeriod', 20, ...
    'L2Regularization', 0.0001, ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', validationFrequency, ...
    'ValidationPatience', validationPatience, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'gpu', ...
    'OutputNetwork', 'best-validation-loss', ...
    'CheckpointPath', modelSavePath);

fprintf('✓ Training configured\n');
fprintf('  Epochs: %d (complete all)\n', maxEpochs);
fprintf('  Early stopping: DISABLED\n');

%% 10. Training Summary
fprintf('\n========================================\n');
fprintf('    COLOR-OPTIMIZED TRAINING PLAN\n');
fprintf('========================================\n');
fprintf('Strategy: Preserve COLOR information\n');
fprintf('\n');
fprintf('Why Color Matters:\n');
fprintf('  - Resistor bands are COLORED (brown, red, orange, etc.)\n');
fprintf('  - Color is PRIMARY identification feature\n');
fprintf('  - Grayscale would lose critical information!\n');
fprintf('\n');
fprintf('Data:\n');
fprintf('  - Total images: %d (ALL color-preserved)\n', totalImages);
fprintf('  - Classes: %d\n', numClasses);
fprintf('  - Input: 224x224x3 RGB\n');
fprintf('\n');
fprintf('Training:\n');
fprintf('  - Epochs: %d\n', maxEpochs);
fprintf('  - Estimated time: 30-60 minutes\n');
fprintf('========================================\n\n');

pause(2);

%% 11. Train
fprintf('Starting training...\n\n');

tic;
[trainedNet, info] = trainNetwork(augimdsTrain, lgraph, options);
trainingTime = toc;

fprintf('\n✓ Training completed: %.2f minutes\n', trainingTime/60);

%% 12. Evaluate
fprintf('\n=== Evaluation ===\n');

YPred = classify(trainedNet, augimdsValidation, 'ExecutionEnvironment', 'gpu');
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation) / numel(YValidation);

% Top-5 accuracy
[~, scores] = classify(trainedNet, augimdsValidation, 'ExecutionEnvironment', 'gpu');
[~, sortedIdx] = sort(scores, 2, 'descend');
top5Correct = 0;
for i = 1:numel(YValidation)
    classIdx = find(strcmp(classNames, char(YValidation(i))));
    if any(sortedIdx(i, 1:5) == classIdx)
        top5Correct = top5Correct + 1;
    end
end
top5Accuracy = top5Correct / numel(YValidation) * 100;

fprintf('\n========================================\n');
fprintf('       RESULTS\n');
fprintf('========================================\n');
fprintf('Top-1 Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Top-5 Accuracy: %.2f%%\n', top5Accuracy);
fprintf('Training Time: %.2f minutes\n', trainingTime/60);
fprintf('========================================\n');

% Confusion matrix
figure;
cm = confusionchart(YValidation, YPred);
cm.Title = sprintf('Color-Optimized Model: %.1f%% Top-1', accuracy * 100);

%% 13. Save Model
fprintf('\n=== Saving Model ===\n');

modelFile = fullfile(modelSavePath, 'resistor_color_model.mat');
configFile = fullfile(modelSavePath, 'model_config.mat');
classFile = fullfile(modelSavePath, 'class_names.txt');

save(modelFile, 'trainedNet', 'info', '-v7.3');
fprintf('✓ Model saved: %s\n', modelFile);

config.inputSize = inputSize;
config.numClasses = numClasses;
config.classNames = classNames;
config.accuracy = accuracy;
config.top5Accuracy = top5Accuracy;
config.trainingTime = trainingTime;
config.colorOptimized = true;
config.datasetPath = datasetPath;
save(configFile, 'config');
fprintf('✓ Config saved: %s\n', configFile);

fid = fopen(classFile, 'w');
for i = 1:numClasses
    fprintf(fid, '%s\n', classNames{i});
end
fclose(fid);
fprintf('✓ Class names saved: %s\n', classFile);

fprintf('\n========================================\n');
fprintf('    TRAINING COMPLETE!\n');
fprintf('========================================\n');
fprintf('Color information: PRESERVED ✓\n');
fprintf('All images used: %d ✓\n', totalImages);
fprintf('Accuracy: %.1f%% ✓\n', accuracy * 100);
fprintf('Model location: %s\n', modelSavePath);
fprintf('========================================\n');

beep;

%% Helper Function - Ensure Color Images
function img = readAndVerifyColor(filename)
    img = imread(filename);
    
    % If grayscale, convert to RGB
    if size(img, 3) == 1
        img = cat(3, img, img, img);
        warning('Grayscale image converted to RGB: %s', filename);
    end
    
    % Ensure uint8
    if ~isa(img, 'uint8')
        img = im2uint8(img);
    end
end
