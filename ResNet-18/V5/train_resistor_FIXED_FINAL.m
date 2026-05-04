%% Resistor Identification - COLOR OPTIMIZED (SIMPLIFIED)
% All fixes + Color emphasis + Simplified class weighting

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

%% 3. Load Dataset
fprintf('\n=== Loading Dataset ===\n');

imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Verify color
fprintf('\n=== Verifying Color Images ===\n');
sampleImg = readimage(imds, 1);
if size(sampleImg, 3) == 3
    fprintf('✓ Images are in COLOR (RGB)\n');
    fprintf('  Image dimensions: %dx%dx%d\n', size(sampleImg, 1), size(sampleImg, 2), size(sampleImg, 3));
else
    warning('Images appear to be grayscale!');
end

labelCounts = countEachLabel(imds);
numClasses = height(labelCounts);
totalImages = numel(imds.Files);

fprintf('\nDataset Summary:\n');
fprintf('  Total Images: %d\n', totalImages);
fprintf('  Total Classes: %d\n', numClasses);
fprintf('  Min per class: %d\n', min(labelCounts.Count));
fprintf('  Max per class: %d\n', max(labelCounts.Count));
fprintf('  Avg per class: %.1f\n', totalImages/numClasses);

%% 4. Calculate Class Weights
fprintf('\n=== Calculating Class Weights ===\n');
fprintf('Strategy: Weight classes by inverse frequency\n');
fprintf('(Rare classes get higher importance)\n\n');

% Simple calculation based on label counts
labelCountsArray = labelCounts.Count;
totalSamples = sum(labelCountsArray);

% Calculate weights: inverse frequency normalized
classWeights = totalSamples ./ (numClasses * labelCountsArray);
classWeights = classWeights / sum(classWeights) * numClasses;

fprintf('Weight statistics:\n');
fprintf('  Min weight: %.3f (common classes)\n', min(classWeights));
fprintf('  Max weight: %.3f (rare classes)\n', max(classWeights));
fprintf('  Weight ratio: %.1fx\n', max(classWeights)/min(classWeights));

%% 5. Split Data
fprintf('\n=== Splitting Data ===\n');

[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.85, 'randomized');

numTrain = numel(imdsTrain.Files);
numVal = numel(imdsValidation.Files);

fprintf('Training: %d images (%.1f per class)\n', numTrain, numTrain/numClasses);
fprintf('Validation: %d images (%.1f per class)\n', numVal, numVal/numClasses);
fprintf('✓ Using ALL %d images\n', totalImages);

%% 6. Display Sample Images
fprintf('\n=== Sample Training Images ===\n');
figure('Name', 'Verify Color Bands', 'Position', [100 100 1000 700]);
for i = 1:min(9, numTrain)
    subplot(3, 3, i);
    img = readimage(imdsTrain, i);
    imshow(img);
    title(string(imdsTrain.Labels(i)), 'Interpreter', 'none', 'FontSize', 8);
end
sgtitle('Sample Images - Verify Color Bands Are Visible!', 'FontSize', 14, 'FontWeight', 'bold');
fprintf('✓ Sample images displayed\n');

%% 7. COLOR-PRESERVING Augmentation
fprintf('\n=== Configuring Augmentation ===\n');

imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-20, 20], ...
    'RandXTranslation', [-30 30], ...
    'RandYTranslation', [-30 30], ...
    'RandXScale', [0.8 1.2], ...
    'RandYScale', [0.8 1.2], ...
    'RandXReflection', true, ...
    'RandXShear', [-15 15], ...
    'RandYShear', [-15 15]);

inputSize = [224 224 3];  % RGB color!

augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', imageAugmenter, ...
    'OutputSizeMode', 'centercrop');

augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation, ...
    'OutputSizeMode', 'centercrop');

fprintf('✓ Augmentation configured\n');
fprintf('✓ RGB color preserved (224x224x3)\n');

%% 8. Load Network
fprintf('\n=== Building Network ===\n');

try
    net = resnet50;
    fprintf('✓ Using ResNet-50\n');
    fcLayerName = 'fc1000';
catch
    net = resnet18;
    fprintf('✓ Using ResNet-18\n');
    fcLayerName = 'fc1000';
end

lgraph = layerGraph(net);

%% 9. Modify Network Layers
fprintf('\n=== Modifying Network ===\n');

newLayers = [
    dropoutLayer(0.5, 'Name', 'dropout_resistor')
    fullyConnectedLayer(numClasses, 'Name', 'fc_resistor', ...
        'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'softmax_resistor')
    classificationLayer('Name', 'classoutput_resistor', ...
        'Classes', categories(imdsTrain.Labels), ...
        'ClassWeights', classWeights)];  % Apply weights directly here!

% Replace layers
lgraph = replaceLayer(lgraph, fcLayerName, newLayers(1));
lgraph = replaceLayer(lgraph, 'prob', newLayers(2));

try
    lgraph = replaceLayer(lgraph, 'prob_softmax', newLayers(3));
catch
    % ResNet-18 doesn't have prob_softmax
end

lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', newLayers(4));

fprintf('✓ Network modified for %d classes\n', numClasses);
fprintf('✓ Class weights applied\n');

%% 10. Training Options
fprintf('\n=== Training Configuration ===\n');

initialLearningRate = 0.0003;
maxEpochs = 60;
miniBatchSize = 32;
validationFrequency = 30;

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
    'ValidationPatience', Inf, ... % NEVER stop early!
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'gpu', ...
    'OutputNetwork', 'best-validation-loss', ...
    'CheckpointPath', modelSavePath);

fprintf('✓ Training configured:\n');
fprintf('  - Epochs: %d (will complete ALL)\n', maxEpochs);
fprintf('  - Batch size: %d\n', miniBatchSize);
fprintf('  - Learning rate: %.4f\n', initialLearningRate);
fprintf('  - Early stopping: DISABLED\n');
fprintf('  - GPU: Enabled\n');

%% 11. Training Summary
fprintf('\n========================================\n');
fprintf('    READY TO TRAIN\n');
fprintf('========================================\n');
fprintf('Data: %d images, %d classes\n', totalImages, numClasses);
fprintf('Strategy: COLOR-preserved + Class-weighted\n');
fprintf('Duration: ~30-60 minutes\n');
fprintf('\n');
fprintf('Color Information:\n');
fprintf('  ✓ RGB color preserved (critical for bands!)\n');
fprintf('  ✓ No grayscale conversion\n');
fprintf('  ✓ Resistor colors intact\n');
fprintf('\n');
fprintf('Class Imbalance:\n');
fprintf('  ✓ ALL images used (none discarded)\n');
fprintf('  ✓ Rare classes weighted %.1fx higher\n', max(classWeights)/min(classWeights));
fprintf('  ✓ Fair training across all types\n');
fprintf('========================================\n\n');

pause(2);

%% 12. Train
fprintf('Starting training in 3 seconds...\n');
pause(3);

fprintf('\n========================================\n');
fprintf('    TRAINING STARTED\n');
fprintf('========================================\n\n');

tic;
[trainedNet, info] = trainNetwork(augimdsTrain, lgraph, options);
trainingTime = toc;

fprintf('\n========================================\n');
fprintf('    TRAINING COMPLETE!\n');
fprintf('========================================\n');
fprintf('Duration: %.2f minutes\n', trainingTime/60);

%% 13. Evaluate
fprintf('\n=== Evaluation ===\n');

YPred = classify(trainedNet, augimdsValidation, 'ExecutionEnvironment', 'gpu');
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation) / numel(YValidation);

% Top-5 accuracy
[~, scores] = classify(trainedNet, augimdsValidation, 'ExecutionEnvironment', 'gpu');
[~, sortedIdx] = sort(scores, 2, 'descend');

classNames = categories(imdsTrain.Labels);
top5Correct = 0;
for i = 1:numel(YValidation)
    classIdx = find(strcmp(classNames, char(YValidation(i))));
    if any(sortedIdx(i, 1:5) == classIdx)
        top5Correct = top5Correct + 1;
    end
end
top5Accuracy = top5Correct / numel(YValidation) * 100;

fprintf('\n========================================\n');
fprintf('       FINAL RESULTS\n');
fprintf('========================================\n');
fprintf('Top-1 Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Top-5 Accuracy: %.2f%%\n', top5Accuracy);
fprintf('Training Time: %.2f minutes\n', trainingTime/60);
fprintf('Epochs Completed: %d/%d\n', maxEpochs, maxEpochs);
fprintf('========================================\n');

% Confusion matrix
figure;
cm = confusionchart(YValidation, YPred);
cm.Title = sprintf('Accuracy: %.1f%% (Top-1) | %.1f%% (Top-5)', ...
    accuracy * 100, top5Accuracy);

%% 14. Save Model
fprintf('\n=== Saving Model ===\n');

modelFile = fullfile(modelSavePath, 'resistor_color_model.mat');
configFile = fullfile(modelSavePath, 'model_config.mat');
classFile = fullfile(modelSavePath, 'class_names.txt');

save(modelFile, 'trainedNet', 'info', '-v7.3');
fprintf('✓ Model: %s\n', modelFile);

config.inputSize = inputSize;
config.numClasses = numClasses;
config.classNames = classNames;
config.accuracy = accuracy;
config.top5Accuracy = top5Accuracy;
config.trainingTime = trainingTime;
config.trainingDate = datetime('now');
config.totalImages = totalImages;
config.colorPreserved = true;
save(configFile, 'config');
fprintf('✓ Config: %s\n', configFile);

fid = fopen(classFile, 'w');
for i = 1:numClasses
    fprintf(fid, '%s\n', classNames{i});
end
fclose(fid);
fprintf('✓ Classes: %s\n', classFile);

fprintf('\n========================================\n');
fprintf('     SUCCESS!\n');
fprintf('========================================\n');
fprintf('Model saved to: %s\n', modelSavePath);
fprintf('Accuracy: %.1f%% (top-1), %.1f%% (top-5)\n', accuracy*100, top5Accuracy);
fprintf('All %d images used with color preserved!\n', totalImages);
fprintf('========================================\n');

beep;
