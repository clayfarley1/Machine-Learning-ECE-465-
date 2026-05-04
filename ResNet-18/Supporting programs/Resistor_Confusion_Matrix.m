%% Resistor Confusion Matrix
clear; clc; close all;

%% Load Models
bandTypeFolder = 'C:\AI\resistor_model_bandtype';
fourBandFolder = 'C:\AI\resistor_model_4band_only';
fiveBandFolder = 'C:\AI\resistor_model_5band_only';

load(fullfile(bandTypeFolder, 'model_bandtype.mat'),     'trainedNet'); bandTypeNet = trainedNet;
load(fullfile(fourBandFolder, 'model_4band_values.mat'), 'trainedNet'); net4Band    = trainedNet;
load(fullfile(fiveBandFolder, 'model_5band_values.mat'), 'trainedNet'); net5Band    = trainedNet;
fprintf('All models loaded\n');

%% Select Test Folder
rootFolder = uigetdir('C:\', 'Select Folder With Class Subfolders');
if isequal(rootFolder, 0), return; end

imds    = imageDatastore(rootFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
nImages = numel(imds.Files);

% Clean true labels same way as training: strip 4B-/5B- prefix and -T5 suffix
trueLabels = regexprep(string(imds.Labels), '^[45][Bb]-', '');
trueLabels = regexprep(trueLabels, '-\w+$', '');

predLabels = strings(nImages, 1);

%% Run Predictions
for i = 1:nImages
    img = imresize(readimage(imds, i), [224 224]);

    [bandPred, ~] = classify(bandTypeNet, img);
    bandType = string(bandPred);

    if contains(bandType, '4', 'IgnoreCase', true)
        targetNet = net4Band;
    else
        targetNet = net5Band;
    end

    [valPred, ~]  = classify(targetNet, img);
    predLabels(i) = string(valPred);

    if mod(i,50)==0 || i==nImages
        fprintf('  %d / %d\n', i, nImages);
    end
end

%% Plot and Save
figure('Name','Confusion Matrix','NumberTitle','off','Position',[50 50 1000 900]);
plotconfusion(categorical(trueLabels), categorical(predLabels));
accuracy = sum(trueLabels == predLabels) / nImages * 100;
title(sprintf('Resistor Confusion Matrix  (Acc: %.2f%%)', accuracy), 'FontSize', 14);

saveas(gcf, fullfile(rootFolder, 'confusion_matrix.png'));
fprintf('Done! Accuracy: %.2f%%\n', accuracy);
fprintf('Saved: %s\n', fullfile(rootFolder, 'confusion_matrix.png'));