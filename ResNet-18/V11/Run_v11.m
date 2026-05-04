%% =======================================================
%% 4-Band Resistor Prediction — Custom CNN
%%  Loads the model trained by the Custom CNN trainer
%%  and classifies a single user-selected image
%% =======================================================
clear; clc; close all;

%% ── Path to your Custom CNN model ──────────────────────
modelFolder = 'C:\AI\resistor_model_4band_customCNN'; % UPDATE IF NEEDED
matFile     = 'model_customCNN_4band.mat';
inputSize   = [224 224];

%% ── 1. Load Model ───────────────────────────────────────
fprintf('Loading 4-Band Custom CNN model...\n');
try
    load(fullfile(modelFolder, matFile), 'trainedNet');
    net4Band = trainedNet;
    fprintf('✓ Model loaded from: %s\n', fullfile(modelFolder, matFile));
catch
    error('Could not load model. Check path:\n  %s', fullfile(modelFolder, matFile));
end

%% ── 2. Select Image ─────────────────────────────────────
[fileName, pathName] = uigetfile( ...
    {'*.jpg;*.png;*.jpeg;*.bmp', 'Image Files'}, ...
    'Select a 4-Band Resistor Image');
if isequal(fileName, 0)
    fprintf('No image selected — exiting.\n');
    return
end

imgRaw     = imread(fullfile(pathName, fileName));
imgResized = imresize(imgRaw, inputSize);

fprintf('\n--- ANALYSIS START ---\n');
fprintf('Image: %s\n', fileName);

%% ── 3. Classify ─────────────────────────────────────────
[valPred, valScores] = classify(net4Band, imgResized);

valueLabel      = string(valPred);
valueConfidence = max(valScores) * 100;

%% ── 4. Top-5 Predictions ────────────────────────────────
classNames = net4Band.Layers(end).Classes;
[sortedScores, idx] = sort(valScores, 'descend');

topN = min(5, numel(classNames));
fprintf('\nTop %d Predictions:\n', topN);
fprintf('  %-4s  %-12s  %s\n', 'Rank', 'Value', 'Confidence');
fprintf('  %s\n', repmat('-', 1, 32));
for i = 1:topN
    marker = '';
    if i == 1, marker = '  ◄ BEST'; end
    fprintf('  %-4d  %-12s  %.1f%%%s\n', ...
        i, string(classNames(idx(i))), sortedScores(i)*100, marker);
end

%% ── 5. Display Result ───────────────────────────────────
figure('Name', '4-Band Resistor — Custom CNN', 'NumberTitle', 'off');

subplot(1, 2, 1);
imshow(imgRaw);

% Green if confident, amber if moderate, red if low
if valueConfidence >= 75
    titleColor = [0 0.55 0];        % green
elseif valueConfidence >= 50
    titleColor = [0.85 0.55 0];     % amber
else
    titleColor = [0.85 0 0];        % red
end

title(sprintf('Value: %s\nConfidence: %.1f%%', valueLabel, valueConfidence), ...
    'FontSize', 14, 'FontWeight', 'bold', 'Color', titleColor);

%% ── Bar chart of top-5 scores ───────────────────────────
subplot(1, 2, 2);

topLabels     = string(classNames(idx(1:topN)));
topVals       = sortedScores(1:topN) * 100;
flippedLabels = flip(topLabels);
flippedVals   = flip(topVals);

barh(flippedVals, 'FaceColor', [0.2 0.5 0.8]);
set(gca, 'YTickLabel', flippedLabels, 'YTick', 1:topN, 'FontSize', 10);
xlabel('Confidence (%)');
title('Top Predictions', 'FontSize', 12);
xlim([0 105]);
grid on;

% Annotate bars with percentages
for i = 1:topN
    text(flippedVals(i) + 1, i, sprintf('%.1f%%', flippedVals(i)), ...
        'VerticalAlignment', 'middle', 'FontSize', 9);
end

sgtitle(sprintf('4-Band Resistor Classifier  |  File: %s', fileName), ...
    'FontSize', 11, 'Interpreter', 'none');

%% ── 6. Console Summary ──────────────────────────────────
fprintf('\n========================================\n');
fprintf('  FINAL RESULT : %s\n', valueLabel);
fprintf('  Confidence   : %.1f%%\n', valueConfidence);
if valueConfidence < 50
    fprintf('  ⚠ Low confidence — check image quality\n');
elseif valueConfidence < 75
    fprintf('  △ Moderate confidence\n');
else
    fprintf('  ✓ High confidence\n');
end
fprintf('========================================\n');