%% Resistor Confusion Matrix - Simplified for large class counts
clear; clc; close all;

%% Load Models
bandTypeFolder = 'C:\AI\resistor_model_bandtype';
fourBandFolder = 'C:\AI\4_Band_final';
fiveBandFolder = 'C:\AI\5_Band_final';

load(fullfile(bandTypeFolder, 'model_bandtype.mat'),     'trainedNet'); bandTypeNet = trainedNet;
load(fullfile(fourBandFolder, 'model_4band_values.mat'), 'trainedNet'); net4Band    = trainedNet;
load(fullfile(fiveBandFolder, 'model_5band_values.mat'), 'trainedNet'); net5Band    = trainedNet;
fprintf('All models loaded\n');

%% Select Test Folder
rootFolder = uigetdir('C:\', 'Select Folder With Class Subfolders');
if isequal(rootFolder, 0), return; end

imds    = imageDatastore(rootFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
nImages = numel(imds.Files);

origLabels = string(imds.Labels);   % keep originals for band-type detection
trueLabels = regexprep(origLabels, '^[45][Bb]-', '');
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

%% Build confusion matrix
allClasses = unique([trueLabels; predLabels]);
numC       = numel(allClasses);
trueIdx    = arrayfun(@(x) find(strcmp(allClasses,x)), trueLabels);
predIdx    = arrayfun(@(x) find(strcmp(allClasses,x)), predLabels);
C          = accumarray([trueIdx, predIdx], 1, [numC numC]);
accuracy   = sum(diag(C)) / sum(C(:)) * 100;

fprintf('\nOverall Accuracy: %.2f%%\n', accuracy);

%% Determine band type per class (for bar chart colouring)
%  Look at the original folder labels to decide 4-band vs 5-band per class
isFiveBand = false(numC, 1);
for k = 1:numC
    matchMask = (trueIdx == k);
    if any(matchMask)
        sampOrig = origLabels(matchMask);
        isFiveBand(k) = any(contains(sampOrig, {'5B','5b','5-'}, 'IgnoreCase', true));
    end
end

%% -----------------------------------------------------------------------
%% FIGURE 1: Heatmap (colour only, no text)
%% -----------------------------------------------------------------------
Cnorm = 100 * C ./ max(sum(C,2), 1);

% Colours
COL_CORRECT = [0.08 0.55 0.15];   % dark green
COL_ERROR   = [0.85 0.12 0.12];   % red
COL_EMPTY   = [1.00 1.00 1.00];   % white

rgb = repmat(reshape(COL_EMPTY, 1, 1, 3), numC, numC);  % all white

for k = 1:numC
    if Cnorm(k,k) > 0
        rgb(k,k,:) = COL_CORRECT;
    end
end
for r = 1:numC
    for c = 1:numC
        if r ~= c && Cnorm(r,c) > 0
            rgb(r,c,:) = COL_ERROR;
        end
    end
end

fig1 = figure('Name','Confusion Heatmap','NumberTitle','off','Position',[50 50 960 880]);
ax   = axes('Parent', fig1);
imshow(rgb, 'Parent', ax, 'InitialMagnification', 'fit');

if numC <= 60
    set(ax, 'XTick', 1:numC, 'XTickLabel', allClasses, 'XTickLabelRotation', 90, ...
            'YTick', 1:numC, 'YTickLabel', allClasses, 'FontSize', 7);
else
    set(ax, 'XTick', [], 'YTick', []);
end
xlabel(ax, 'Predicted Class',  'FontSize', 11, 'FontWeight', 'bold');
ylabel(ax, 'True Class',       'FontSize', 11, 'FontWeight', 'bold');

% --- Professional title (two lines) ---
title(ax, { ...
    sprintf('Resistor Recognition — Confusion Matrix  |  Overall Accuracy: %.2f%%', accuracy), ...
    sprintf('n = %d images  ·  %d classes', nImages, numC)}, ...
    'FontSize', 13, 'FontWeight', 'bold');

% --- Colour-swatch legend (right side, outside the plot) ---
% Position: [left bottom width height] in normalised figure units
% Placed to the right of the axes, vertically centred

% Background box
annotation(fig1, 'rectangle', [0.825 0.40 0.155 0.175], ...
    'FaceColor', 'white', 'EdgeColor', [0.35 0.35 0.35], 'LineWidth', 1.2);

% Title text
annotation(fig1, 'textbox', [0.830 0.548 0.145 0.025], ...
    'String', 'Legend', ...
    'FontSize', 10, 'FontWeight', 'bold', 'Color', [0 0 0], ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center');

% Green swatch + label
annotation(fig1, 'rectangle', [0.835 0.508 0.025 0.022], ...
    'FaceColor', COL_CORRECT, 'EdgeColor', [0.2 0.2 0.2], 'LineWidth', 0.8);
annotation(fig1, 'textbox',   [0.865 0.502 0.110 0.030], ...
    'String', 'Correct', ...
    'FontSize', 9.5, 'Color', [0 0 0], 'EdgeColor', 'none', 'VerticalAlignment', 'middle');

% Red swatch + label
annotation(fig1, 'rectangle', [0.835 0.468 0.025 0.022], ...
    'FaceColor', COL_ERROR, 'EdgeColor', [0.2 0.2 0.2], 'LineWidth', 0.8);
annotation(fig1, 'textbox',   [0.865 0.462 0.110 0.030], ...
    'String', 'Misclassified', ...
    'FontSize', 9.5, 'Color', [0 0 0], 'EdgeColor', 'none', 'VerticalAlignment', 'middle');

% White swatch + label
annotation(fig1, 'rectangle', [0.835 0.428 0.025 0.022], ...
    'FaceColor', COL_EMPTY, 'EdgeColor', [0.5 0.5 0.5], 'LineWidth', 0.8);
annotation(fig1, 'textbox',   [0.865 0.422 0.110 0.030], ...
    'String', 'No predictions', ...
    'FontSize', 9.5, 'Color', [0 0 0], 'EdgeColor', 'none', 'VerticalAlignment', 'middle');

saveas(fig1, fullfile(rootFolder, 'confusion_heatmap.png'));
fprintf('Saved: confusion_heatmap.png\n');

%% -----------------------------------------------------------------------
%% FIGURE 2: Per-class accuracy bar chart — 4-band vs 5-band coloured
%% -----------------------------------------------------------------------
COL_4BAND = [0.18 0.49 0.80];   % solid blue
COL_5BAND = [0.92 0.95 1.00];   % near-white (very light blue tint)

classAcc = diag(C) ./ max(sum(C,2), 1) * 100;

% Build per-bar colour matrix
barColors = zeros(numC, 3);
for k = 1:numC
    if isFiveBand(k)
        barColors(k,:) = COL_5BAND;
    else
        barColors(k,:) = COL_4BAND;
    end
end

fig2 = figure('Name','Per-Class Accuracy','NumberTitle','off','Position',[100 100 1200 540]);
ax2  = axes('Parent', fig2);
b    = bar(ax2, classAcc, 'FaceColor', 'flat', 'EdgeColor', [0.3 0.3 0.3], 'LineWidth', 0.4);
b.CData = barColors;

hold(ax2, 'on');
yline(accuracy, 'r--', 'LineWidth', 1.8, ...
      'Label', sprintf('Overall  %.1f%%', accuracy), ...
      'LabelVerticalAlignment', 'bottom', 'FontSize', 9);
hold(ax2, 'off');

if numC <= 80
    set(ax2, 'XTick', 1:numC, 'XTickLabel', allClasses, ...
             'XTickLabelRotation', 90, 'FontSize', 7);
else
    set(ax2, 'XTick', []);
    xlabel(ax2, sprintf('%d classes  (too many to label individually)', numC), 'FontSize', 10);
end
ylim(ax2, [0 105]);
ylabel(ax2, 'Per-Class Accuracy (%)', 'FontSize', 11, 'FontWeight', 'bold');

title(ax2, { ...
    sprintf('Resistor Recognition — Per-Class Accuracy  |  Overall: %.2f%%', accuracy), ...
    sprintf('n = %d images  ·  %d classes', nImages, numC)}, ...
    'FontSize', 13, 'FontWeight', 'bold');

grid(ax2, 'on');
ax2.GridAlpha = 0.25;

% Legend for band type colouring
h4 = patch(ax2, NaN, NaN, COL_4BAND, 'EdgeColor', [0.3 0.3 0.3]);
h5 = patch(ax2, NaN, NaN, COL_5BAND, 'EdgeColor', [0.3 0.3 0.3]);
legend(ax2, [h4 h5], {'4-Band resistor', '5-Band resistor'}, ...
    'Location', 'northeast', 'FontSize', 10, 'Box', 'on');

saveas(fig2, fullfile(rootFolder, 'accuracy_per_class.png'));
fprintf('Saved: accuracy_per_class.png\n');

%% -----------------------------------------------------------------------
%% CONSOLE: Top 20 most confused pairs
%% -----------------------------------------------------------------------
Coff = C;
Coff(eye(numC)==1) = 0;

fprintf('\n--- Top 20 Most Confused Pairs ---\n');
fprintf('%-15s  %-15s  %s\n', 'True', 'Predicted', 'Count');
fprintf('%s\n', repmat('-',1,40));
for k = 1:20
    [maxVal, flatIdx] = max(Coff(:));
    if maxVal == 0, break; end
    [r, c] = ind2sub([numC numC], flatIdx);
    fprintf('%-15s  %-15s  %d\n', allClasses{r}, allClasses{c}, maxVal);
    Coff(r,c) = 0;
end