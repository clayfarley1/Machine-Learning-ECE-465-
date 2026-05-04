%% Resistor Confusion Matrix - Robust Version
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

%% Create datastore (ONLY valid image extensions)
imds = imageDatastore(rootFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'FileExtensions', {'.jpg','.jpeg','.png','.bmp'});

fprintf('Initial file count: %d\n', numel(imds.Files));

%% -----------------------------------------------------------------------
%% STEP 1: REMOVE CORRUPTED / INVALID IMAGES
%% -----------------------------------------------------------------------
validMask = true(numel(imds.Files),1);

for k = 1:numel(imds.Files)
    try
        imfinfo(imds.Files{k}); % quick validation
    catch
        validMask(k) = false;
        fprintf('Removing bad file: %s\n', imds.Files{k});
    end
end

imds.Files  = imds.Files(validMask);
imds.Labels = imds.Labels(validMask);

fprintf('Valid file count: %d\n', numel(imds.Files));

nImages = numel(imds.Files);

%% Labels
origLabels = string(imds.Labels);

trueLabels = regexprep(origLabels, '^[45][Bb]-', '');
trueLabels = regexprep(trueLabels, '-\w+$', '');

predLabels = strings(nImages, 1);

%% -----------------------------------------------------------------------
%% STEP 2: SAFE PREDICTION LOOP
%% -----------------------------------------------------------------------
fprintf('Running predictions...\n');

validPredMask = true(nImages,1);

for i = 1:nImages
    try
        img = readimage(imds, i);
    catch
        warning('Skipping unreadable file: %s', imds.Files{i});
        validPredMask(i) = false;
        continue;
    end

    % Ensure 3 channels
    if size(img,3) == 1
        img = cat(3, img, img, img);
    end

    img = imresize(img, [224 224]);

    % Band type classifier
    [bandPred, ~] = classify(bandTypeNet, img);
    bandType = string(bandPred);

    % Select correct model
    if contains(bandType, '4', 'IgnoreCase', true)
        targetNet = net4Band;
    else
        targetNet = net5Band;
    end

    % Value classifier
    [valPred, ~]  = classify(targetNet, img);
    predLabels(i) = string(valPred);

    if mod(i,50)==0 || i==nImages
        fprintf('  %d / %d\n', i, nImages);
    end
end

%% Remove failed predictions
trueLabels = trueLabels(validPredMask);
predLabels = predLabels(validPredMask);
origLabels = origLabels(validPredMask);

nImages = numel(trueLabels);

%% -----------------------------------------------------------------------
%% STEP 3: CONFUSION MATRIX
%% -----------------------------------------------------------------------
allClasses = unique([trueLabels; predLabels]);
numC       = numel(allClasses);

trueIdx = arrayfun(@(x) find(strcmp(allClasses,x)), trueLabels);
predIdx = arrayfun(@(x) find(strcmp(allClasses,x)), predLabels);

C = accumarray([trueIdx, predIdx], 1, [numC numC]);

accuracy = sum(diag(C)) / sum(C(:)) * 100;

fprintf('\nOverall Accuracy: %.2f%%\n', accuracy);

%% Determine band type per class
isFiveBand = false(numC, 1);
for k = 1:numC
    matchMask = (trueIdx == k);
    if any(matchMask)
        sampOrig = origLabels(matchMask);
        isFiveBand(k) = any(contains(sampOrig, {'5B','5b','5-'}, 'IgnoreCase', true));
    end
end

%% -----------------------------------------------------------------------
%% FIGURE 1: HEATMAP
%% -----------------------------------------------------------------------
Cnorm = 100 * C ./ max(sum(C,2), 1);

COL_CORRECT = [0.08 0.55 0.15];
COL_ERROR   = [0.85 0.12 0.12];
COL_EMPTY   = [1 1 1];

rgb = repmat(reshape(COL_EMPTY,1,1,3), numC, numC);

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

imshow(rgb, 'Parent', ax);

if numC <= 60
    set(ax, 'XTick', 1:numC, 'XTickLabel', allClasses, 'XTickLabelRotation', 90, ...
            'YTick', 1:numC, 'YTickLabel', allClasses, 'FontSize', 7);
else
    set(ax, 'XTick', [], 'YTick', []);
end

xlabel(ax, 'Predicted Class');
ylabel(ax, 'True Class');

title(ax, { ...
    sprintf('Resistor Recognition — Confusion Matrix | Accuracy: %.2f%%', accuracy), ...
    sprintf('n = %d images · %d classes', nImages, numC)});

saveas(fig1, fullfile(rootFolder, 'confusion_heatmap.png'));

%% -----------------------------------------------------------------------
%% FIGURE 2: PER-CLASS ACCURACY
%% -----------------------------------------------------------------------
COL_4BAND = [0.18 0.49 0.80];
COL_5BAND = [0.92 0.95 1.00];

classAcc = diag(C) ./ max(sum(C,2), 1) * 100;

barColors = zeros(numC,3);
for k = 1:numC
    if isFiveBand(k)
        barColors(k,:) = COL_5BAND;
    else
        barColors(k,:) = COL_4BAND;
    end
end

fig2 = figure('Name','Per-Class Accuracy','NumberTitle','off','Position',[100 100 1200 540]);
ax2  = axes('Parent', fig2);

b = bar(ax2, classAcc, 'FaceColor', 'flat');
b.CData = barColors;

hold(ax2,'on');
yline(accuracy,'r--','LineWidth',1.5);
hold(ax2,'off');

if numC <= 80
    set(ax2, 'XTick', 1:numC, 'XTickLabel', allClasses, 'XTickLabelRotation', 90);
else
    set(ax2, 'XTick', []);
end

ylabel(ax2,'Accuracy (%)');
ylim(ax2,[0 105]);

title(ax2, sprintf('Per-Class Accuracy | Overall %.2f%%', accuracy));

grid(ax2,'on');

saveas(fig2, fullfile(rootFolder, 'accuracy_per_class.png'));

%% -----------------------------------------------------------------------
%% TOP CONFUSIONS
%% -----------------------------------------------------------------------
Coff = C;
Coff(eye(numC)==1) = 0;

fprintf('\n--- Top 20 Confusions ---\n');
for k = 1:20
    [val, idx] = max(Coff(:));
    if val == 0, break; end
    [r,c] = ind2sub(size(Coff), idx);

    fprintf('%s -> %s : %d\n', allClasses{r}, allClasses{c}, val);
    Coff(r,c) = 0;
end