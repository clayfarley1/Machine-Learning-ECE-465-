%% =========================
%% Resistor Prediction Script
%% =========================

clear; clc; close all;

%% -------------------------
%% Paths to saved models
%% -------------------------
modelFolder = 'C:\AI\resistor_model_hierarchical'; % <--- update if needed

stage1File = fullfile(modelFolder, 'stage1_band_type.mat');
stage2aFile = fullfile(modelFolder, 'stage2a_4band.mat');
stage2bFile = fullfile(modelFolder, 'stage2b_5band.mat');

%% -------------------------
%% Load Models
%% -------------------------
fprintf('Loading Stage 1 model...\n');
load(stage1File, 'trainedNet_s1');

fprintf('Loading Stage 2a model (4-Band)...\n');
if isfile(stage2aFile)
    load(stage2aFile, 'trainedNet_s2a');
else
    trainedNet_s2a = [];
end

fprintf('Loading Stage 2b model (5-Band)...\n');
if isfile(stage2bFile)
    load(stage2bFile, 'trainedNet_s2b');
else
    trainedNet_s2b = [];
end

%% -------------------------
%% Select Resistor Image
%% -------------------------
[fileName, pathName] = uigetfile({'*.jpg;*.png;*.jpeg;*.bmp','Image Files'}, ...
    'Select a Resistor Image');

if isequal(fileName,0)
    disp('No file selected. Exiting.');
    return;
end

imgPath = fullfile(pathName, fileName);
img = imread(imgPath);

inputSize = trainedNet_s1.Layers(1).InputSize(1:2);
imgResized = imresize(img, inputSize);

%% -------------------------
%% Stage 1: Band Type Detection
%% -------------------------
[bandPred, scores1] = classify(trainedNet_s1, imgResized);
bandConfidence = max(scores1);

fprintf('\nDetected Band Type: %s (Confidence: %.1f%%)\n', string(bandPred), bandConfidence*100);

%% -------------------------
%% Stage 2: Value Prediction
%% -------------------------
switch string(bandPred)
    case '4Band'
        if isempty(trainedNet_s2a)
            warning('No 4-Band model found. Cannot predict value.');
            return;
        end
        [valPred, scores2] = classify(trainedNet_s2a, imgResized);
    case '5Band'
        if isempty(trainedNet_s2b)
            warning('No 5-Band model found. Cannot predict value.');
            return;
        end
        [valPred, scores2] = classify(trainedNet_s2b, imgResized);
    otherwise
        warning('Unknown band type, cannot predict value.');
        return;
end

valueConfidence = max(scores2);
valueLabel = string(valPred);
fprintf('Predicted Resistor Label: %s (Confidence: %.1f%%)\n', valueLabel, valueConfidence*100);

%% -------------------------
%% Convert Resistor Label to Ohms
%% -------------------------
valueStr = regexprep(valueLabel, '^[45]B-', ''); % remove prefix
ohms = NaN;

try
    if contains(valueStr,'k','IgnoreCase',true)
        % 4.7k or 4k7 formats
        if contains(valueStr,'.')
            ohms = str2double(strrep(valueStr,'k',''))*1e3; % 4.7k -> 4700
        else
            % handle '4k7' -> 4700
            parts = regexp(valueStr,'(\d+)k(\d+)','tokens');
            if ~isempty(parts)
                nums = parts{1};
                ohms = str2double([nums{1} '.' nums{2}])*1e3;
            else
                % fallback: replace k with . then convert
                ohms = str2double(strrep(valueStr,'k','.'))*1e3;
            end
        end
    elseif contains(valueStr,'M','IgnoreCase',true)
        ohms = str2double(strrep(valueStr,'M',''))*1e6;
    else
        ohms = str2double(valueStr);
    end
catch
    warning('Could not parse ohm value from label: %s', valueStr);
end

if isnan(ohms)
    fprintf('Could not convert label to numeric ohms. Label: %s\n', valueStr);
else
    fprintf('Estimated Resistance: %.0f Ω\n', ohms);
end

%% -------------------------
%% Display Image with Prediction
%% -------------------------
figure;
imshow(img);
if ~isnan(ohms)
    title(sprintf('%s\n%.0f Ω (%.1f%%)', string(bandPred), ohms, valueConfidence*100));
else
    title(sprintf('%s\n%s (%.1f%%)', string(bandPred), valueLabel, valueConfidence*100));
end