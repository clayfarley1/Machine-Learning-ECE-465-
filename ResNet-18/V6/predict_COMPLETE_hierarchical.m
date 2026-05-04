%% COMPLETE Hierarchical Resistor Prediction System
% Uses Stage 1 + Stage 2 models to identify resistor values

clear; clc; close all;

%% 1. Configuration
datasetPath = 'path/to/resistor_dataset'; % UPDATE THIS

[datasetDir, ~, ~] = fileparts(datasetPath);
if isempty(datasetDir)
    datasetDir = pwd;
end
modelSavePath = fullfile(datasetDir, 'resistor_model_hierarchical');

%% 2. Load All Models
fprintf('========================================\n');
fprintf('  Loading Hierarchical Models\n');
fprintf('========================================\n\n');

% Stage 1: Band Type
fprintf('Loading Stage 1 (Band Type)...\n');
load(fullfile(modelSavePath, 'stage1_band_type.mat'), 'trainedNet_s1');
load(fullfile(modelSavePath, 'stage1_config.mat'), 'config_s1');
fprintf('✓ Stage 1 loaded (%.1f%% accuracy)\n', config_s1.accuracy * 100);

% Stage 2a: 4-Band
if exist(fullfile(modelSavePath, 'stage2a_4band.mat'), 'file')
    fprintf('Loading Stage 2a (4-Band)...\n');
    load(fullfile(modelSavePath, 'stage2a_4band.mat'), 'trainedNet_s2a');
    load(fullfile(modelSavePath, 'stage2a_config.mat'), 'config_s2a');
    fprintf('✓ Stage 2a loaded (%.1f%% accuracy)\n', config_s2a.accuracy * 100);
    has4Band = true;
else
    fprintf('⚠ No 4-Band model found\n');
    has4Band = false;
end

% Stage 2b: 5-Band
if exist(fullfile(modelSavePath, 'stage2b_5band.mat'), 'file')
    fprintf('Loading Stage 2b (5-Band)...\n');
    load(fullfile(modelSavePath, 'stage2b_5band.mat'), 'trainedNet_s2b');
    load(fullfile(modelSavePath, 'stage2b_config.mat'), 'config_s2b');
    fprintf('✓ Stage 2b loaded (%.1f%% accuracy)\n', config_s2b.accuracy * 100);
    has5Band = true;
else
    fprintf('⚠ No 5-Band model found\n');
    has5Band = false;
end

%% 3. GPU Setup
parallel.gpu.enableCUDAForwardCompatibility(true);

%% 4. Single Image Prediction
fprintf('\n========================================\n');
fprintf('  Single Image Prediction\n');
fprintf('========================================\n\n');

imagePath = 'path/to/test/image.jpg'; % UPDATE THIS

if exist(imagePath, 'file')
    % Read image
    img = imread(imagePath);
    originalImg = img;
    imgResized = imresize(img, [224 224]);
    
    fprintf('Analyzing: %s\n\n', imagePath);
    
    %% STAGE 1: Determine band type
    fprintf('--- STAGE 1 ---\n');
    [bandType, scores_s1] = classify(trainedNet_s1, imgResized, ...
        'ExecutionEnvironment', 'auto');
    
    fprintf('Band Type: %s (%.1f%% confidence)\n', ...
        string(bandType), max(scores_s1) * 100);
    
    %% STAGE 2: Get specific value
    fprintf('\n--- STAGE 2 ---\n');
    
    if strcmp(char(bandType), '4Band') && has4Band
        [finalValue, scores_s2] = classify(trainedNet_s2a, imgResized, ...
            'ExecutionEnvironment', 'auto');
        fprintf('Using 4-Band model...\n');
        
    elseif strcmp(char(bandType), '5Band') && has5Band
        [finalValue, scores_s2] = classify(trainedNet_s2b, imgResized, ...
            'ExecutionEnvironment', 'auto');
        fprintf('Using 5-Band model...\n');
        
    else
        fprintf('⚠ No Stage 2 model available for %s\n', string(bandType));
        finalValue = categorical({'Unknown'});
        scores_s2 = 0;
    end
    
    %% Display Results
    fprintf('\n========================================\n');
    fprintf('  FINAL PREDICTION\n');
    fprintf('========================================\n');
    fprintf('Resistor: %s\n', string(finalValue));
    fprintf('Confidence: %.1f%%\n', max(scores_s2) * 100);
    fprintf('Band Type: %s\n', string(bandType));
    fprintf('========================================\n\n');
    
    % Visual display
    figure('Position', [100 100 900 700]);
    
    % Show image
    subplot(2,2,[1 2]);
    imshow(originalImg);
    title(sprintf('Predicted: %s\nConfidence: %.1f%%', ...
        string(finalValue), max(scores_s2) * 100), ...
        'Interpreter', 'none', 'FontSize', 14, 'FontWeight', 'bold');
    
    % Stage 1 results
    subplot(2,2,3);
    bar(scores_s1 * 100);
    set(gca, 'XTickLabel', config_s1.classes);
    ylabel('Confidence (%)');
    title('Stage 1: Band Type');
    ylim([0 100]);
    grid on;
    
    % Stage 2 results (top 5)
    if max(scores_s2) > 0
        subplot(2,2,4);
        [sortedScores, sortedIdx] = sort(scores_s2, 'descend');
        
        if strcmp(char(bandType), '4Band') && has4Band
            classes_s2 = config_s2a.classes;
        elseif strcmp(char(bandType), '5Band') && has5Band
            classes_s2 = config_s2b.classes;
        else
            classes_s2 = {'Unknown'};
        end
        
        numShow = min(5, length(sortedScores));
        bar(sortedScores(1:numShow) * 100);
        set(gca, 'XTickLabel', classes_s2(sortedIdx(1:numShow)));
        xtickangle(45);
        ylabel('Confidence (%)');
        title('Stage 2: Top 5 Values');
        ylim([0 100]);
        grid on;
    end
    
    % Print top 5 predictions
    if max(scores_s2) > 0
        fprintf('Top 5 Predictions:\n');
        for i = 1:numShow
            fprintf('%d. %s: %.1f%%\n', i, ...
                string(classes_s2{sortedIdx(i)}), ...
                sortedScores(i) * 100);
        end
    end
    
else
    fprintf('Image not found: %s\n', imagePath);
    fprintf('Please update imagePath variable.\n');
end

%% 5. Batch Prediction (Optional)
fprintf('\n========================================\n');
fprintf('  Batch Prediction\n');
fprintf('========================================\n');
fprintf('To predict on multiple images, uncomment below\n\n');

% UNCOMMENT FOR BATCH PREDICTION:
% testFolder = 'path/to/test/folder';
% 
% if exist(testFolder, 'dir')
%     imdsTest = imageDatastore(testFolder);
%     numImages = numel(imdsTest.Files);
%     
%     fprintf('Processing %d images...\n', numImages);
%     
%     predictions = cell(numImages, 3);  % [filename, bandtype, value]
%     
%     for i = 1:numImages
%         img = readimage(imdsTest, i);
%         imgResized = imresize(img, [224 224]);
%         
%         % Stage 1
%         bandType = classify(trainedNet_s1, imgResized, 'ExecutionEnvironment', 'auto');
%         
%         % Stage 2
%         if strcmp(char(bandType), '4Band') && has4Band
%             [value, scores] = classify(trainedNet_s2a, imgResized, 'ExecutionEnvironment', 'auto');
%         elseif strcmp(char(bandType), '5Band') && has5Band
%             [value, scores] = classify(trainedNet_s2b, imgResized, 'ExecutionEnvironment', 'auto');
%         else
%             value = categorical({'Unknown'});
%             scores = 0;
%         end
%         
%         [~, filename] = fileparts(imdsTest.Files{i});
%         predictions{i,1} = filename;
%         predictions{i,2} = char(bandType);
%         predictions{i,3} = char(value);
%         predictions{i,4} = max(scores) * 100;
%         
%         if mod(i, 10) == 0
%             fprintf('  Processed %d/%d\n', i, numImages);
%         end
%     end
%     
%     % Save results
%     resultsTable = cell2table(predictions, ...
%         'VariableNames', {'Filename', 'BandType', 'Value', 'Confidence'});
%     
%     resultFile = fullfile(modelSavePath, 'batch_predictions.csv');
%     writetable(resultsTable, resultFile);
%     fprintf('\n✓ Results saved: %s\n', resultFile);
%     
%     % Display first 9
%     figure('Position', [100 100 1200 800]);
%     for i = 1:min(9, numImages)
%         subplot(3, 3, i);
%         img = readimage(imdsTest, i);
%         imshow(img);
%         title(sprintf('%s\n%.1f%%', predictions{i,3}, predictions{i,4}), ...
%             'Interpreter', 'none', 'FontSize', 9);
%     end
%     sgtitle('Batch Prediction Results');
% end

fprintf('\nPrediction system ready!\n');
fprintf('Update imagePath (line 39) to test your resistors.\n');
