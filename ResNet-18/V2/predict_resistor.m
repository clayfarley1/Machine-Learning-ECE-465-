%% Resistor Identification - Prediction Script
% This script uses the trained model to identify resistor values from new images

clear; clc; close all;

%% 1. Load the Trained Model
fprintf('Loading trained model...\n');
load('resistor_identification_model.mat', 'trainedNet');
load('model_config.mat', 'config');

fprintf('Model loaded successfully!\n');
fprintf('Number of classes: %d\n', config.numClasses);
fprintf('Classes: %s\n', strjoin(string(config.classNames), ', '));

%% 2. Predict Single Image
% Method 1: Load and predict a single image
imagePath = 'path/to/test/resistor/image.jpg'; % UPDATE THIS PATH

if exist(imagePath, 'file')
    % Read and preprocess image
    img = imread(imagePath);
    img = imresize(img, config.inputSize(1:2));
    
    % Make prediction
    [predictedLabel, scores] = classify(trainedNet, img);
    
    % Display results
    figure;
    imshow(img);
    title(sprintf('Predicted: %s\nConfidence: %.2f%%', ...
        string(predictedLabel), max(scores) * 100), 'Interpreter', 'none');
    
    % Display top 5 predictions
    [sortedScores, sortedIdx] = sort(scores, 'descend');
    fprintf('\n=== Prediction Results ===\n');
    fprintf('Top 5 Predictions:\n');
    numToShow = min(5, length(sortedScores));
    for i = 1:numToShow
        fprintf('%d. %s: %.2f%%\n', i, ...
            string(config.classNames(sortedIdx(i))), ...
            sortedScores(i) * 100);
    end
    
    % Parse resistor value from prediction if possible
    predStr = char(predictedLabel);
    fprintf('\nPredicted Resistor Type: %s\n', predStr);
    
    % Extract approximate resistance value for common formats
    if contains(predStr, 'K', 'IgnoreCase', true)
        fprintf('Approximate Value: In kilohm (kΩ) range\n');
    elseif contains(predStr, 'M', 'IgnoreCase', true)
        fprintf('Approximate Value: In megohm (MΩ) range\n');
    elseif contains(predStr, 'R', 'IgnoreCase', true)
        fprintf('Approximate Value: In ohm (Ω) range\n');
    end
else
    fprintf('Test image not found. Please update the imagePath variable.\n');
end

%% 3. Batch Prediction on Multiple Images
% Uncomment this section to predict on a folder of images

% testFolder = 'path/to/test/images/folder'; % UPDATE THIS PATH
% 
% if exist(testFolder, 'dir')
%     imdsTest = imageDatastore(testFolder);
%     
%     % Prepare test images
%     augimdsTest = augmentedImageDatastore(config.inputSize(1:2), imdsTest, ...
%         'ColorPreprocessing', 'gray2rgb');
%     
%     % Predict
%     [predictedLabels, scores] = classify(trainedNet, augimdsTest);
%     
%     % Display results for each image
%     numImages = numel(imdsTest.Files);
%     
%     figure;
%     for i = 1:min(9, numImages) % Show first 9 images
%         subplot(3, 3, i);
%         img = readimage(imdsTest, i);
%         imshow(img);
%         title(sprintf('%s\n%.1f%%', string(predictedLabels(i)), max(scores(i,:))*100));
%     end
%     
%     % Export predictions to CSV
%     [~, fileNames, ~] = fileparts(imdsTest.Files);
%     resultsTable = table(fileNames', predictedLabels, max(scores, [], 2), ...
%         'VariableNames', {'FileName', 'PredictedValue', 'Confidence'});
%     writetable(resultsTable, 'prediction_results.csv');
%     fprintf('\nResults saved to prediction_results.csv\n');
% end

%% 4. Real-time Webcam Prediction (Optional)
% Uncomment this section for real-time prediction using webcam

% fprintf('\nStarting webcam prediction. Press Ctrl+C to stop.\n');
% cam = webcam;
% 
% figure;
% while true
%     % Capture frame
%     img = snapshot(cam);
%     img = imresize(img, config.inputSize(1:2));
%     
%     % Predict
%     [predictedLabel, scores] = classify(trainedNet, img);
%     
%     % Display
%     imshow(img);
%     title(sprintf('Predicted: %s (%.1f%%)', ...
%         string(predictedLabel), max(scores) * 100));
%     drawnow;
%     
%     pause(0.1);
% end
% 
% clear cam;

fprintf('\nPrediction complete!\n');
