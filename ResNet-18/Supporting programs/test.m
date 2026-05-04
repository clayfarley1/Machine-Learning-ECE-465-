%% Load image
imgPath = 'C:\AI\data1\4-band\4B-1R-T5\4B-1R-T5-4.jpg';
img = imread(imgPath);
inputSize = trainedNet_s1.Layers(1).InputSize(1:2);
imgResized = imresize(img, inputSize);

%% Stage 1 prediction
[bandPred, ~] = classify(trainedNet_s1, imgResized);
disp(['Predicted Band Type: ', string(bandPred)]);

%% Stage 2 prediction
switch string(bandPred)
    case '4Band'
        [valPred, ~] = classify(trainedNet_s2a, imgResized);
    case '5Band'
        [valPred, ~] = classify(trainedNet_s2b, imgResized);
end

valueLabel = string(valPred);
disp(['Predicted Resistor Label: ', valueLabel]);

%% Parse to ohms
valueStr = regexprep(valueLabel, '^[45]B-', ''); 
ohms = NaN;

if contains(valueStr,'k','IgnoreCase',true)
    if contains(valueStr,'.')
        ohms = str2double(strrep(valueStr,'k',''))*1e3;
    else
        parts = regexp(valueStr,'(\d+)k(\d+)','tokens');
        if ~isempty(parts)
            nums = parts{1};
            ohms = str2double([nums{1} '.' nums{2}])*1e3;
        else
            ohms = str2double(strrep(valueStr,'k','.'))*1e3;
        end
    end
elseif contains(valueStr,'M','IgnoreCase',true)
    ohms = str2double(strrep(valueStr,'M',''))*1e6;
else
    ohms = str2double(valueStr);
end

disp(['Estimated Resistance: ', num2str(ohms), ' Ω']);