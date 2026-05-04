% Set input and output folders
inputFolder = 'C:\Users\messi\OneDrive\Desktop\Test Resistors\4 Band';
outputFolder = 'C:\Users\messi\OneDrive\Desktop\Test Resistors\4 Band';

% Create output folder if it doesn't exist
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Get all .webp files in the folder
files = dir(fullfile(inputFolder, '*.webp'));

% Loop through each file
for k = 1:length(files)
    % Full path to input file
    inputFile = fullfile(inputFolder, files(k).name);
    
    % Read the image
    img = imread(inputFile);
    
    % Create output file name (change extension to .jpg)
    [~, name, ~] = fileparts(files(k).name);
    outputFile = fullfile(outputFolder, [name, '.jpg']);
    
    % Write as JPG
    imwrite(img, outputFile, 'jpg');
    
    fprintf('Converted: %s -> %s\n', files(k).name, [name, '.jpg']);
end

disp('Conversion complete!');