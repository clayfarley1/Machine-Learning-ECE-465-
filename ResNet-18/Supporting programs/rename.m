% Define the parent directories for your 4-band and 5-band data.
% UPDATE THESE PATHS to the actual locations of your folders.
parentFolders = {
    'C:\AI\data1\4-band', 
    'C:\AI\data1\5-band'
};

for p = 1:length(parentFolders)
    currentParent = parentFolders{p};
    
    % Check if the directory exists to prevent errors
    if ~isfolder(currentParent)
        warning('Directory does not exist: %s', currentParent);
        continue;
    end
    
    % Get a list of all subfolders inside the current parent folder
    subfolders = dir(currentParent);
    subfolders = subfolders([subfolders.isdir]); % Keep only directories
    subfolders = subfolders(~ismember({subfolders.name}, {'.', '..'})); % Remove . and ..
    
    for s = 1:length(subfolders)
        currentSubfolderName = subfolders(s).name;
        currentSubfolderPath = fullfile(currentParent, currentSubfolderName);
        
        % Extract the resistor value from the folder name.
        % The format is "4B-1K2-T5", so we split by '-' and take the 2nd part.
        nameParts = split(currentSubfolderName, '-');
        if length(nameParts) >= 2
            % Convert to lowercase to match your "1k_1" example
            resistorValue = lower(nameParts{2}); 
        else
            resistorValue = 'unknown';
        end
        
        % Get all jpg images in this subfolder. 
        % (Change to '*.png' if you have PNGs, or use a loop for multiple types)
        images = dir(fullfile(currentSubfolderPath, '*.jpg'));
        
        for i = 1:length(images)
            oldName = images(i).name;
            oldPath = fullfile(currentSubfolderPath, oldName);
            
            % Create the new name: e.g., 1k2_1.jpg
            newName = sprintf('%s_%d.jpg', resistorValue, i);
            newPath = fullfile(currentSubfolderPath, newName);
            
            % Rename the file
            % Note: movefile can be slow on large datasets, but it's safe.
            if ~strcmp(oldName, newName)
                movefile(oldPath, newPath);
            end
        end
        
        fprintf('Renamed %d images in folder: %s\n', length(images), currentSubfolderName);
    end
end

disp('Renaming complete!');