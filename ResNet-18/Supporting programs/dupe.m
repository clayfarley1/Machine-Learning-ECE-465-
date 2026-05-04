% Define the parent directories for your 4-band and 5-band data.
% UPDATE THESE PATHS to the actual locations of your folders.
parentFolders = {
    'C:\AI\data1\4-band',
    'C:\AI\data1\5-band'
};

% =========================================================
%  AUGMENTATION SETTINGS  (tweak these to taste)
% =========================================================
augConfig.numCopies      = 3;           % extra copies per original
augConfig.maxRotation    = 174;          % degrees  (+/-)
augConfig.contrastRange  = [0.7 1.4];   % contrast scale factor
augConfig.brightnessRange= [-30 30];    % additive brightness offset (0-255)
augConfig.saturationRange= [0.6 1.4];   % saturation scale factor
augConfig.hueShiftRange  = [-15 15];    % hue shift in degrees
% =========================================================

for p = 1:length(parentFolders)
    currentParent = parentFolders{p};

    if ~isfolder(currentParent)
        warning('Directory does not exist: %s', currentParent);
        continue;
    end

    subfolders = dir(currentParent);
    subfolders = subfolders([subfolders.isdir]);
    subfolders = subfolders(~ismember({subfolders.name}, {'.', '..'}));

    for s = 1:length(subfolders)
        currentSubfolderName = subfolders(s).name;
        currentSubfolderPath = fullfile(currentParent, currentSubfolderName);

        % --- Extract resistor value from folder name ---
        nameParts = split(currentSubfolderName, '-');
        if length(nameParts) >= 2
            resistorValue = lower(nameParts{2});
        else
            resistorValue = 'unknown';
        end

        images = dir(fullfile(currentSubfolderPath, '*.jpg'));
        totalAug = 0;

        for i = 1:length(images)
            oldName = images(i).name;
            oldPath = fullfile(currentSubfolderPath, oldName);

            % --- 1. Rename original ---
            newName = sprintf('%s_%d.jpg', resistorValue, i);
            newPath = fullfile(currentSubfolderPath, newName);
            if ~strcmp(oldPath, newPath)
                movefile(oldPath, newPath);
            end

            % --- 2. Generate augmented copies ---
            img = imread(newPath);

            for c = 1:augConfig.numCopies
                augImg = applyAugmentation(img, augConfig);

                % Naming: e.g.  1k2_1_aug1.jpg
                augName = sprintf('%s_%d_aug%d.jpg', resistorValue, i, c);
                augPath = fullfile(currentSubfolderPath, augName);
                imwrite(augImg, augPath, 'Quality', 95);
                totalAug = totalAug + 1;
            end
        end

        fprintf('Folder: %-30s | Originals: %3d | Augmented copies added: %3d\n', ...
                currentSubfolderName, length(images), totalAug);
    end
end

disp('Renaming and augmentation complete!');


% =========================================================
%  LOCAL FUNCTION: applyAugmentation
%  Applies random rotation, contrast, brightness, hue, and
%  saturation changes to a single RGB image.
% =========================================================
function outImg = applyAugmentation(img, cfg)

    % --- Convert to double for maths ---
    imgD = im2double(img);   % values in [0,1]

    % ---- ROTATION ---------------------------------------------------
    angle = cfg.maxRotation * (2*rand - 1);   % uniform in [-max, +max]
    imgD  = imrotate(imgD, angle, 'bilinear', 'crop');

    % ---- CONTRAST ---------------------------------------------------
    % Scale around the mean luminance so highlights/shadows move together
    cf    = cfg.contrastRange(1) + diff(cfg.contrastRange) * rand;
    mu    = mean(imgD(:));
    imgD  = mu + cf .* (imgD - mu);

    % ---- BRIGHTNESS -------------------------------------------------
    bf    = (cfg.brightnessRange(1) + diff(cfg.brightnessRange) * rand) / 255;
    imgD  = imgD + bf;

    % Clamp to [0,1] before HSV conversion
    imgD  = max(0, min(1, imgD));

    % ---- HUE & SATURATION  (work in HSV space) ----------------------
    if size(imgD, 3) == 3   % only for RGB
        hsv   = rgb2hsv(imgD);

        % Hue shift (circular, mapped to [0,1])
        hShift      = (cfg.hueShiftRange(1) + diff(cfg.hueShiftRange) * rand) / 360;
        hsv(:,:,1)  = mod(hsv(:,:,1) + hShift, 1);

        % Saturation scale
        sf          = cfg.saturationRange(1) + diff(cfg.saturationRange) * rand;
        hsv(:,:,2)  = max(0, min(1, hsv(:,:,2) .* sf));

        imgD = hsv2rgb(hsv);
    end

    % Final clamp & convert back to uint8
    imgD   = max(0, min(1, imgD));
    outImg = im2uint8(imgD);
end