%% combine_datasets.m
% Combines two resistor image datasets into one output folder.
%
% DATASET 1 (e.g. folders like "4K7_1-4W"):
%   - Images are CENTER-CROPPED by 30% (15% trimmed from each side)
%   - Folders renamed:  {VALUE}_{WATTAGE}  ->  4B-{VALUE}-T5
%   - Images renumbered sequentially:  {value}_1.jpg, {value}_2.jpg ...
%
% DATASET 2 (e.g. folders like "4B-1K2-T5"):
%   - Copied COMPLETELY UNTOUCHED (folders and filenames kept as-is)
%
% USAGE:
%   1. Run the script
%   2. A dialog will ask you to select:
%        - Dataset 1 root folder
%        - Dataset 2 root folder
%        - Output folder
%   3. Done!

function combine_datasets()

    %% ── Select folders via dialogs ───────────────────────────────────────
    fprintf('Select Dataset 1 root folder (e.g. contains 4K7_1-4W, 1K_2W ...):\n');
    ds1_root = uigetdir(pwd, 'Select Dataset 1 Root Folder');
    if ds1_root == 0
        error('No folder selected for Dataset 1. Aborted.');
    end

    fprintf('Select Dataset 2 root folder (e.g. contains 4B-1K2-T5, 4B-1K5-T5 ...):\n');
    ds2_root = uigetdir(pwd, 'Select Dataset 2 Root Folder');
    if ds2_root == 0
        error('No folder selected for Dataset 2. Aborted.');
    end

    fprintf('Select OUTPUT folder:\n');
    out_root = uigetdir(pwd, 'Select Output Folder');
    if out_root == 0
        error('No output folder selected. Aborted.');
    end

    %% ── Dataset 2: copy untouched ────────────────────────────────────────
    fprintf('\n── Processing Dataset 2 (untouched) ──\n');
    ds2_folders = dir(ds2_root);
    ds2_folders = ds2_folders([ds2_folders.isdir] & ~startsWith({ds2_folders.name}, '.'));

    for i = 1:numel(ds2_folders)
        src  = fullfile(ds2_root, ds2_folders(i).name);
        dest = fullfile(out_root, ds2_folders(i).name);
        copyFolder(src, dest);
        fprintf('  Copied DS2: %s\n', ds2_folders(i).name);
    end

    %% ── Dataset 1: rename + crop ─────────────────────────────────────────
    fprintf('\n── Processing Dataset 1 (crop 30%% + rename) ──\n');
    ds1_folders = dir(ds1_root);
    ds1_folders = ds1_folders([ds1_folders.isdir] & ~startsWith({ds1_folders.name}, '.'));

    image_exts = {'.jpg','.jpeg','.png','.bmp','.tiff','.tif','.webp'};

    for i = 1:numel(ds1_folders)
        old_name   = ds1_folders(i).name;
        new_name   = ds1ToDs2Name(old_name);         % e.g. 4K7_1-4W -> 4B-4K7-T5
        value_part = extractValuePart(old_name);     % e.g. 4K7
        dest_dir   = fullfile(out_root, new_name);

        if ~exist(dest_dir, 'dir')
            mkdir(dest_dir);
        end

        % Gather all images in this DS1 folder
        src_dir = fullfile(ds1_root, old_name);
        img_files = [];
        for e = 1:numel(image_exts)
            found = dir(fullfile(src_dir, ['*' image_exts{e}]));
            img_files = [img_files; found]; %#ok<AGROW>
        end

        % Sort by filename so numbering is consistent
        [~, sort_idx] = sort({img_files.name});
        img_files = img_files(sort_idx);

        counter = 1;
        for j = 1:numel(img_files)
            src_path = fullfile(src_dir, img_files(j).name);

            % Read image
            try
                img = imread(src_path);
            catch
                fprintf('    [SKIP] Could not read: %s\n', img_files(j).name);
                continue;
            end

            % Center-crop by 30%
            img_cropped = centerCrop(img, 0.30);

            % New filename: {value_lower}_{counter}.jpg
            new_filename = sprintf('%s_%d.jpg', lower(value_part), counter);
            dest_path    = fullfile(dest_dir, new_filename);

            % Write as JPEG (quality 95)
            imwrite(img_cropped, dest_path, 'jpg', 'Quality', 95);
            counter = counter + 1;
        end

        fprintf('  Processed DS1: %-20s  ->  %-20s  (%d images)\n', ...
            old_name, new_name, counter - 1);
    end

    fprintf('\nDone! Combined dataset saved to:\n  %s\n', out_root);
    msgbox(sprintf('Done!\n\nCombined dataset saved to:\n%s', out_root), ...
        'combine_datasets', 'help');
end


%% ═══════════════════════════════════════════════════════
%  Helper: convert DS1 folder name  ->  DS2 folder name
%  e.g.  '4K7_1-4W'  ->  '4B-4K7-T5'
%        '1K_2W'     ->  '4B-1K-T5'
%        '10_1-4W'   ->  '4B-10-T5'
%% ═══════════════════════════════════════════════════════
function new_name = ds1ToDs2Name(folder_name)
    value_part = extractValuePart(folder_name);
    new_name = ['4B-' value_part '-T5'];
end


%% ═══════════════════════════════════════════════════════
%  Helper: extract resistance value from DS1 folder name
%  (everything before the first underscore)
%% ═══════════════════════════════════════════════════════
function value = extractValuePart(folder_name)
    parts = strsplit(folder_name, '_');
    value = parts{1};
end


%% ═══════════════════════════════════════════════════════
%  Helper: center-crop an image by `frac` of each dimension
%  frac = 0.30 removes 15% from each of the 4 sides
%  Result is (1-frac) * original size in W and H
%% ═══════════════════════════════════════════════════════
function img_out = centerCrop(img, frac)
    [h, w, ~] = size(img);
    trim_x = round(w * frac / 2);
    trim_y = round(h * frac / 2);
    x1 = trim_x + 1;
    y1 = trim_y + 1;
    x2 = w - trim_x;
    y2 = h - trim_y;
    img_out = img(y1:y2, x1:x2, :);
end


%% ═══════════════════════════════════════════════════════
%  Helper: recursively copy a folder
%% ═══════════════════════════════════════════════════════
function copyFolder(src, dest)
    if ~exist(dest, 'dir')
        mkdir(dest);
    end
    items = dir(src);
    for k = 1:numel(items)
        if startsWith(items(k).name, '.')
            continue;
        end
        s = fullfile(src, items(k).name);
        d = fullfile(dest, items(k).name);
        if items(k).isdir
            copyFolder(s, d);
        else
            copyfile(s, d);
        end
    end
end