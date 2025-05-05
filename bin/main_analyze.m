%% Analyze single folder
clc; clear; close all;
addpath(genpath(".\"));

folderpath_images = ".\Materials\CUBIC\kidney\250um\skrawki 2-5";
folderpath_masks = ".\Materials\results_mask\kidney\250um\skrawki 2-5";
analyzeFolderPair(folderpath_images, ...
                  folderpath_masks);
%% Analyze all folders in a path
clc; clear; close all;
addpath(genpath("."));

% === SETTINGS ===
root_images_dir = '.\Materials\CUBIC';
root_masks_dir  = '.\Materials\results_mask';
excluded_folders = {'odrzucone'};  % Folders to exclude

% === FIND DEEPEST FOLDERS ===
image_dirs = findDeepestDirs(root_images_dir, excluded_folders);
mask_dirs  = findDeepestDirs(root_masks_dir,  excluded_folders);

% === MATCH AND ANALYZE ===
for i = 1:length(image_dirs)
    img_path = image_dirs{i};
    best_match = '';
    best_score = 0;

    for j = 1:length(mask_dirs)
        mask_path = mask_dirs{j};
        score = countMatchingPathEnd(img_path, mask_path);
        if score > best_score
            best_score = score;
            best_match = mask_path;
        end
    end

    if best_score > 0
        fprintf('Processing pair:\n  Image folder: %s\n  Mask folder:  %s\n\n', img_path, best_match);
        analyzeFolderPair(img_path, best_match);
    else
        fprintf('No match found for image folder: %s\n', img_path);
    end
end

%% Functions
function analyzeFolderPair(path_images, path_masks)
%ANALYZEFOLDERPAIR Processes image and mask folders to compute metrics and create animations.
%   analyzeFolderPair(path_images, path_masks)
%   - path_images (string): Folder containing original images.
%   - path_masks  (string): Folder containing corresponding masks.
%   Matches images with masks, computes area, contrast, and transmittance metrics,
%   displays metric plots, and generates an animation saved under a common name.
%
%   Example:
%       % Analyze images in "data/imgs" and masks in "data/masks"
%       analyzeFolderPair("data/imgs", "data/masks");
    filename_save = extractCommonName(path_images, path_masks);

    paths_pairs = findImageMaskPairs(path_images, path_masks);
    paths_pairs = sort_image_mask_struct(paths_pairs);

    RADIUS_DILAT = 20;

    metrics_array = calcMetricsOfAllImages(paths_pairs, RADIUS_DILAT);

    image_extensions = {'*.png','*.jpg','*.jpeg','*.tiff','*.bmp','*.gif'};
    all_imgs = [];
    for ext = image_extensions
        all_imgs = [all_imgs; dir(fullfile(path_images, ext{1}))];
    end
    all_paths = fullfile(path_images, {all_imgs.name});


    baseline = computeBaselineFromMetadata(all_paths);
    times_days_array = getTimeFromMetadatas(string({paths_pairs.image_path}));
    times_days_array = times_days_array - baseline;

    displayMetrics(metrics_array, times_days_array);

    createAnimationOfObjectDetection(filename_save, paths_pairs, metrics_array, times_days_array);
end

function common_name = extractCommonName(path1, path2)
%EXTRACTCOMMONNAME Generates a common name from two file or folder paths.
%   common_name = extractCommonName(path1, path2)
%   - path1, path2 (string): Paths to compare.
%   Finds the longest shared trailing segments or concatenates base names if none match.
%
%   Example:
%       % Should return "session1"
%       common_name = extractCommonName("C:/proj/session1/imgs", "C:/proj/session1/masks");

    s1 = fliplr(strsplit(path1, filesep));
    s2 = fliplr(strsplit(path2, filesep));
    n = min(length(s1), length(s2));
    common_parts = {};
    for i = 1:n
        if strcmp(s1{i}, s2{i})
            common_parts{end+1} = s1{i};  % dodaj do końca
        else
            break;
        end
    end
    if isempty(common_parts)
        [~, name1] = fileparts(path1);
        [~, name2] = fileparts(path2);
        common_name = [name1, '_', name2];
    else
        common_parts = fliplr(common_parts);  % przywróć oryginalną kolejność
        common_name = strjoin(common_parts, ' ');
    end
end

function file_pairs = findImageMaskPairs(folder_images, folder_masks, mask_suffix)
%FINDIMAGEMASKPAIRS Retrieves matched image and mask file pairs.
%   file_pairs = findImageMaskPairs(folder_images, folder_masks, mask_suffix)
%   - folder_images (string): Path to image folder.
%   - folder_masks  (string): Path to mask folder.
%   - mask_suffix   (string, optional): Suffix identifying mask filenames (default '_mask').
%   Returns struct array with fields:
%       .image_path
%       .mask_path
%
%   Example:
%       pairs = findImageMaskPairs("imgs", "masks", "_seg");

    % Set default mask suffix if not provided
    if nargin < 3
        mask_suffix = '_mask';    % Default mask suffix
    end

    % List of possible file extensions
    image_extensions = {'*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp', '*.gif'};
    mask_extensions = {'*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp', '*.gif'};

    % Initialize a structure to store results
    file_pairs = struct('image_path', {}, 'mask_path', {});

    % Initialize a list to store image files
    image_files = [];
    for ext = image_extensions
        image_files = [image_files; dir(fullfile(folder_images, ext{1}))];
    end

    % Initialize a list to store mask files
    mask_files = [];
    for ext = mask_extensions
        mask_files = [mask_files; dir(fullfile(folder_masks, ext{1}))];
    end

    % Map mask names to their full paths for quick lookup
    mask_map = containers.Map();
    for i = 1:length(mask_files)
        [~, mask_name, mask_ext] = fileparts(mask_files(i).name);
        mask_map([mask_name, mask_ext]) = fullfile(folder_masks, mask_files(i).name);
    end

    % Iterate over all image files
    for i = 1:length(image_files)
        % Get the image filename without extension
        [~, filename, ~] = fileparts(image_files(i).name);

        % Construct the corresponding mask name
        mask_name = [filename, mask_suffix];

        % Check all possible mask extensions
        found_mask = false;
        for ext = mask_extensions
            mask_filename = [mask_name, ext{1}(2:end)]; % Remove '*' from extension
            if mask_map.isKey(mask_filename)
                found_mask = true;
                % Add the image and mask paths to the structure
                file_pairs(end+1).image_path = fullfile(folder_images, image_files(i).name);
                file_pairs(end).mask_path = mask_map(mask_filename);
                break;
            end
        end

        % If no mask is found, skip this image
        if ~found_mask
            continue;
        end
    end
end

function binary_mask = processMask(mask, target_size)
%PROCESSMASK Processes and binarizes a mask image.
%   binary_mask = processMask(mask, target_size)
%   - mask        (matrix): Input mask image (grayscale or RGB).
%   - target_size (1×2 vector): Desired output size [rows, cols].
%   Converts to grayscale if needed, resizes, then binarizes.
%
%   Example:
%       img = imread("mask.png");
%       bin = processMask(img, [256,256]);

    % If the mask is color, convert it to grayscale
    if size(mask, 3) == 3
        mask = rgb2gray(mask);
    end
    
    % Resize the mask to the target size
    resized_mask = imresize(mask, target_size);
    
    % Binarize the mask
    binary_mask = imbinarize(resized_mask);
end

function displayImageWithMaskContour(image_path, mask_path)
%DISPLAYIMAGEWITHMASKCONTOUR Shows image with mask boundary overlay.
%   displayImageWithMaskContour(image_path, mask_path)
%   - image_path (string): Path to the image file.
%   - mask_path  (string): Path to the mask file.
%   Loads image and mask, processes the mask, finds contours, and overlays them.
%
%   Example:
%       displayImageWithMaskContour("img.jpg", "img_mask.png");

    % Load the original image
    original_image = imread(image_path);
    
    % Load the mask
    mask = imread(mask_path);
    
    % Process the mask (using a helper function)
    binary_mask = processMask(mask, [size(original_image, 1), size(original_image, 2)]);
    
    % Find the mask contours
    mask_contours = bwperim(binary_mask);
    
    % Display the original image
    imshow(original_image);
    hold on;
    
    % Overlay the contours on the image (red contour color)
    visboundaries(mask_contours, 'Color', 'r');
    
    hold off;
end

function C = weberContrast(I_sample, I_background)
%   C = weberContrast(I_sample, I_background)
%   - I_sample     (numeric): Sample intensity(s).
%   - I_background (numeric): Background intensity(s).
%   Returns C = (I_sample - I_background) ./ I_background.
%
%   Example:
%       c = weberContrast(150,100);  % c == 0.5

    C = (I_sample - I_background) ./ I_background;
end

function sortedStruct = sort_image_mask_struct(inputStruct)
%SORT_IMAGE_MASK_STRUCT Sorts image-mask pairs by numeric suffix in image filenames.
%   sortedStruct = sort_image_mask_struct(inputStruct)
%   - inputStruct (struct array): Fields .image_path, .mask_path.
%   Returns the struct array sorted on the numeric token extracted from .image_path.
%
%   Example:
%       s = sort_image_mask_struct(pairs);  % orders by number in "img_001.png"

    % Extract image paths
    imagePaths = {inputStruct.image_path};
    
    % Initialize an array to hold numerical values
    nums = zeros(size(imagePaths));

    % Loop through each image path to extract numerical values
    for i = 1:length(imagePaths)
        % Use regexp to find numeric part
        match = regexp(imagePaths{i}, '_(\d+)\.png$', 'tokens'); % Extract numeric part
        if ~isempty(match)
            nums(i) = str2double(match{1}{1}); % Convert to double
        else
            nums(i) = NaN; % Handle cases with no match
        end
    end

    % Sort structure based on the extracted numbers, ignoring NaNs
    [~, sortIdx] = sort(nums); % Sort indices
    sortedStruct = inputStruct(sortIdx);  % Reorder structure based on sorted indices
end

function displayImagesWithMasks(paths_pairs)
%DISPLAYIMAGESWITHMASKS Iteratively displays images with mask contours and progress.
%   displayImagesWithMasks(paths_pairs)
%   - paths_pairs (struct array): Fields .image_path, .mask_path.
%   Displays each image with overlaid contours and a progress title.
%
%   Example:
%       displayImagesWithMasks(pairs);

    for ind = 1 : length(paths_pairs)
        path_img = paths_pairs(ind).image_path;
        path_mask = paths_pairs(ind).mask_path;
        displayImageWithMaskContour(path_img, path_mask)
    
        progress = (ind / length(paths_pairs)) * 100;
        title(['Processing... ', num2str(progress, '%.1f'), '%']);
    
        pause(0.01)
    end
end

function metrics = calcMetricsFromImageAndMask(img_origin, img_mask, radius_dilat)
%CALCMETRICSFROMIMAGEANDMASK Calculates metrics for a single image and mask.
%   metrics = calcMetricsFromImageAndMask(img_origin, img_mask, radius_dilat)
%   - img_origin   (matrix): Original image.
%   - img_mask     (logical matrix): Binary object mask.
%   - radius_dilat (integer): Dilation radius for background.
%   Returns struct with fields:
%       .web_contrast
%       .area
%       .transmittance
%
%   Example:
%       I = imread("img.png");
%       M = processMask(imread("mask.png"), size(I));
%       m = calcMetricsFromImageAndMask(I, M, 10);

    mean_value_object = mean(img_origin(img_mask(:)));
    mask_background = bwmorph(img_mask, 'dilate', radius_dilat) & ~img_mask;
    mean_value_background = mean(img_origin(mask_background(:)));
    
    metrics = initializeStructureForMetrics(1);
    metrics.web_contrast = -weberContrast(mean_value_object, mean_value_background);
    metrics.area = sum(img_mask(:));
    metrics.transmittance = mean_value_object/mean_value_background;
end

function struct_metrics = initializeStructureForMetrics(num_of_elements)
%INITIALIZESTRUCTUREFORMETRICS Preallocates metric struct array.
%   struct_metrics = initializeStructureForMetrics(num_of_elements)
%   - num_of_elements (integer): Number of entries.
%   Returns struct array with empty fields .web_contrast, .area, .transmittance.
%
%   Example:
%       arr = initializeStructureForMetrics(5);

    struct_metrics(num_of_elements) = struct('web_contrast', [], 'area', [], 'transmittance', []);
end

function metrics_array = calcMetricsOfAllImages(paths_pairs, radius_dilate)
%CALCMETRICSOFALLIMAGES Computes metrics over all image–mask pairs.
%   metrics_array = calcMetricsOfAllImages(paths_pairs, radius_dilate)
%   - paths_pairs   (struct array): Fields .image_path, .mask_path.
%   - radius_dilate (integer): Dilation radius for background.
%   Returns an array of metric structs, one per pair.
%
%   Example:
%       metrics = calcMetricsOfAllImages(pairs, 15);

    metrics_array = initializeStructureForMetrics(length(paths_pairs));

    for ind = 1 : length(paths_pairs)
        img_origin = imread(paths_pairs(ind).image_path);
        img_mask = imread(paths_pairs(ind).mask_path);
        img_mask = processMask(img_mask, size(img_origin));
    
        metrics_array(ind) = calcMetricsFromImageAndMask(img_origin, img_mask, radius_dilate);
    
        disp("Progress: " + num2str(ind/length(paths_pairs) * 100) + "%")
    end
end

function displayMetrics(metrics_array, time_x)
%DISPLAYMETRICS Plots area, contrast, and transmittance over time.
%   displayMetrics(metrics_array, time_x)
%   - metrics_array (struct array): Fields .area, .web_contrast, .transmittance.
%   - time_x        (numeric vector): Time in days.
%   Generates a figure with three subplots for normalized area, Weber contrast, and transmittance (%).
%
%   Example:
%       times = [0;1;2];
%       displayMetrics(metrics, times);

    area_array = [metrics_array(:).area];
    web_contrast_array = [metrics_array(:).web_contrast];
    transmittance_array = [metrics_array(:).transmittance];

    % Processing data
    area_array = area_array / area_array(1);
    transmittance_array = transmittance_array * 100;
 
    time_x = time_x * 24; % Days to hours

    % Create a figure with a white background
    h = figure;
    set(h, 'Color', 'w');

    % Subplot for Area
    subplot(3, 1, 1);  % 3 rows, 1 column, 1st subplot
    plot(time_x, area_array, 'LineWidth', 2);
    title('Area plot');
    xlabel('Time (hours)');
    ylabel('Normalized Area');
    
    % Subplot for Weber Contrast
    subplot(3, 1, 2);  % 3 rows, 1 column, 2nd subplot
    plot(time_x, web_contrast_array, 'LineWidth', 2);
    title('Weber contrast plot');
    xlabel('Time (hours)');
    ylabel('Weber Contrast');
    
    % Subplot for Transmittance
    subplot(3, 1, 3);  % 3 rows, 1 column, 3rd subplot
    plot(time_x, transmittance_array, 'LineWidth', 2);
    title('Transmittance plot');
    xlabel('Time (hours)');
    ylabel('Transmittance (%)');
end


function time = getImageTime(filename)
%GETIMAGETIME Retrieves timestamp from image metadata.
%   time = getImageTime(filename)
%   - filename (string): Path to image file.
%   Returns date string from EXIF 'DateTimeOriginal' or 'FileModDate', or empty if unavailable.
%
%   Example:
%       t = getImageTime("photo.jpg");

    % Function returns the timestamp of the image if available in the EXIF metadata.
    % If the information is not available, it returns an empty array.
    
    % Retrieve the image metadata
    info = imfinfo(filename);
    
    % Initialize the variable 'time' as empty
    time = [];
    
    % Check if the timestamp information exists in the EXIF metadata
    if isfield(info, 'DigitalCamera') && isfield(info.DigitalCamera, 'DateTimeOriginal')
        % Retrieve the timestamp of when the image was taken
        time = info.DigitalCamera.DateTimeOriginal;
    elseif isfield(info, 'FileModDate')
        % Alternatively, use the file modification timestamp
        time = info.FileModDate;
    else
        fprintf('Timestamp metadata is not available for the file: %s\n', filename);
    end
end


% function numericTime = getImageTimeAsNumeric(filename)
%     % getImageTimeAsNumeric Extracts and converts image capture time to numeric format.
%     %
%     % Description:
%     %   This function retrieves the capture time from an image file's metadata 
%     %   (EXIF or other metadata fields), and converts it to a numeric format 
%     %   representing the number of days since January 0, 0000 in MATLAB's datenum format.
%     %   This numeric format allows for easy time-based calculations, such as 
%     %   determining time differences between images.
%     %
%     %   Note: The returned value is in days. To calculate the time difference in 
%     %   seconds between two time points, subtract one numeric time from another, 
%     %   then multiply by 24 * 3600 (the number of seconds in a day).
%     %
%     % Input:
%     %   filename - A string specifying the path and name of the image file.
%     %
%     % Output:
%     %   numericTime - A numeric value representing the capture time in days since 
%     %                 January 0, 0000. Returns an error if no time information is found 
%     %                 in the metadata.
%     %
%     % Example:
%     %   timeNum = getImageTimeAsNumeric('image1.jpg');
%     %
%     %   % To calculate the difference in seconds between two images:
%     %   time1 = getImageTimeAsNumeric('image1.jpg');
%     %   time2 = getImageTimeAsNumeric('image2.jpg');
%     %   timeDifferenceInSeconds = (time2 - time1) * 24 * 3600;
%     %
%     % Note:
%     %   This function requires that the `getImageTime` function is available to extract 
%     %   the time as a text string from the image metadata.
% 
%     timeStr = getImageTime(filename);
% 
%     if isempty(timeStr)
%         error('Failed to find time information in the metadata.');
%     end
%     try
%        dateTimeObj = datetime(timeStr, 'InputFormat', 'dd-MMM-yyyy HH:mm:ss');
%     catch
%        dateTimeObj = datetime(timeStr, 'InputFormat', 'dd-MMM-yyyy HH:mm:ss', 'Locale','pl-PL');
%     end
%     numericTime = datenum(dateTimeObj);
% end

function t = getImageTimeAsNumeric(filename)
%GETIMAGETIME Retrieves timestamp from image metadata.
%   time = getImageTime(filename)
%   - filename (string): Path to image file.
%   Returns date string from EXIF 'DateTimeOriginal' or 'FileModDate', or empty if unavailable.
%
%   Example:
%       t = getImageTime("photo.jpg");

    [~, name, ~] = fileparts(filename);

    try
        t = datenum(name, 'yyyy-mm-dd HH-MM-SS');
    catch
        error('Nieprawidłowy format nazwy pliku: %s', filename);
    end
end


function times_array = getTimeFromMetadatas(paths_images)
%GETIMAGETIMEASNUMERIC Extracts numeric time (datenum) from filename.
%   t = getImageTimeAsNumeric(filename)
%   - filename (string): File path with name in 'yyyy-mm-dd HH-MM-SS' format.
%   Parses the timestamp and returns MATLAB datenum; errors if format is invalid.
%
%   Example:
%       dn = getImageTimeAsNumeric("2025-04-01 12-30-00.jpg");

    n = numel(paths_images);
    times_array = zeros(n,1);
    for i = 1:n
        times_array(i) = getImageTimeAsNumeric(paths_images{i});
    end
end

function createAnimationOfObjectDetection(filename_save, paths_pairs, metrics_array, times_days_array)
%CREATEANIMATIONOFOBJECTDETECTION Generates video of detection and metric plots.
%   createAnimationOfObjectDetection(filename_save, paths_pairs, metrics_array, times_days_array)
%   - filename_save    (string): Output video filename.
%   - paths_pairs      (struct array): .image_path, .mask_path.
%   - metrics_array    (struct array): Metrics per frame.
%   - times_days_array (numeric vector): Time in days since baseline.
%   Saves an MPEG-4 video showing images with contours and evolving metric plots.
%
%   Example:
%       createAnimationOfObjectDetection("out.mp4", pairs, metrics, times);

    % Create a video writer
    videoWriter = VideoWriter(filename_save, 'MPEG-4');
    videoWriter.Quality = 100;
    open(videoWriter);

    % Preprocess data
    area_array = [metrics_array(:).area] / max([metrics_array(:).area]) * 100; % Normalize area
    web_contrast_array = [metrics_array(:).web_contrast];
    transmittance_array = [metrics_array(:).transmittance] * 100; % Convert to percentage
    times_hours = times_days_array * 24; % Convert days to hours

    % Define y-limits outside the loop for performance
    area_limits = [min(area_array), max(area_array)];
    contrast_limits = [min(web_contrast_array), max(web_contrast_array)];
    transmittance_limits = [min(transmittance_array), max(transmittance_array)];

    % Create the figure and layout
    h_fig = figure('Units', 'Normalized', 'Position', [0 0 1 1]); 
    set(h_fig, 'Color', 'w');

    % Configure tile layout
    tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    for ind = 1:length(paths_pairs)
        path_img = paths_pairs(ind).image_path;
        path_mask = paths_pairs(ind).mask_path;

        % Convert time to hh:mm:ss format
        current_time = seconds(times_hours(ind) * 3600);
        time_string = string(duration(current_time, 'Format', 'hh:mm:ss'));

        % Create a layout with two columns
        subplot('Position', [0.05 0.1 0.5 0.8]); % Image on the left (large)
        displayImageWithMaskContour(path_img, path_mask);
        title(['Image with Mask Contours', ...
            "Elapsed Time [hh:mm:ss]: " + time_string]);
        
        % Plots on the right
        % First plot (area)
        subplot('Position', [0.6 0.54 0.35 0.3]);
        plot(times_hours(1:ind), area_array(1:ind), 'LineWidth', 2, 'Color', 'b');
        title('Normalized Area Over Time [%]');
        xlabel('Time [hours]');
        ylabel('Normalized Area');
        xlim([times_hours(1), times_hours(end)]);
        ylim(area_limits);
        grid on;
        
        % Second plot (transmittance)
        subplot('Position', [0.6 0.15 0.35 0.3]);
        plot(times_hours(1:ind), transmittance_array(1:ind), 'LineWidth', 2, 'Color', '#2e8b57');
        title('Transparency Over Time');
        xlabel('Time [hours]');
        ylabel('Transparency [%]');
        xlim([times_hours(1), times_hours(end)]);
        ylim(transmittance_limits);
        grid on;
    
        % Save the frame to the video
        frameData = getframe(gcf);
        writeVideo(videoWriter, frameData);
    
        % Short pause for visualization (optional)
        pause(0.01);
    end

    % Close the video writer and figure
    close(videoWriter);
    close(h_fig);

    disp(['Animation completed and saved as ', filename_save]);
end

function baseline = computeBaselineFromMetadata(input)
%COMPUTEBASELINEFROMMETADATA Determines earliest timestamp among images.
%   baseline = computeBaselineFromMetadata(input)
%   - input (string folder path, or cell/string array of file paths)
%   Scans images, extracts numeric timestamps, and returns the minimum as baseline.
%
%   Example:
%       b = computeBaselineFromMetadata("data/imgs");

    if ischar(input) && isfolder(input)
        exts = {'*.png','*.jpg','*.jpeg','*.tiff','*.bmp','*.gif'};
        files = [];
        for k = 1:numel(exts)
            files = [files; dir(fullfile(input, exts{k}))]; 
        end
        paths = fullfile(input, {files.name});
    elseif isstring(input) || iscell(input)
        paths = cellstr(input);
    else
        error('computeBaselineFromMetadata: nieprawidłowy typ wejścia.');
    end

    n = numel(paths);
    times = zeros(n,1);
    for i = 1:n
        times(i) = getImageTimeAsNumeric(paths{i});
    end
    baseline = min(times);
end


function score = countMatchingPathEnd(path1, path2)
%COUNTMATCHINGPATHEND Counts matching trailing segments of two paths.
%   score = countMatchingPathEnd(path1, path2)
%   - path1, path2 (string): Paths to compare.
%   Returns count of identical folder/file names starting from the end.
%
%   Example:
%       s = countMatchingPathEnd("a/b/c","x/b/c");  % s == 2

    s1 = fliplr(strsplit(normalizePath(path1), filesep));
    s2 = fliplr(strsplit(normalizePath(path2), filesep));
    score = 0;
    for i = 1:min(length(s1), length(s2))
        if strcmp(s1{i}, s2{i})
            score = score + 1;
        else
            break;
        end
    end
end

function folders = findDeepestDirs(root_dir, exclude_name)
%FINDDEEPESTDIRS Finds deepest subdirectories excluding a given name.
%   folders = findDeepestDirs(root_dir, exclude_name)
%   - root_dir     (string): Starting directory.
%   - exclude_name (string): Folder name to skip.
%   Returns a cell array of full paths to the deepest-level directories.
%
%   Example:
%       dirs = findDeepestDirs("project",".git");

    folders = {};
    queue = {normalizePath(root_dir)};
    while ~isempty(queue)
        current = queue{1};
        queue(1) = [];
        subdirs = dir(current);
        subdirs = subdirs([subdirs.isdir] & ~ismember({subdirs.name}, {'.', '..'}));

        subdirs = subdirs(~strcmp({subdirs.name}, exclude_name));

        if isempty(subdirs)
            folders{end+1} = current; %#ok<AGROW>
        else
            for i = 1:length(subdirs)
                queue{end+1} = fullfile(current, subdirs(i).name); %#ok<AGROW>
            end
        end
    end
end

function p = normalizePath(p)
%NORMALIZEPATH Normalizes file separators to the OS-specific filesep.
%   p = normalizePath(p)
%   - p (string): Path to normalize.
%   Returns the path with consistent '/' or '\' separators.
%
%   Example:
%       np = normalizePath("folder/sub\file.txt");

    p = strrep(p, '/', filesep);
    p = strrep(p, '\', filesep);
end
