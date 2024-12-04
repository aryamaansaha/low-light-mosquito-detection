% This is a MATLAB file which can be executed in the MATLAB command window
% by running the command "illuminate"


% Directory containing input images
input_dir = 'C:\Users\aryam\MSCS\Fall24\DLCV\Project\mainbase\datasets\aug_dataset\images\train\';

% Directory to save processed enhanced images
output_dir = 'C:\Users\aryam\MSCS\Fall24\DLCV\Project\mainbase\datasets\aug_il_dataset\images\train\';

% Get list of files in input directory
file_list = dir(fullfile(input_dir, '*.jpeg'));

% Loop through each file
for i = 1:length(file_list)
    % Read input image
    filename = fullfile(input_dir, file_list(i).name);
    img = imread(filename);
    
    % Process the image
    processed_img = imlocalbrighten(img);           % Ensure that the Image Processing Toolbox is installed
    
    % Save processed image to output directory
    [~, name, ext] = fileparts(filename);
    output_filename = fullfile(output_dir, [name ext]);
    imwrite(processed_img, output_filename);
    
    % Display progress
    fprintf('Processed and saved %f percent of %d files\n', i*100/length(file_list), length(file_list));
end