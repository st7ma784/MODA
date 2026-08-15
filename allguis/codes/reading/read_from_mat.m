function M = read_from_mat
% Read MAT file and create MATLAB variable
%
% This function opens a file dialog to select a MAT file,
% then loads and returns the data. If the MAT file contains
% a single variable, returns that variable. If multiple variables,
% returns the struct containing all variables.
%
% Output:
%   M - Variable(s) from MAT file (numeric array, table, or struct)
%
% Modernized from GUIDE version (March 2026):
%   - Added input validation and user cancellation handling
%   - Added semicolon after fullfile statement
%   - Added error handling with try-catch
%   - Improved documentation
%   - Compatible with MATLAB R2023a through R2026a

[filename, pathname, filterindex] = uigetfile('*.mat', 'Select MAT file');

% User cancelled dialog
if isequal(filename, 0)
    M = [];
    return;
end

filepath = fullfile(pathname, filename);

try
    % load() function compatible with all modern MATLAB versions
    % Returns struct with variable names as fields
    M = load(filepath);
    
    % If only one variable, extract it directly
    vars = fieldnames(M);
    if length(vars) == 1
        M = M.(vars{1});
    end
catch ME
    % Graceful error handling
    warning('Error reading %s: %s', filename, ME.message);
    M = [];
end
