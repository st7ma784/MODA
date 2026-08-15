function M = read_from_csv
% Read CSV file and create MATLAB variable
%
% This function opens a file dialog to select a CSV file,
% then reads and returns the data as a numeric array.
%
% Output:
%   M - Numeric array containing CSV data
%
% Modernized from GUIDE version (March 2026):
%   - Replaced csvread() with readmatrix() [csvread removed in R2024a]
%   - Added input validation and error handling
%   - Added semicolon after fullfile statement
%   - Compatible with MATLAB R2023a through R2026a

[filename, pathname, filterindex] = uigetfile('*.csv', 'Select CSV file');

% User cancelled dialog
if isequal(filename, 0)
    M = [];
    return;
end

filepath = fullfile(pathname, filename);

try
    % Use modern readmatrix function (available R2019a+)
    M = readmatrix(filepath);
catch ME
    % Graceful error handling
    warning('Error reading %s: %s', filename, ME.message);
    M = [];
end
