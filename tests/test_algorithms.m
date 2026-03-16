% Example Test File for MODA
% Location: /home/user/MODA/tests/test_algorithms.m
% Purpose: Validate core algorithms work correctly in Docker container
%
% Run with: matlab -batch "runtests tests/test_algorithms.m; exit(0);"

function tests = test_algorithms
    % Create test suite from all local functions
    tests = functiontests(localfunctions);
end

% =========================================================================
% Test 1: Check MATLAB Version Compatibility
% =========================================================================
function testMATLABVersion(testCase)
    % Verify R2023a or later
    v = ver('MATLAB');
    
    % Extract version year (e.g., "9.14.0" from R2023a is version 23)
    verstr = sscanf(v.Version, '%d.%d');
    year = 2020 + verstr(1) - 13;  % Version 13.0 is R2023a
    
    testCase.verifyGreaterThanOrEqual(year, 2023, ...
        'MATLAB version must be R2023a or later for App Designer');
    
    disp(sprintf('✓ MATLAB %s detected', v.Release));
end

% =========================================================================
% Test 2: Check Toolbox Availability
% =========================================================================
function testToolboxes(testCase)
    % Verify required toolboxes are installed
    toolboxes_required = {
        'Signal Processing Toolbox'
        'Wavelet Toolbox'
        'Statistics and Machine Learning Toolbox'
    };
    
    v = ver;
    installed = {v.Name};
    
    for i = 1:length(toolboxes_required)
        tbx = toolboxes_required{i};
        found = any(contains(installed, tbx));
        testCase.verifyTrue(found, sprintf('%s not found', tbx));
        disp(sprintf('✓ %s available', tbx));
    end
end

% =========================================================================
% Test 3: Data I/O - CSV Reading (updated function)
% =========================================================================
function testCSVReading(testCase)
    % Test readmatrix() works (replacement for csvread)
    
    % Create test data
    testData = [1, 2, 3; 4, 5, 6; 7, 8, 9];
    testFile = tempname;
    testFileCSV = [testFile '.csv'];
    
    % Write using writematrix (replacement for csvwrite)
    writematrix(testData, testFileCSV);
    
    % Verify file was created
    testCase.verifyTrue(isfile(testFileCSV), 'CSV file not created');
    
    % Read back using readmatrix
    loaded = readmatrix(testFileCSV);
    
    % Verify data integrity
    testCase.verifyEqual(testData, loaded, ...
        'CSV data not preserved in read/write cycle');
    
    % Cleanup
    delete(testFileCSV);
    
    disp('✓ CSV read/write working correctly');
end

% =========================================================================
% Test 4: Data I/O - MAT File Reading
% =========================================================================
function testMATFileReading(testCase)
    % Test load() and save() still work
    
    testData = sin(2*pi*(1:100)/10);
    testFile = tempname;
    testFileMAT = [testFile '.mat'];
    
    % Save as MAT
    save(testFileMAT, 'testData');
    testCase.verifyTrue(isfile(testFileMAT), 'MAT file not created');
    
    % Load back
    loaded = load(testFileMAT);
    
    % Verify
    testCase.verifyEqual(testData, loaded.testData, ...
        'MAT data not preserved');
    
    % Cleanup
    delete(testFileMAT);
    
    disp('✓ MAT file read/write working correctly');
end

% =========================================================================
% Test 5: Wavelet Transform Algorithm
% =========================================================================
function testWaveletTransform(testCase)
    % Test the core wt.m function
    
    % Create test signal: 1Hz + 2Hz combination
    fs = 100;  % Sample rate
    t = (0:0.01:5)';
    signal = sin(2*pi*1*t) + 0.5*sin(2*pi*2*t);
    
    % Call WT
    try
        [WT, freq] = wt(signal, fs, 'Display', 'off');
    catch ME
        testCase.verificationFailed(sprintf('WT failed: %s', ME.message));
        return;
    end
    
    % Verify output dimensions
    testCase.verifyEqual(size(WT, 2), length(signal), ...
        'WT output length mismatch');
    testCase.verifyTrue(all(freq > 0), ...
        'Frequencies must be positive');
    
    % Verify no NaN or Inf values
    testCase.verifyFalse(any(isnan(WT(:))), 'WT contains NaN values');
    testCase.verifyFalse(any(isinf(WT(:))), 'WT contains Inf values');
    
    disp(sprintf('✓ Wavelet Transform working: %d freq bins, %d time points', ...
        size(WT,1), size(WT,2)));
end

% =========================================================================
% Test 6: String Function Modernization (strfind → contains)
% =========================================================================
function testStringFunctions(testCase)
    % Verify modernized string functions work
    
    % Test 1: contains() function
    str = 'hello world';
    
    % Old: ~isempty(strfind(str, 'world'))
    % New: contains(str, 'world')
    result1 = contains(str, 'world');
    testCase.verifyTrue(result1, 'contains() function not working');
    
    % Test 2: Case-insensitive
    result2 = contains(str, 'HELLO', 'IgnoreCase', true);
    testCase.verifyTrue(result2, 'Case-insensitive contains() failed');
    
    % Test 3: Multiple patterns
    result3 = contains(str, {'world', 'notfound', 'hello'});
    testCase.verifyEqual(sum(result3), 2, ...
        'Multiple pattern matching failed');
    
    disp('✓ String functions modernized and working');
end

% =========================================================================
% Test 7: File Operations
% =========================================================================
function testFileOperations(testCase)
    % Verify file I/O functions work correctly
    
    % Test existence of core files
    files_required = {
        'MODA.m'
        'allguis/codes/reading/read_from_csv.m'
        'allguis/codes/reading/read_from_mat.m'
        'allguis/guis/tfa/TimeFrequencyAnalysis.m'
        'allguis/guis/filtering/Functions/ecurve.m'
        'allguis/guis/tfa/Functions/wt.m'
        'allguis/guis/tfa/Functions/wft.m'
    };
    
    for i = 1:length(files_required)
        fpath = files_required{i};
        testCase.verifyTrue(isfile(fpath), ...
            sprintf('Required file missing: %s', fpath));
    end
    
    disp('✓ All required files present');
end

% =========================================================================
% Test 8: MODA App Structure (New App Designer version)
% =========================================================================
function testMODAAppStructure(testCase)
    % Verify MODA.m is properly structured as App Designer class
    
    % Read MODA.m to check for proper structure
    code = fileread('MODA.m');
    
    % Check for classdef (App Designer)
    testCase.verifyTrue(contains(code, 'classdef MODAApp'), ...
        'MODA not converted to App Designer classdef');
    
    % Check for key App Designer methods
    testCase.verifyTrue(contains(code, 'createComponents'), ...
        'MODAApp missing createComponents method');
    
    testCase.verifyTrue(contains(code, 'function startupFcn'), ...
        'MODAApp missing startupFcn method');
    
    % Verify no old GUIDE references
    testCase.verifyFalse(contains(code, 'gui_mainfcn'), ...
        'MODA still contains old GUIDE references (gui_mainfcn)');
    testCase.verifyFalse(contains(code, 'guidata('), ...
        'MODA still contains old GUIDE references (guidata)');
    testCase.verifyFalse(contains(code, 'guihandles('), ...
        'MODA still contains old GUIDE references (guihandles)');
    
    disp('✓ MODA.m properly converted to App Designer');
end

% =========================================================================
% Test 9: Verify Deprecated Functions Replaced
% =========================================================================
function testDeprecatedFunctionsReplaced(testCase)
    % Verify critical deprecated functions are replaced
    
    % Files to check
    files_to_check = {
        'allguis/codes/reading/read_from_csv.m'
        'allguis/guis/tfa/TimeFrequencyAnalysis.m'
        'allguis/guis/filtering/Functions/ecurve.m'
    };
    
    for i = 1:length(files_to_check)
        fpath = files_to_check{i};
        if isfile(fpath)
            code = fileread(fpath);
            
            % Check for deprecated functions
            has_csvread = contains(code, 'csvread(');
            has_csvwrite = contains(code, 'csvwrite(');
            
            testCase.verifyFalse(has_csvread, ...
                sprintf('%s still uses deprecated csvread', fpath));
            testCase.verifyFalse(has_csvwrite, ...
                sprintf('%s still uses deprecated csvwrite', fpath));
            
            disp(sprintf('✓ %s: deprecated functions replaced', fpath));
        end
    end
end

% =========================================================================
% Test 10: Path and Loading
% =========================================================================
function testPathAndLoading(testCase)
    % Verify MODA modules can be found and loaded
    
    addpath(genpath('.'));
    
    % Check if MODA can be found in path
    w = which('MODA');
    testCase.verifyTrue(~isempty(w), 'MODA.m cannot be found in path');
    
    % Check auxiliary modules (these may still be GUIDE-based for now)
    modules = {
        'TimeFrequencyAnalysis'
        'read_from_csv'
        'read_from_mat'
    };
    
    for i = 1:length(modules)
        mod = modules{i};
        w = which(mod);
        testCase.verifyTrue(~isempty(w), ...
            sprintf('%s cannot be found in path', mod));
    end
    
    disp('✓ All modules can be found in MATLAB path');
end
