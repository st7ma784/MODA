function tests = test_moda_algorithms
% test_moda_algorithms  MATLAB Test Framework suite for MODA core algorithms.
%
% Run from the MODA root directory:
%   results = runtests('tests/test_moda_algorithms.m');
%   table(results)
%
% Or via CI (matlab-actions/run-tests points source-folder at repo root):
%   runtests('tests/test_moda_algorithms', 'OutputDetail', 'Detailed')

    tests = functiontests(localfunctions);
end

% =========================================================================
% Fixture — run once before all tests
% =========================================================================
function setupOnce(tc)
    % Add every MODA subfolder to the path.
    root = fileparts(fileparts(mfilename('fullpath')));
    addpath(genpath(root));
    tc.TestData.root = root;

    % Short synthetic signals — fast enough for CI.
    fs  = 10;          % Hz  (matches example_sigs)
    dur = 20;          % seconds
    t   = (0 : 1/fs : dur - 1/fs);

    tc.TestData.fs   = fs;
    tc.TestData.t    = t;
    % 1 Hz oscillation
    tc.TestData.sig1 = sin(2*pi*1.0*t);
    % 2 Hz oscillation weakly coupled to sig1
    tc.TestData.sig2 = sin(2*pi*2.0*t + 0.3*sin(2*pi*1.0*t));
end

% =========================================================================
% 1. Required files exist
% =========================================================================
function testRequiredFilesExist(tc)
    root = tc.TestData.root;
    required = {
        'allguis/guis/tfa/Functions/wt.m'
        'allguis/guis/tfa/Functions/wft.m'
        'allguis/guis/tfa/Functions/wtwrapper.m'
        'allguis/guis/filtering/Functions/loop_butter.m'
        'allguis/guis/filtering/Functions/ecurve.m'
        'allguis/guis/filtering/Functions/rectfr.m'
        'allguis/codes/reading/csv_to_mvar.m'
        'allguis/guis/coherence/Functions/wphcoh.m'
        'allguis/guis/coherence/Functions/tlphcoh.m'
        'allguis/guis/tfa/TimeFrequencyAnalysis.m'
        'allguis/guis/filtering/Filtering.m'
        'allguis/guis/bayesian/Bayesian.m'
        'allguis/guis/coherence/CoherenceMulti.m'
        'allguis/guis/bispectrum/Bispectrum.m'
        'allguis/codes/Universal/MODAread.m'
        'allguis/codes/Universal/MODATFAcalc.m'
        'example_sigs/1signal_10Hz.mat'
        'example_sigs/2signals_10Hz.mat'
    };
    for i = 1:numel(required)
        tc.verifyTrue(isfile(fullfile(root, required{i})), ...
            sprintf('Missing required file: %s', required{i}));
    end
end

% =========================================================================
% 2. csv_to_mvar — frequency-band string parser
% =========================================================================
function testCsvToMvarPair(tc)
    result = csv_to_mvar('0.5,1.5');
    tc.verifyEqual(result, [0.5 1.5], 'AbsTol', 1e-12, ...
        'csv_to_mvar must parse "0.5,1.5" as [0.5 1.5]');
end

function testCsvToMvarTriple(tc)
    result = csv_to_mvar('1,2,3');
    tc.verifyEqual(result, [1 2 3], 'AbsTol', 1e-12);
end

function testCsvToMvarInteger(tc)
    result = csv_to_mvar('2');
    tc.verifyEqual(result, 2, 'AbsTol', 1e-12);
end

% =========================================================================
% 3. Wavelet transform (wt.m — Iatsenko implementation)
% =========================================================================
function testWtOutputDimensions(tc)
    sig = tc.TestData.sig1;
    fs  = tc.TestData.fs;

    [WT, freqarr, wopt] = wt(sig, fs, ...
        'fmin', 0.5, 'fmax', 4, 'CutEdges', 'off', 'Preprocess', 'off');

    tc.verifyEqual(size(WT, 2), length(sig), ...
        'WT time axis must equal signal length');
    tc.verifyTrue(all(freqarr > 0), ...
        'All frequency bins must be positive');
    tc.verifyFalse(all(isnan(WT(:))), ...
        'WT must not be entirely NaN');
    tc.verifyNotEmpty(wopt, 'wopt struct must be returned');
end

function testWtAmplitudeAtPeak(tc)
    % A pure 1 Hz sinusoid should produce maximum WT amplitude near 1 Hz.
    sig = tc.TestData.sig1;
    fs  = tc.TestData.fs;

    [WT, freqarr, ~] = wt(sig, fs, ...
        'fmin', 0.5, 'fmax', 4, 'CutEdges', 'off', 'Preprocess', 'off');

    avg_amp   = nanmean(abs(WT), 2);   % time-average amplitude per freq
    [~, imax] = max(avg_amp);
    peak_freq = freqarr(imax);

    tc.verifyLessThanOrEqual(abs(peak_freq - 1.0), 0.3, ...
        sprintf('WT peak should be near 1 Hz; got %.3f Hz', peak_freq));
end

% =========================================================================
% 4. Windowed Fourier transform (wft.m)
% =========================================================================
function testWftOutputDimensions(tc)
    sig = tc.TestData.sig1;
    fs  = tc.TestData.fs;

    [WFT, freqarr, wopt] = wft(sig, fs, ...
        'fmin', 0.5, 'fmax', 4, 'CutEdges', 'off', 'Preprocess', 'off');

    tc.verifyEqual(size(WFT, 2), length(sig), ...
        'WFT time axis must equal signal length');
    tc.verifyTrue(all(freqarr > 0));
    tc.verifyNotEmpty(wopt);
end

% =========================================================================
% 5. wtwrapper — both WT and WFT modes
% =========================================================================
function testWtWrapperWaveletMode(tc)
    sig = tc.TestData.sig1;
    fs  = tc.TestData.fs;

    [WT, freqarr, wopt] = wtwrapper(sig, fs, NaN, 0.5, 4, ...
        1, 'Lognorm', 'off', 'off');

    tc.verifyEqual(size(WT, 2), length(sig));
    tc.verifyTrue(all(freqarr > 0));
    tc.verifyNotEmpty(wopt);
end

function testWtWrapperWFTMode(tc)
    sig = tc.TestData.sig1;
    fs  = tc.TestData.fs;

    [WFT, freqarr, wopt] = wtwrapper(sig, fs, NaN, 0.5, 4, ...
        2, 'Gauss', 'off', 'off');

    tc.verifyEqual(size(WFT, 2), length(sig));
    tc.verifyTrue(all(freqarr > 0));
    tc.verifyNotEmpty(wopt);
end

function testWtWrapperAutoFreqRange(tc)
    % NaN fmin/fmax should not error — wt handles auto range.
    sig = tc.TestData.sig1;
    fs  = tc.TestData.fs;

    [WT, freqarr, ~] = wtwrapper(sig, fs, NaN, NaN, NaN, ...
        1, 'Lognorm', 'off', 'off');

    tc.verifyEqual(size(WT, 2), length(sig));
    tc.verifyTrue(all(freqarr > 0));
end

% =========================================================================
% 6. loop_butter — iterative Butterworth band filter
% =========================================================================
function testLoopButterLength(tc)
    sig = tc.TestData.sig1;
    fs  = tc.TestData.fs;

    [filtered, order] = loop_butter(sig, [0.5, 2.0], fs);

    tc.verifyEqual(length(filtered), length(sig), ...
        'Filtered signal must be same length as input');
    tc.verifyGreaterThanOrEqual(order, 1, ...
        'Filter order must be at least 1');
end

function testLoopButterRemovesOutOfBand(tc)
    % High-frequency component (4 Hz) should be attenuated by a 0.5-2 Hz bandpass.
    fs  = tc.TestData.fs;
    t   = tc.TestData.t;
    sig = sin(2*pi*4*t);   % 4 Hz — outside the 0.5-2 Hz band

    filtered = loop_butter(sig, [0.5, 2.0], fs);

    tc.verifyLessThan(max(abs(filtered)), max(abs(sig)), ...
        '4 Hz signal should be attenuated by a 0.5-2 Hz Butterworth filter');
end

% =========================================================================
% 7. Wavelet phase coherence (wphcoh.m)
% =========================================================================
function testWphcohRange(tc)
    sig1 = tc.TestData.sig1;
    sig2 = tc.TestData.sig2;
    fs   = tc.TestData.fs;

    [WT1, freqarr, ~] = wt(sig1, fs, 'fmin', 0.5, 'fmax', 4, ...
        'CutEdges', 'off', 'Preprocess', 'off');
    [WT2, ~, ~]       = wt(sig2, fs, 'fmin', 0.5, 'fmax', 4, ...
        'CutEdges', 'off', 'Preprocess', 'off');

    [phcoh, ~] = wphcoh(WT1, WT2);

    tc.verifyEqual(length(phcoh), length(freqarr), ...
        'Coherence vector length must match frequency array');
    tc.verifyTrue(all(phcoh >= 0 & phcoh <= 1), ...
        'Wavelet phase coherence must lie in [0, 1]');
end

function testWphcohSelfCoherence(tc)
    % A signal with itself should have coherence = 1 everywhere.
    sig = tc.TestData.sig1;
    fs  = tc.TestData.fs;

    [WT, ~, ~] = wt(sig, fs, 'fmin', 0.5, 'fmax', 4, ...
        'CutEdges', 'off', 'Preprocess', 'off');

    [phcoh, ~] = wphcoh(WT, WT);

    tc.verifyTrue(all(abs(phcoh - 1) < 1e-9), ...
        'Self-coherence must equal 1 at every frequency');
end

% =========================================================================
% 8. Integration — example_sigs sample data
% =========================================================================
function testIntegration1SignalWT(tc)
    d   = load(fullfile(tc.TestData.root, 'example_sigs', '1signal_10Hz.mat'));
    sig = d.y;
    fs  = 10;

    [WT, freqarr, ~] = wt(sig, fs, 'fmin', 0.1, 'fmax', 4, ...
        'CutEdges', 'off', 'Preprocess', 'off');

    tc.verifyEqual(size(WT, 2), length(sig));
    tc.verifyTrue(all(freqarr > 0));
    tc.verifyFalse(all(isnan(WT(:))));
end

function testIntegration2SignalsCoherence(tc)
    d    = load(fullfile(tc.TestData.root, 'example_sigs', '2signals_10Hz.mat'));
    sigs = d.a;
    fs   = 10;

    [WT1, freqarr, ~] = wt(sigs(1,:), fs, 'fmin', 0.1, 'fmax', 4, ...
        'CutEdges', 'off', 'Preprocess', 'off');
    [WT2, ~, ~]       = wt(sigs(2,:), fs, 'fmin', 0.1, 'fmax', 4, ...
        'CutEdges', 'off', 'Preprocess', 'off');

    [phcoh, ~] = wphcoh(WT1, WT2);

    tc.verifyEqual(length(phcoh), length(freqarr));
    tc.verifyTrue(all(phcoh >= 0 & phcoh <= 1));
end

function testIntegration6SignalsButter(tc)
    d    = load(fullfile(tc.TestData.root, 'example_sigs', '6signals_10Hz.mat'));
    sigs = d.IEEEex_10Hz;
    fs   = 10;

    for k = 1:size(sigs, 1)
        [filtered, order] = loop_butter(sigs(k,:), [0.5, 2.0], fs);
        tc.verifyEqual(length(filtered), size(sigs, 2), ...
            sprintf('Signal %d: filtered length mismatch', k));
        tc.verifyGreaterThanOrEqual(order, 1);
    end
end

% =========================================================================
% 9. App Designer migration sanity check
% =========================================================================
function testGuisMigratedToAppDesigner(tc)
    root = tc.TestData.root;
    guis = {
        'allguis/guis/tfa/TimeFrequencyAnalysis.m'
        'allguis/guis/filtering/Filtering.m'
        'allguis/guis/bayesian/Bayesian.m'
        'allguis/guis/coherence/CoherenceMulti.m'
        'allguis/guis/bispectrum/Bispectrum.m'
    };
    for i = 1:numel(guis)
        code = fileread(fullfile(root, guis{i}));
        tc.verifyTrue(contains(code, 'classdef'), ...
            sprintf('%s: must use App Designer classdef', guis{i}));
        tc.verifyTrue(contains(code, 'matlab.apps.AppBase'), ...
            sprintf('%s: must inherit from matlab.apps.AppBase', guis{i}));
        tc.verifyFalse(contains(code, 'gui_mainfcn'), ...
            sprintf('%s: must not contain deprecated gui_mainfcn', guis{i}));
    end
end

% =========================================================================
% 10. Deprecated function check — csvread/csvwrite must be gone
% =========================================================================
function testNoCsvreadCsvwrite(tc)
    root  = tc.TestData.root;
    files = {
        'allguis/codes/reading/read_from_csv.m'
        'allguis/codes/reading/read_from_mat.m'
        'allguis/codes/Universal/MODAread.m'
    };
    for i = 1:numel(files)
        p = fullfile(root, files{i});
        if ~isfile(p); continue; end
        code = fileread(p);
        tc.verifyFalse(contains(code, 'csvread('), ...
            sprintf('%s still uses deprecated csvread()', files{i}));
        tc.verifyFalse(contains(code, 'csvwrite('), ...
            sprintf('%s still uses deprecated csvwrite()', files{i}));
    end
end

% =========================================================================
% 11. UI-helper coverage — helper modules contain agnostic helpers
% =========================================================================
function testUiHelpersPresent(tc)
    root  = tc.TestData.root;
    files = {
        'allguis/codes/Universal/MODAread.m'
        'allguis/codes/Universal/MODATFAcalc.m'
        'allguis/guis/filtering/Functions/MODAridge_filter.m'
        'allguis/guis/bayesian/Functions/MODAbayes_loadfilt.m'
        'allguis/guis/bayesian/Functions/MODAbayes_intdelete.m'
    };
    for i = 1:numel(files)
        p = fullfile(root, files{i});
        if ~isfile(p); continue; end
        code = fileread(p);
        % Each migrated helper must define a setStr or setEnable local helper.
        tc.verifyTrue( ...
            contains(code, 'function setStr') || contains(code, 'function setEnable'), ...
            sprintf('%s must contain UI-agnostic helper functions', files{i}));
    end
end
