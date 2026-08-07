% gen_moda_reference.m
% Generate MODA (MATLAB) reference outputs so that
% test_numeric_equivalence.py::test_matches_moda_reference becomes a *real*
% MODA-vs-FastMODA numerical diff instead of being skipped.
%
% Run from the repo root in MATLAB:
%     addpath(genpath('allguis'));
%     run('tests/parity/gen_moda_reference.m');
%
% Writes tests/parity/reference/moda_<algo>.mat, each containing:
%     signal   - the input time series
%     fs       - sampling frequency (Hz)
%     algorithm- string tag ('wt' | 'wft')
%     result   - |transform| magnitude matrix [freq x time]

outdir = fullfile(fileparts(mfilename('fullpath')), 'reference');
if ~exist(outdir, 'dir'); mkdir(outdir); end

fs = 40;                       % melanoma LDF protocol
N  = 4096;
t  = (0:N-1) / fs;

cases = struct( ...
    'wt',  @(x) abs(wt(x,  fs, 'fmin', 1, 'fmax', 15)), ...
    'wft', @(x) abs(wft(x, fs)));

f0 = 8;                        % clean tone with a known peak
signal = sin(2*pi*f0*t);

fn = fieldnames(cases);
for i = 1:numel(fn)
    algo = fn{i};
    try
        result = cases.(algo)(signal); %#ok<NASGU>
        save(fullfile(outdir, ['moda_' algo '.mat']), ...
             'signal', 'fs', 'algorithm', 'result');
        algorithm = algo; %#ok<NASGU>
        save(fullfile(outdir, ['moda_' algo '.mat']), ...
             'signal', 'fs', 'algorithm', 'result');
        fprintf('wrote moda_%s.mat\n', algo);
    catch err
        fprintf('skip %s: %s\n', algo, err.message);
    end
end
