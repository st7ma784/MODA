function buildToolbox()
% BUILDTOOLBOX  Package MODA into a distributable MATLAB toolbox (.mltbx).
%
% Run from anywhere with the repo on the MATLAB path, or just:
%   cd(fileparts(fileparts(mfilename('fullpath'))))
%   buildToolbox
%
% Produces MODA.mltbx at the repo root. End users install it by
% double-clicking the file in MATLAB, via the Add-On Manager, or
% programmatically with matlab.addons.toolbox.installToolbox('MODA.mltbx').
% Once installed, MATLAB adds MODA's folders to the path automatically —
% no manual addpath/genpath step, unlike running this repo directly.
%
% This only bundles the MATLAB desktop app (MODA.m, MODAApp.m, allguis/,
% example_sigs/) — NOT FastMODA (a separate Python/Flask application), the
% docs website, Docker/deployment scripts, or anything else in the repo
% that isn't part of the MATLAB toolbox itself.

repoRoot = fileparts(fileparts(mfilename('fullpath'))); % .../MODA (this file lives in .../MODA/scripts)

% A fixed identifier so re-running this script re-packages the SAME
% toolbox (upgrade in place) rather than registering a new one each time.
% This is MODA's own toolbox identifier — do not regenerate it casually,
% since existing installs are matched to it.
identifier = '7d1f9c2a-7d34-4b7a-9b52-2f6a2c8e6e91';

opts = matlab.addons.toolbox.ToolboxOptions(repoRoot, identifier);
opts.ToolboxName    = 'MODA';
opts.ToolboxVersion = '1.0.0';
opts.Summary = 'Multiscale Oscillatory Dynamics Analysis: time-frequency, coherence, ridge/filtering, bispectrum, and dynamical Bayesian inference for non-autonomous dynamical systems.';
opts.Description = [ ...
    'MODA (Multiscale Oscillatory Dynamics Analysis) analyses real-life ' ...
    'time-series assumed to be the output of some a priori unknown ' ...
    'non-autonomous dynamical system. It provides five analysis modules ' ...
    'in a single desktop app: Time-Frequency Analysis, Wavelet Phase ' ...
    'Coherence, Ridge Extraction & Filtering, Wavelet Bispectrum, and ' ...
    'Dynamical Bayesian Inference. Developed by the Nonlinear & ' ...
    'Biomedical Physics group at Lancaster University and the Nonlinear ' ...
    'Dynamics and Synergetics Group, University of Ljubljana. ' ...
    'Run MODA to launch the app after installing.'];
opts.AuthorName    = 'Nonlinear & Biomedical Physics Group';
opts.AuthorCompany = 'Lancaster University';
opts.MinimumMatlabRelease = 'R2017a';
opts.OutputFile = fullfile(repoRoot, 'MODA.mltbx');

% Explicit whitelist: only the MATLAB desktop app, not FastMODA (Python),
% the docs website, Docker/deployment tooling, or anything else at repo
% root that isn't part of this toolbox.
included = {'MODA.m', 'MODAApp.m', 'allguis', 'example_sigs', 'README.md', 'LICENSE', 'User Manual.pdf'};
toolboxFiles = {};
for i = 1:numel(included)
    p = fullfile(repoRoot, included{i});
    if exist(p, 'file') || exist(p, 'dir')
        toolboxFiles{end+1} = p; %#ok<AGROW>
    else
        warning('buildToolbox:missingFile', 'Expected file/folder not found, skipping: %s', p);
    end
end
opts.ToolboxFiles = toolboxFiles;

% So MATLAB automatically puts the algorithm code on the path once
% installed, matching what MODAApp's constructor currently does by hand
% via addpath(genpath(...)).
opts.ToolboxMatlabPath = {fullfile(repoRoot, 'allguis')};

matlab.addons.toolbox.packageToolbox(opts);
fprintf('Built %s\n', opts.OutputFile);
end
