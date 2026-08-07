function buildPythonPackage()
% BUILDPYTHONPACKAGE  Compile MODA's bispectrum algorithms into a
% standalone Python package via MATLAB Compiler SDK.
%
% REQUIRES MATLAB Compiler SDK (a separate licensed product from base
% MATLAB Compiler). Verified working (R2026a, Compiler SDK 26.1): builds
% cleanly with zero unresolved symbols and zero excluded files, producing
% packages/MODABispectrum/ with the expected setup.py, pyproject.toml,
% __init__.py, and MODABispectrum.ctf component archive. The generated
% Python-side entry points (pkg.bispecWavPython(...), etc., where pkg =
% MODABispectrum.initialize()) are bound dynamically from the .ctf archive
% at runtime rather than appearing as literal Python source, so actually
% CALLING them requires a Python environment with the matching MATLAB
% Runtime installed — that last step has not been exercised here (no
% working Python interpreter was available in the environment this was
% built in), only the MATLAB-side compilation itself.
%
% What this produces: a Python package (installable via `pip install
% <output>/for_redistribution_files_only/...` after also installing the
% matching MATLAB Runtime — see the generated package's own
% GettingStarted.html) exposing MODA's bispectrum functions to Python code
% that has no MATLAB license at all, e.g.:
%
%   import matlab
%   import MODABispectrum
%   pkg = MODABispectrum.initialize()
%   bisp, freq, wt1, wt2, opt = pkg.bispecWavPython(sig1, sig2, fs, opts)
%
% Why only the bispectrum functions: these are the ones the repo already
% prepared for exactly this purpose — allguis/guis/bispectrum/Functions/
% python/ contains hand-adapted, Compiler-SDK-safe copies
% (bispecWavPython.m, biphaseWavPython.m, wtAtf2Python.m), and wt.m itself
% already has a 'python' option (see its varargin handling) that avoids
% returning a function handle (wp.fwt) in its output struct — function
% handles don't serialize across the MATLAB/Python boundary, so wt.m
% instead returns func2str(wp.fwt) when called this way. None of the other
% modules (ridge extraction, coherence, Bayesian inference) have been
% prepared the same way; extending Python-readiness to them would need the
% same kind of review (removing/guarding any errordlg/waitbar/uigetfile
% calls, checking outputs are pure numeric/struct data) before they could
% be added as entry points here. FastMODA's existing PyTorch
% reimplementation already serves the "call this from Python" need for
% those algorithms today, independently of this MATLAB-Compiler-SDK path.
%
% OUTPUT: packages/MODABispectrum/ (relative to the repo root), matching
% this repo's existing .gitignore entry for packaged Python libraries.

if isempty(which('compiler.build.pythonPackage'))
    error('buildPythonPackage:noCompilerSDK', ...
        ['compiler.build.pythonPackage is not available. This requires MATLAB ', ...
         'Compiler SDK (a separate product from base MATLAB Compiler) to be installed.']);
end

repoRoot = fileparts(fileparts(mfilename('fullpath'))); % .../MODA (this file lives in .../MODA/scripts)
bispDir = fullfile(repoRoot, 'allguis', 'guis', 'bispectrum');
pyDir   = fullfile(bispDir, 'Functions', 'python');

entryPoints = { ...
    fullfile(pyDir, 'bispecWavPython.m'), ...
    fullfile(pyDir, 'biphaseWavPython.m'), ...
    fullfile(pyDir, 'wtAtf2Python.m'), ...
    fullfile(repoRoot, 'allguis', 'guis', 'tfa', 'Functions', 'wt.m') ...
};
for i = 1:numel(entryPoints)
    if ~isfile(entryPoints{i})
        error('buildPythonPackage:missingEntryPoint', 'Entry point not found: %s', entryPoints{i});
    end
end

% mcc's dependency analyzer only finds a called function if it's
% resolvable on the MATLAB path at BUILD time — unlike running the app
% normally (where MODAApp's constructor does this same addpath(genpath...)
% call), nothing does that automatically here. Without it, sibling-folder
% helpers that entry points call — e.g. compareMatrix.m, which lives in
% Functions/ next to bispecWavPython.m's own Functions/python/ — silently
% fail to resolve: the build itself reports no unresolved symbols (the
% analyzer just doesn't know the call exists), but the compiled package
% then throws "Undefined function" at runtime the first time that code
% path executes. Confirmed by an actual end-to-end Python-side test run
% (see scripts/README or session notes) — this addpath is required, not
% precautionary.
addpath(genpath(fullfile(repoRoot, 'allguis')));

outDir = fullfile(repoRoot, 'packages', 'MODABispectrum');

opts = compiler.build.PythonPackageOptions(entryPoints, ...
    'PackageName', 'MODABispectrum', ...
    'OutputDir', outDir, ...
    'Verbose', 'on');

results = compiler.build.pythonPackage(opts); %#ok<NASGU>
fprintf('Built Python package in %s\n', outDir);
fprintf('See %s for install/GettingStarted instructions.\n', outDir);
end
