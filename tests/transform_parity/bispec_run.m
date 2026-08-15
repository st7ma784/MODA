function bispec_run(repoRoot, baselineDir)
%BISPEC_RUN  Bispectrum parity across the vectorised transform change.
%
%   The bispectrum chain (bispecWavNew -> myWt / wtAtf2 / wtAtf2_batch)
%   calls wt() by name across several files, so the two implementations are
%   swapped by putting the pre-vectorisation copies at the front of the path
%   and rehashing, rather than by renaming functions.

addpath(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(repoRoot,'allguis')));

fs = 40; L = 512;
rng(2024,'twister');
t  = (0:L-1)'/fs;
% two signals with genuine quadratic phase coupling, so the bispectrum has
% real structure rather than being numerical noise
ph = 2*pi*1.0*t;
sig1 = sin(ph) + 0.5*sin(2*ph) + 0.3*sin(3*ph) + 0.05*randn(L,1);
sig2 = sin(ph + 0.4) + 0.5*sin(2*ph + 0.4) + 0.05*randn(L,1);

fprintf('\n===== BISPECTRUM PARITY =====\n');

% ---- baseline (pre-vectorisation wt/wft shadowing production) ----
% transformCached keeps a session-wide persistent cache keyed on the signal
% and parameters only — nothing identifying WHICH wt implementation produced
% the entry. Swapping the path mid-session would otherwise give the second
% run a stale hit from the first, making this comparison vacuous. Clear it
% before each side.
clear transformCached
addpath(baselineDir,'-begin'); rehash;
fprintf('baseline wt -> %s\n', which('wt'));
okA = true;
try
    [BispA, freqA] = bispecWavNew(sig1, sig2, fs);
catch ME
    okA = false; errA = sprintf('%s: %s', ME.identifier, ME.message);
    fprintf('baseline ERROR %s\n', errA);
end
rmpath(baselineDir); rehash;

% ---- production ----
clear transformCached          % force a genuine recomputation, not a cache hit
fprintf('production wt -> %s\n', which('wt'));
okB = true;
try
    [BispB, freqB] = bispecWavNew(sig1, sig2, fs);
catch ME
    okB = false; errB = sprintf('%s: %s', ME.identifier, ME.message);
    fprintf('production ERROR %s\n', errB);
end

if ~okA || ~okB
    fprintf('VERDICT: could not compare (one side failed)\n');
    fprintf('===== end =====\n'); return
end

fprintf('Bisp size: baseline %s, production %s\n', ...
        mat2str(size(BispA)), mat2str(size(BispB)));

eF = relerr(freqA(:), freqB(:));
eB = relerr(BispA, BispB);
fprintf('freq  rel error : %.3e\n', eF);
fprintf('Bisp  rel error : %.3e\n', eB);
if isfinite(eB) && eB <= 1e-9 && isfinite(eF) && eF <= 1e-9
    fprintf('VERDICT: MATCH\n');
else
    fprintf('VERDICT: MISMATCH\n');
end
fprintf('===== end =====\n');
end

function e = relerr(A,B)
if ~isequal(size(A),size(B)), e = Inf; return, end
na=isnan(A); nb=isnan(B);
if ~isequal(na,nb), e = Inf; return, end
Av=A(~na); Bv=B(~nb);
if isempty(Av), e=0; return, end
sc = max(abs(Av(:))); if sc==0 || ~isfinite(sc), sc=1; end
e = max(abs(Av-Bv))/sc;
end
