function downstream_run(repoRoot, refDir)
%DOWNSTREAM_RUN  End-to-end parity through ridge extraction and coherence.
%
%   ecurve/rectfr/wphcoh/tlphcoh all accept a precomputed transform, so both
%   implementations can be driven in one process: the transform is computed
%   with wt_base/wft_base and with the production wt/wft, then the identical
%   downstream chain is run on each and the results compared.
%
%   This mirrors ridge_extraction.m, whose body is:
%       [WT,freqarr,wopt] = wt(...);
%       tfsupp = ecurve(WT,freqarr,wopt);
%       [iamp,iphi,ifreq] = rectfr(tfsupp,WT,freqarr,wopt,'direct');
%       recon = iamp.*cos(iphi);

addpath(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(repoRoot,'allguis')));
addpath(refDir);

fs = 40; L = 512;
sig1 = mksig(L, fs, 1);
sig2 = mksig(L, fs, 2);

wavelets = {'Lognorm','Morlet','Bump'};
windows  = {'Gaussian','Hann','Blackman'};

pass=0; fail=0; worst=0; worstWhat='';
fprintf('\n===== DOWNSTREAM: ridge extraction (ecurve -> rectfr) =====\n');

for i=1:numel(wavelets)
    [ok,w,what] = ridge_case('wt', wavelets{i}, sig1, fs);
    [pass,fail,worst,worstWhat] = tally(ok,w,what,pass,fail,worst,worstWhat);
end
for i=1:numel(windows)
    [ok,w,what] = ridge_case('wft', windows{i}, sig1, fs);
    [pass,fail,worst,worstWhat] = tally(ok,w,what,pass,fail,worst,worstWhat);
end

fprintf('\n===== DOWNSTREAM: coherence (wphcoh / tlphcoh) =====\n');
for i=1:numel(wavelets)
    [ok,w,what] = coh_case('wt', wavelets{i}, sig1, sig2, fs);
    [pass,fail,worst,worstWhat] = tally(ok,w,what,pass,fail,worst,worstWhat);
end
for i=1:numel(windows)
    [ok,w,what] = coh_case('wft', windows{i}, sig1, sig2, fs);
    [pass,fail,worst,worstWhat] = tally(ok,w,what,pass,fail,worst,worstWhat);
end

fprintf('\n===== DOWNSTREAM SUMMARY =====\n');
fprintf('passed          : %d\n', pass);
fprintf('failed          : %d\n', fail);
fprintf('worst rel error : %.3e\n', worst);
fprintf('worst quantity  : %s\n', worstWhat);
fprintf('===== end =====\n');
end

% ------------------------------------------------------------------------
function [ok,worst,worstWhat] = ridge_case(fn, kern, sig, fs)
ok=true; worst=0; worstWhat='';
opts = {'fmin',0.2,'fmax',8,'CutEdges','off','Preprocess','on', ...
        'Display','off','Plot','off'};
if strcmp(fn,'wt'), opts=[opts {'Wavelet',kern}]; else, opts=[opts {'Window',kern}]; end

try
    [Wa,fa,wa] = callbase(fn,sig,fs,opts);
    [Wb,fb,wb] = callprod(fn,sig,fs,opts);

    sA = ecurve(Wa,fa,wa);  sB = ecurve(Wb,fb,wb);
    [ampA,phiA,frqA] = rectfr(sA,Wa,fa,wa,'direct');
    [ampB,phiB,frqB] = rectfr(sB,Wb,fb,wb,'direct');
    recA = ampA.*cos(phiA); recB = ampB.*cos(phiB);

    checks = { 'tfsupp', sA, sB; 'iamp', ampA, ampB; ...
               'ifreq', frqA, frqB; 'recon', recA, recB };
    for k=1:size(checks,1)
        e = relerr(checks{k,2}, checks{k,3});
        if e>worst, worst=e; worstWhat=sprintf('%s|%s|%s',fn,kern,checks{k,1}); end
    end
    % phase compared on the unit circle so 2*pi wrapping is not a difference
    e = relerr(exp(1i*phiA), exp(1i*phiB));
    if e>worst, worst=e; worstWhat=sprintf('%s|%s|iphi',fn,kern); end

    ok = worst <= 1e-9;
    fprintf('  %-4s %-10s ridge : %s  (worst %.3e)\n', fn, kern, verdict(ok), worst);
catch ME
    ok=false; fprintf('  %-4s %-10s ridge : ERROR %s: %s\n', fn, kern, ME.identifier, ME.message);
end
end

% ------------------------------------------------------------------------
function [ok,worst,worstWhat] = coh_case(fn, kern, s1, s2, fs)
ok=true; worst=0; worstWhat='';
opts = {'fmin',0.2,'fmax',8,'CutEdges','off','Preprocess','on', ...
        'Display','off','Plot','off'};
if strcmp(fn,'wt'), opts=[opts {'Wavelet',kern}]; else, opts=[opts {'Window',kern}]; end

try
    [W1a,fa,~] = callbase(fn,s1,fs,opts);
    [W2a,~,~ ] = callbase(fn,s2,fs,opts);
    [W1b,fb,~] = callprod(fn,s1,fs,opts);
    [W2b,~,~ ] = callprod(fn,s2,fs,opts);

    pA = wphcoh(W1a,W2a);   pB = wphcoh(W1b,W2b);
    e1 = relerr(pA,pB);
    if e1>worst, worst=e1; worstWhat=sprintf('%s|%s|wphcoh',fn,kern); end

    tA = tlphcoh(W1a,W2a,fa,fs);  tB = tlphcoh(W1b,W2b,fb,fs);
    e2 = relerr(tA,tB);
    if e2>worst, worst=e2; worstWhat=sprintf('%s|%s|tlphcoh',fn,kern); end

    ok = worst <= 1e-9;
    fprintf('  %-4s %-10s coh   : %s  (worst %.3e)\n', fn, kern, verdict(ok), worst);
catch ME
    ok=false; fprintf('  %-4s %-10s coh   : ERROR %s: %s\n', fn, kern, ME.identifier, ME.message);
end
end

% ------------------------------------------------------------------------
function [W,f,w] = callbase(fn,sig,fs,opts)
if strcmp(fn,'wt'), [W,f,w]=wt_base(sig,fs,opts{:}); else, [W,f,w]=wft_base(sig,fs,opts{:}); end
end
function [W,f,w] = callprod(fn,sig,fs,opts)
if strcmp(fn,'wt'), [W,f,w]=wt(sig,fs,opts{:}); else, [W,f,w]=wft(sig,fs,opts{:}); end
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

function s = verdict(ok)
if ok, s='MATCH'; else, s='FAIL '; end
end

function [pass,fail,worst,worstWhat] = tally(ok,w,what,pass,fail,worst,worstWhat)
if ok, pass=pass+1; else, fail=fail+1; end
if w>worst && isfinite(w), worst=w; worstWhat=what; end
end

function sig = mksig(L, fs, seed)
rng(1000+seed,'twister');
t=(0:L-1)'/fs;
sig = sin(2*pi*(1.0+0.01*t).*t) + 0.6*sin(2*pi*2.5*t + seed) + 0.05*randn(L,1);
end
