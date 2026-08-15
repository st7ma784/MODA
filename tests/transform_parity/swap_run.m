function swap_run(repo, baseDir)
% swap_run  Checks the suite's main run does not cover.
%
%   swap_run(repo, baseDir)
%
% repo     repository root
% baseDir  the .baseline folder written by make_baseline.sh (uses ref/ and swap/)
%
% 1. Each rewritten routine against upstream's copy, swapped in on its own:
%    wphcoh, ecurve, the bispectrum row loop (bispecWavNew), and the Bayesian
%    chain CFprint, dirc, bayes_main, full_bayesian. The upstream copies are
%    renamed <name>_base and call everything else by name, so a difference
%    can only come from that one file.
% 2. Numeric 'nv' (wt) and 'fstep' (wft), against upstream (item 6).
% 3. The overflow retry path in wt and wft (item 8): a wavelet and a window
%    that overflow to NaN far out on their support, against upstream.
% 4. A wavelet given only in the time domain, against the built-in Morlet,
%    through both the batched path and the serial loop.
% 5. The GPU path against the CPU (item 2), when a GPU is available.
%
% Same metric and thresholds as parity_run: max|X-Y|/max|X|, NaN positions
% must agree, 1e-12 for transforms and 1e-9 downstream.

addpath(genpath(fullfile(repo,'allguis')));
addpath(fullfile(baseDir,'ref'));
addpath(fullfile(baseDir,'swap'));

R = struct('id',{},'rel',{},'tol',{},'note',{});

fs = 40; L = 2400; t = (0:L-1)'/fs;
rng(3);
s1 = cos(2*pi*1.1*t + 0.4*sin(2*pi*0.07*t)) + 0.6*cos(2*pi*3.2*t) + 0.2*randn(L,1);
s2 = cos(2*pi*1.1*t + 0.9) .* (1 + 0.5*cos(2*pi*3.2*t)) + 0.2*randn(L,1);
wa = {'fmin',0.3,'fmax',8,'f0',1,'Display','off','Plot','off'};

% ---------------------------------------------------------------- wphcoh
secName = 'wphcoh'; secT = tic;
for cut = {'on','off'}
    W1 = wt(s1,fs,wa{:},'CutEdges',cut{1});
    W2 = wt(s2,fs,wa{:},'CutEdges',cut{1});
    [cA,dA] = wphcoh_base(W1,W2); [cB,dB] = wphcoh(W1,W2);
    R(end+1) = rec(['wphcoh | CutEdges=' cut{1}], max(relerr(cA,cB),phaseerr(dA,dB)), 1e-9);
end

fprintf('  [%-26s %6.1f s]\n', secName, toc(secT));
% ---------------------------------------------------------------- ecurve
secName = 'ecurve'; secT = tic;
[W1,f1,o1] = wt(s1,fs,wa{:},'CutEdges','off');
for m = {1, 2, 'max'}
    eo = {'Display','off','Method',m{1}};
    tA = ecurve_base(W1,f1,o1,eo{:}); tB = ecurve(W1,f1,o1,eo{:});
    R(end+1) = rec(sprintf('ecurve | Method=%s', num2str(m{1})), relerr(tA,tB), 1e-12);
end

fprintf('  [%-26s %6.1f s]\n', secName, toc(secT));
% ---------------------------------------------------------- bispectrum row
secName = 'bispectrum row'; secT = tic;
for pair = {{s1,s2,'cross'},{s1,s1,'auto'}}
    p = pair{1};
    clear transformCached
    BA = bispecWavNew_base(p{1},p{2},fs,wa{:});
    clear transformCached
    BB = bispecWavNew(p{1},p{2},fs,wa{:});
    R(end+1) = rec(['bispecWavNew | ' p{3}], relerr(BA,BB), 1e-9);
end

fprintf('  [%-26s %6.1f s]\n', secName, toc(secT));
% ------------------------------------------------------------ Bayesian
secName = 'Bayesian'; secT = tic;
rng(11);
for bn = 1:3
    K = (2*bn+1)^2; cc = randn(1,2*K);
    [~,~,qa1,qa2] = CFprint_base(cc,bn); [~,~,qb1,qb2] = CFprint(cc,bn);
    R(end+1) = rec(sprintf('CFprint | bn=%d', bn), max(relerr(qa1,qb1),relerr(qa2,qb2)), 1e-12);
    [a1,a2] = dirc_base(cc,bn); [b1,b2] = dirc(cc,bn);
    R(end+1) = rec(sprintf('dirc | bn=%d', bn), max(relerr(a1,b1),relerr(a2,b2)), 1e-12);
end

[x1,x2,bfs] = coupled_pair();
ba = {50, 1/bfs, 0.5, 0.2};          % win (s), h, ovr, pr
p1 = angle(hilbert(x1)); p2 = angle(hilbert(x2));
for bn = 1:2
    [tA,cA,eA] = bayes_main_base(p1,p2,ba{1},ba{2},ba{3},ba{4},0,bn);
    [tB,cB,eB] = bayes_main(p1,p2,ba{1},ba{2},ba{3},ba{4},0,bn);
    R(end+1) = rec(sprintf('bayes_main | bn=%d', bn), max([relerr(tA,tB),relerr(cA,cB),relerr(eA,eB)]), 1e-12);
end

fbArgs = {x1, x2, 0.8, 1.2, 1.1, 1.5, bfs, 50, 0.2, 0.5, 2, 4, 95};
outA = cell(1,11); outB = cell(1,11);
rng(5); [outA{:}] = full_bayesian_base(fbArgs{:});
rng(5); [outB{:}] = full_bayesian(fbArgs{:});
e = 0; for k = 1:11, e = max(e, relerr(outA{k},outB{k})); end
R(end+1) = rec('full_bayesian | bn=2, 4 surrogates', e, 1e-12);

fprintf('  [%-26s %6.1f s]\n', secName, toc(secT));
% -------------------------------------------------- numeric nv / fstep
secName = 'numeric nv / fstep'; secT = tic;
[A,fA] = wt_base(s1,fs,wa{:},'nv',16); [B,fB] = wt(s1,fs,wa{:},'nv',16);
R(end+1) = rec('wt | nv=16 (numeric)', max(relerr(A,B),relerr(fA,fB)), 1e-12);
wfa = {'fmin',0.3,'fmax',8,'f0',1,'Display','off','Plot','off'};
[A,fA] = wft_base(s1,fs,wfa{:},'fstep',0.05); [B,fB] = wft(s1,fs,wfa{:},'fstep',0.05);
R(end+1) = rec('wft | fstep=0.05 (numeric)', max(relerr(A,B),relerr(fA,fB)), 1e-12);

fprintf('  [%-26s %6.1f s]\n', secName, toc(secT));
% ------------------------------------------------------- overflow retry
secName = 'overflow retry'; secT = tic;
% Morlet written so the product underflows times overflows (0*Inf = NaN)
% for xi above ~113, inside the support; the retry at xi+1e-14 fails too,
% so those bins are zeroed and the warning is printed.
om0 = 2*pi;
fwtO = @(xi) exp(-xi.^2/2).*exp(om0*xi)*exp(-om0^2/2) - exp(-(om0^2+xi.^2)/2);
ov = {'fmin',0.1,'fmax',8,'Display','off','Plot','off','Wavelet',{fwtO,[0 Inf],[],[]}};
[A,~,wA] = capture(@() wt_base(s1,fs,ov{:}));
[B,~,wB] = capture(@() wt(s1,fs,ov{:}));
R(end+1) = rec(sprintf('wt | overflow retry (warned: base %d, new %d)', wA, wB), relerr(A,B), 1e-12);
% Gaussian window written the same way: NaN for |xi| above ~37.7
fwtG = @(xi) exp(-xi.^2).*exp(xi.^2/2);
ovw = {'fmin',0.3,'fmax',8,'Display','off','Plot','off','Window',{fwtG,[-Inf Inf],[],[]}};
[A,~,wA] = capture(@() wft_base(s1,fs,ovw{:}));
[B,~,wB] = capture(@() wft(s1,fs,ovw{:}));
R(end+1) = rec(sprintf('wft | overflow retry (warned: base %d, new %d)', wA, wB), relerr(A,B), 1e-12);

fprintf('  [%-26s %6.1f s]\n', secName, toc(secT));
% ------------------------------------------------- time-domain-only wavelet
secName = 'time-domain-only wavelet'; secT = tic;
% Upstream cannot run this case at all ("Unrecognized function or variable
% 'xx'"), so the reference is the built-in Morlet, which has the same twf
% in closed form. The tolerance reflects wt.m's own numerical inversion of
% twf to its Fourier transform (wavelet parameters agree to ~1e-8).
twfM = @(t)(1/sqrt(2*pi))*(exp(1i*om0*t)-exp(-om0^2/2)).*exp(-t.^2/2);
to = {'fmin',0.5,'fmax',8,'Display','off','Plot','off','CutEdges','off','Padding',0};
[M,fM] = wt(s1,fs,to{:},'Wavelet','Morlet','f0',1);
[T,fT] = wt(s1,fs,to{:},'Wavelet',{[],[],twfM,[-Inf Inf]});
R(end+1) = rec('wt | twf-only Morlet, batched', max(relerr(M,T),relerr(fM,fT)), 1e-6);
[T2,fT2] = wt(s1,fs,to{:},'Wavelet',{[],[],@(x)vecOnly(twfM,x),[-Inf Inf]});
R(end+1) = rec('wt | twf-only Morlet, serial loop', max(relerr(M,T2),relerr(fM,fT2)), 1e-6);
R(end+1) = rec('wt | twf-only, batched vs loop', relerr(T,T2), 1e-12);

fprintf('  [%-26s %6.1f s]\n', secName, toc(secT));
% ------------------------------------------------------------------ GPU
secName = 'GPU'; secT = tic;
gpuOk = false;
try, gpuOk = gpuDeviceCount > 0; catch, end
if gpuOk
    % NL*SN must exceed 4e6 for wt/wft to take the GPU branch
    Lg = 2^17; tg = (0:Lg-1)'/fs; sg = cos(2*pi*1.1*tg) + 0.3*randn(Lg,1);
    ga = {'fmin',0.05,'fmax',8,'f0',1,'Display','off','Plot','off'};
    A = wt_base(sg,fs,ga{:}); B = wt(sg,fs,ga{:});
    R(end+1) = rec('wt | GPU path vs upstream CPU', relerr(A,B), 1e-12);
    A = wft_base(sg,fs,ga{:}); B = wft(sg,fs,ga{:});
    R(end+1) = rec('wft | GPU path vs upstream CPU', relerr(A,B), 1e-12);
else
    fprintf('GPU path: SKIPPED (no Parallel Computing Toolbox or no GPU device)\n');
end

fprintf('  [%-26s %6.1f s]\n', secName, toc(secT));
% ---------------------------------------------------------------- report
fprintf('\n%-58s %12s %9s  %s\n', 'check', 'max rel', 'tol', 'result');
nfail = 0;
for k = 1:numel(R)
    ok = R(k).rel <= R(k).tol;
    nfail = nfail + ~ok;
    fprintf('%-58s %12.3e %9.0e  %s\n', R(k).id, R(k).rel, R(k).tol, tern(ok,'ok','FAIL'));
end
fprintf('\n%d checks, %d failed\n', numel(R), nfail);
if nfail > 0, error('swap_run:fail', '%d checks failed', nfail); end
end

% ======================================================================
function r = rec(id, rel, tol)
r = struct('id',id,'rel',rel,'tol',tol,'note','');
end

function e = relerr(A, B)
% max|A-B|/max|A|; Inf on a size mismatch or a NaN-pattern mismatch
if ~isequal(size(A),size(B)), e = Inf; return; end
if isstruct(A) || iscell(A), e = double(~isequaln(A,B))*Inf; return; end
nA = isnan(A); if ~isequal(nA, isnan(B)), e = Inf; return; end
d = abs(A(~nA) - B(~nA)); s = max(abs(A(~nA)));
if isempty(d), e = 0; elseif s == 0, e = max(d); else, e = max(d)/s; end
end

function e = phaseerr(a, b)
% phases compared on the unit circle so a 2*pi wrap does not count
e = relerr(exp(1i*a), exp(1i*b));
end

function [W,f,warned] = capture(fn)
txt = evalc('[W,f] = fn();');
warned = contains(txt, 'overflow');
end

function y = vecOnly(fn, x)
% Rejects matrix input, so wt.m's probe falls back to the serial loop
if ~isvector(x), error('swap_run:matrix', 'vector input only'); end
y = fn(x);
end

function [x1,x2,fs] = coupled_pair()
% Two noisy phase oscillators, 1 -> 2 coupling stronger than 2 -> 1
fs = 20; n = 6000; h = 1/fs;
p = zeros(n,2); p(1,:) = [0 1];
rng(21);
for k = 1:n-1
    d1 = 2*pi*1.0 + 0.3*sin(p(k,2)-p(k,1));
    d2 = 2*pi*1.3 + 0.9*sin(p(k,1)-p(k,2));
    p(k+1,:) = p(k,:) + h*[d1 d2] + sqrt(h)*0.15*randn(1,2);
end
x1 = cos(p(:,1)); x2 = cos(p(:,2));
end

function s = tern(c, a, b)
if c, s = a; else, s = b; end
end
