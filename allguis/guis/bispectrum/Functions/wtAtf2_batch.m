% wtAtf2_batch — batched version of wtAtf2.m
% author: (batched variant, mirrors wtAtf2.m's algorithm exactly)
%
% Computes the wavelet transform of the SAME signal at MULTIPLE arbitrary
% frequencies in one call, instead of one call per frequency. The signal
% detrending/filtering/padding/FFT (the expensive, frequency-independent
% part of wtAtf2.m) is done exactly once and shared across all requested
% frequencies; only the per-frequency kernel build, ifft, and edge-trim
% are repeated (across a batched matrix, not a loop).

function wt = wtAtf2_batch(sig, fs, frArray, opt)
% INPUT:
% sig:      signal
% fs:       sampling frequency
% frArray:  vector of frequencies at which the wavelet transform is computed
% opt:      structure of optimal parameters returned by wt.m
%
% OUTPUT:
% wt:       M x N matrix, wt(m,:) = wtAtf2(sig, fs, frArray(m), opt)

p = 1;
N = length(sig); sig = sig(:);
PadMode = opt.Padding;
L = N; fmin = opt.fmin; fmax = opt.fmax;
frArray = frArray(:).';
M = numel(frArray);

% ======== Signal preprocessing: detrending, filtering and padding =========
% (identical to wtAtf2.m — this block does not depend on frArray, so it
% now runs once per call instead of once per frequency)
dflag = 0;
if ~iscell(PadMode)
    if ~ischar(PadMode) && ~isempty(PadMode(PadMode ~= 0)), dflag = 1; end
    if strcmpi(PadMode, 'predictive') && fmin < 5 * fs / L, dflag = 1; end
else
    if ~ischar(PadMode{1}) && ~isempty(PadMode{1}(PadMode{1} ~= 0)), dflag = 1; end
    if ~ischar(PadMode{2}) && ~isempty(PadMode{2}(PadMode{2} ~= 0)), dflag = 1; end
    if strcmpi(PadMode{1}, 'predictive') && fmin < 5 * fs / L, dflag = 1; end
    if strcmpi(PadMode{2}, 'predictive') && fmin < 5 * fs / L, dflag = 1; end
end

if strcmpi(opt.Preprocess, 'on') && dflag == 0
    [sig, ~, ~] = preprocess(sig, L, fs, fmin, fmax);
end
padleft = opt.PadLR{1}; padright = opt.PadLR{2};
sig = [padleft; sig; padright];
NL = length(sig);

if strcmpi(opt.Preprocess, 'on') && dflag == 1
    [sig, ~, ~] = preprocess(sig, NL, fs, 0, fs / 2);
end

Nq = ceil((NL + 1) / 2);
ff = [(0 : Nq - 1), -fliplr(1 : NL - Nq)] * fs / NL; ff = ff(:);
fx = fft(sig, NL); fx(ff <= 0) = 0;
if strcmpi(opt.Preprocess,'on')
    fx(ff <= max([fmin, fs / L]) | ff >= fmax) = 0;
end

% ======== Per-frequency edge-trim parameters (cheap scalar math) =========
coib1 = ceil(abs(opt.wp.t1e * fs * (opt.wp.ompeak ./ (2 * pi * frArray))));
coib2 = ceil(abs(opt.wp.t2e * fs * (opt.wp.ompeak ./ (2 * pi * frArray))));
if (opt.wp.t2e - opt.wp.t1e) * opt.wp.ompeak / (2 * pi * fmax) > L / fs
    coib1(:) = 0; coib2(:) = 0;
end

n1 = zeros(1,M); n2 = zeros(1,M);
zeroMask = (coib1==0 & coib2==0);
n1(zeroMask) = floor((NL-L)/2); n2(zeroMask) = ceil((NL-L)/2);
nz = ~zeroMask;
n1(nz) = floor((NL-L)*coib1(nz)./(coib1(nz)+coib2(nz)));
n2(nz) = ceil((NL-L)*coib2(nz)./(coib1(nz)+coib2(nz)));

% ======== Batched kernel build + wavelet transform ========
% freqwf: NL x M — one column per requested frequency (broadcast of the
% same per-frequency rescale wtAtf2.m applies one frequency at a time)
freqwf = ff * (opt.wp.ompeak ./ (2 * pi * frArray));
inSupport = freqwf > opt.wp.xi1/(2*pi) & freqwf < opt.wp.xi2/(2*pi);

if ~isempty(opt.wp.fwt)
    FW = zeros(NL, M);
    FW(inSupport) = conj(opt.wp.fwt(2*pi*freqwf(inSupport)));
    badId = isnan(FW) | ~isfinite(FW);
    if any(badId(:))
        FW(badId) = conj(opt.wp.fwt(2*pi*freqwf(badId) + 1e-14));
        stillBad = isnan(FW) | ~isfinite(FW);
        FW(stillBad) = 0;
    end
else
    twav = (1/fs)*[-(1:ceil((NL-1)/2))+1, NL+1-(ceil((NL-1)/2)+1:NL)]';
    timewf = twav * (2*pi*frArray/opt.wp.ompeak); % NL x M
    inTimeSupport = timewf > opt.wp.t1 & timewf < opt.wp.t2;
    TW = zeros(NL, M);
    TW(inTimeSupport) = conj(opt.wp.twf(timewf(inTimeSupport)));
    badId = isnan(TW) | ~isfinite(TW);
    if any(badId(:))
        TW(badId) = conj(opt.wp.twf(timewf(badId) + 1e-14));
        stillBad = isnan(TW) | ~isfinite(TW);
        TW(stillBad) = 0;
    end
    FW = (1/fs) * fft(TW, [], 1); % column-wise fft, NL x M
end
FW(~inSupport) = 0;

CC = fx .* FW; % NLx1 broadcasts against NLxM
WTfull = ifft(CC, NL, 1); % batched column-wise ifft, NL x M

normFactor = (opt.wp.ompeak ./ (2*pi*frArray)).^(1-p); % 1 x M

wt = NaN(M, N);
for m = 1:M
    seg = WTfull(1+n1(m):NL-n2(m), m) * normFactor(m);
    wt(m,1:L) = seg.';
    if strcmpi(opt.CutEdges,'on')
        wt(m,1:coib1(m)) = nan;
        wt(m,end-coib2(m):end) = nan;
    end
end
end

function [newSig, fx, ff] = preprocess(sig, N, fs, fmin, fmax)
% Detrending
X = (1 : length(sig))' / fs; XM = ones(length(X), 4);
for pn = 1 : 3
    CX = X .^ pn;
    XM(:, pn + 1) = (CX - mean(CX)) / std(CX);
end
w = warning('off', 'all'); sig = sig - XM * (pinv(XM) * sig); warning(w);

% Filtering
fx = fft(sig, N); % Fourier transform of a signal
Nq = ceil((N + 1) / 2);
ff = [(0 : Nq - 1), -fliplr(1 : N - Nq)] * fs / N; ff = ff(:); % frequencies in Fourier transform
fx(abs(ff) <= max([fmin, fs / N]) | abs(ff) >= fmax) = 0; % filter signal in a chosen frequency domain
newSig = ifft(fx);
end
