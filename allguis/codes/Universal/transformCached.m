function [WT, freqarr, wopt] = transformCached(tfun, typeTag, varargin)
% TRANSFORMCACHED  Cached dispatch to wt()/wft(), shared across every GUI
% module in one MATLAB session.
%
% Re-running the same signal through the same transform with the same
% parameters happens often in normal use — e.g. Coherence, Filtering, or
% Bispectrum recomputing a wavelet transform that the Time-Frequency
% Analysis tab (or an earlier button press in the same tab) already
% computed for the identical signal. This wrapper keeps a session-wide
% cache keyed on a fingerprint of the signal + all transform parameters,
% so a cache hit returns the previous result instead of repeating the
% (expensive) FFT-based transform.
%
% INPUT:
% tfun:     function handle, @wt or @wft
% typeTag:  'Wavelet' or 'Window' — included in the key so a wt() call and
%           a wft() call on identical (sig,fs,args) never collide
% varargin: the exact arguments to forward to tfun, i.e. tfun(varargin{:})
%           would be the uncached equivalent call
%
% OUTPUT: identical to calling tfun(varargin{:}) directly.

persistent cacheMap
if isempty(cacheMap)
    cacheMap = containers.Map('KeyType','char','ValueType','any');
end

key = localCacheKey(typeTag, varargin);
if isKey(cacheMap, key)
    cached = cacheMap(key);
    WT = cached.WT; freqarr = cached.freqarr; wopt = cached.wopt;
    return;
end

[WT, freqarr, wopt] = tfun(varargin{:});
cacheMap(key) = struct('WT', WT, 'freqarr', freqarr, 'wopt', wopt); %#ok<NASGU>

% Cap cache size so a long session doesn't grow memory unboundedly; simple
% FIFO eviction (not LRU) since MODA's typical use pattern is a handful of
% signals analysed across a handful of tabs, not thousands of distinct
% signals in one session.
if cacheMap.Count > 100
    ks = keys(cacheMap);
    remove(cacheMap, ks{1});
end
end

function key = localCacheKey(typeTag, argsCell)
% Not a cryptographic hash — a fingerprint specific enough in practice to
% distinguish genuinely different signals/parameters: signal length, fs,
% sum and sum-of-squares (catches amplitude/scale differences), a spread
% of sample values (catches shape differences same-length/same-energy
% signals could otherwise share), plus every scalar/string parameter
% serialized in argument order (order matters for name-value pairs, which
% is fine since call sites always build them in a fixed order).
sig = argsCell{1}(:);
fs = argsCell{2};
n = numel(sig);
sampleIdx = unique(round(linspace(1, n, min(n, 8))));
fp = [n, fs, sum(sig), sum(sig.^2), sig(sampleIdx).'];

paramStr = typeTag;
for k = 3:numel(argsCell)
    v = argsCell{k};
    if ischar(v) || isstring(v)
        paramStr = [paramStr, '|', char(v)]; %#ok<AGROW>
    elseif isnumeric(v) || islogical(v)
        paramStr = [paramStr, '|', mat2str(v)]; %#ok<AGROW>
    else
        paramStr = [paramStr, '|other']; %#ok<AGROW>
    end
end
key = sprintf('%s#%s', mat2str(fp), paramStr);
end
