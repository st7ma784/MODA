function e = fitBinsToPeaksEdges(freqs, P, smoothW)
%FITBINSTOPEAKSEDGES  Bin edges placed at troughs of the marginal spectrum, so
%   each resulting bin is centred on a peak. Mirrors
%   fastmoda.spectral_bins.fit_bins_to_peaks_edges.
    if nargin < 3 || isempty(smoothW), smoothW = 5; end
    freqs = freqs(:);
    Ps = local_smooth(P(:), smoothW);
    [~, troughIdx] = findpeaks(-Ps);         % local minima of the marginal
    idx = unique([1; troughIdx(:); numel(freqs)]);
    e = freqs(idx).';
end

function y = local_smooth(x, w)
    w = max(1, round(w));
    if mod(w, 2) == 0, w = w + 1; end
    if w <= 1 || numel(x) < w, y = x; return; end
    y = movmean(x, w);
end
