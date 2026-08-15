function e = uniformEdges(fmin, fmax, nBins, scale)
%UNIFORMEDGES  Frequency-bin edges over [fmin, fmax].
%   Mirrors fastmoda.spectral_bins.uniform_edges.
%   scale : 'log' (default) or 'linear'.
    if nargin < 3 || isempty(nBins), nBins = 20; end
    if nargin < 4 || isempty(scale), scale = 'log'; end
    nBins = max(1, round(nBins));
    fmin = max(fmin, 1e-12);
    if strcmpi(scale, 'log')
        e = logspace(log10(fmin), log10(fmax), nBins + 1);
    else
        e = linspace(fmin, fmax, nBins + 1);
    end
end
