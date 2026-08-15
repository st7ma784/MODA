function band = binPowerOverTime(freqs, Sxx, edges, usePower)
%BINPOWEROVERTIME  Integrate a spectrogram over frequency bins, per time column.
%   Mirrors fastmoda.changepoint.bin_power_over_time. Returns a (T x nBins)
%   matrix: band energy in each bin at each time.
%
%   freqs : frequency vector (length F)
%   Sxx   : magnitude spectrogram (F x T)
%   edges : bin edges (length nBins+1), from uniformEdges()
%   usePower : true -> square magnitude (energy) first (default true)
    if nargin < 4 || isempty(usePower), usePower = true; end
    freqs = freqs(:);
    S = Sxx;
    if usePower, S = S.^2; end
    edges = edges(:);
    T = size(S, 2);
    nb = numel(edges) - 1;
    band = zeros(T, nb);
    for i = 1:nb
        lo = edges(i); hi = edges(i+1);
        if i == nb
            m = freqs >= lo & freqs <= hi;
        else
            m = freqs >= lo & freqs < hi;
        end
        if nnz(m) >= 2
            band(:, i) = trapz(freqs(m), S(m, :), 1).';
        elseif nnz(m) == 1
            band(:, i) = sum(S(m, :), 1).';
        end
    end
end
