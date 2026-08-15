function bins = binnedSpectrum(freqs, P, edges)
%BINNEDSPECTRUM  Integrate a marginal spectrum P over each frequency bin.
%   Mirrors fastmoda.spectral_bins.bin_spectrum. Returns a struct array with
%   fields f_lo, f_hi, f_center (geometric), density (integral) and
%   density_norm (density / max, for background-bar heights).
%
%   Use with uniformEdges() or fitBinsToPeaksEdges() to build the edges.
    freqs = freqs(:); P = P(:); edges = edges(:);
    nb = numel(edges) - 1;
    dens = zeros(nb, 1);
    for i = 1:nb
        lo = edges(i); hi = edges(i+1);
        if i == nb
            m = freqs >= lo & freqs <= hi;
        else
            m = freqs >= lo & freqs < hi;
        end
        if nnz(m) >= 2
            dens(i) = trapz(freqs(m), P(m));
        elseif nnz(m) == 1
            dens(i) = P(m);
        else
            dens(i) = 0;
        end
    end
    mx = max(dens); if mx == 0, mx = 1; end

    bins = struct('f_lo', cell(1, nb), 'f_hi', [], 'f_center', [], ...
                  'density', [], 'density_norm', []);
    for i = 1:nb
        bins(i).f_lo = edges(i);
        bins(i).f_hi = edges(i+1);
        bins(i).f_center = sqrt(max(edges(i), 1e-9) * edges(i+1));
        bins(i).density = dens(i);
        bins(i).density_norm = dens(i) / mx;
    end
end
