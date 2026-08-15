function out = changepointsLogBinnedPower(freqs, times, Sxx, varargin)
%CHANGEPOINTSLOGBINNEDPOWER  Changepoints on the full power split into freq bins.
%   Mirrors fastmoda.changepoint.changepoints_logbinned_power. Bins the
%   spectrogram into log (or linear) frequency bands, forming a (T x nBins)
%   feature matrix, and finds changepoints jointly across all bands with
%   findchangepts — a change in any band is detected.
%
%   out = changepointsLogBinnedPower(freqs, times, Sxx, Name, Value)
%     Name-Value: 'NBins' (12), 'Scale' ('log'|'linear'), 'UsePower' (true),
%                 'MinThreshold' ([]=auto), 'Statistic' ('mean'),
%                 'Fmin' ([]), 'Fmax' ([])
%
%   out fields: changepoints (time idx), changepoint_times (s), band_power
%   (T x nBins), bin_edges, bin_centers, scale, n_bins, times.

    p = inputParser;
    addParameter(p, 'NBins', 12);
    addParameter(p, 'Scale', 'log');
    addParameter(p, 'UsePower', true);
    addParameter(p, 'MinThreshold', []);
    addParameter(p, 'Statistic', 'mean');
    addParameter(p, 'Fmin', []);
    addParameter(p, 'Fmax', []);
    parse(p, varargin{:});

    freqs = freqs(:); times = times(:);
    lo = p.Results.Fmin; if isempty(lo), lo = max(min(freqs(freqs > 0)), 1e-9); end
    hi = p.Results.Fmax; if isempty(hi), hi = max(freqs); end
    edges = uniformEdges(lo, hi, p.Results.NBins, p.Results.Scale);

    % NaN in Sxx (e.g. cone-of-influence-masked WT) → treat as no energy, so the
    % band integral is finite and findchangepts accepts the feature matrix.
    Sxx(~isfinite(Sxx)) = 0;
    band = binPowerOverTime(freqs, Sxx, edges, p.Results.UsePower);   % (T x nb)
    feat = local_sanitize(local_standardize(band));
    thr = p.Results.MinThreshold;
    if isempty(thr), thr = local_autoThreshold(feat); end

    % findchangepts on a matrix treats each ROW as a channel and finds points
    % common across channels, so pass (nBins x T).
    ipt = findchangepts(feat.', 'Statistic', p.Results.Statistic, ...
                        'MinThreshold', thr);
    ipt = ipt(ipt > 1 & ipt < numel(times));

    centers = sqrt(max(edges(1:end-1), 1e-9) .* edges(2:end));
    out = struct();
    out.changepoints = ipt(:).';
    out.changepoint_times = times(ipt).';
    out.band_power = band;
    out.bin_edges = edges;
    out.bin_centers = centers;
    out.scale = p.Results.Scale;
    out.n_bins = numel(centers);
    out.times = times;
end

function z = local_standardize(a)
    z = (a - mean(a, 1, 'omitnan')) ./ (std(a, 0, 1, 'omitnan') + 1e-12);
end

function z = local_sanitize(z)
    % findchangepts rejects NaN/Inf; after standardisation 0 is the neutral mean.
    z(~isfinite(z)) = 0;
end

function thr = local_autoThreshold(feat)
    [T, d] = size(feat);
    base = log(max(T, 2)) * d;
    variability = mean(std(feat, 0, 1), 'omitnan');
    thr = base * (1 + variability);
end
