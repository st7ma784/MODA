function out = changepointsAtFrequency(freqs, times, Sxx, targetFreq, varargin)
%CHANGEPOINTSATFREQUENCY  Changepoints in the power/amplitude at one frequency.
%   Mirrors fastmoda.changepoint.changepoints_at_frequency. The chosen frequency
%   is snapped to the nearest spectrogram bin; changepoints are found in that
%   single time series with findchangepts (PELT-equivalent).
%
%   out = changepointsAtFrequency(freqs, times, Sxx, targetFreq, Name, Value)
%     Name-Value: 'UsePower' (true), 'MinThreshold' ([]=auto),
%                 'Statistic' ('mean')
%
%   out struct fields: changepoints (time indices), changepoint_times (s),
%   series, actual_freq, times.
%
%   NOTE: parity is behavioural (detects the same changes) — the MATLAB penalty
%   scale differs from ruptures', as documented in docs/roadmap/changepoints.

    p = inputParser;
    addParameter(p, 'UsePower', true);
    addParameter(p, 'MinThreshold', []);
    addParameter(p, 'Statistic', 'mean');
    parse(p, varargin{:});

    freqs = freqs(:); times = times(:);
    [~, fi] = min(abs(freqs - targetFreq));
    actual = freqs(fi);
    series = Sxx(fi, :).';
    if p.Results.UsePower, series = series.^2; end

    feat = local_sanitize(local_standardize(series));
    thr = p.Results.MinThreshold;
    if isempty(thr), thr = local_autoThreshold(feat); end

    ipt = findchangepts(feat.', 'Statistic', p.Results.Statistic, ...
                        'MinThreshold', thr);
    ipt = ipt(ipt > 1 & ipt < numel(times));

    out = struct();
    out.changepoints = ipt(:).';
    out.changepoint_times = times(ipt).';
    out.series = series;
    out.actual_freq = actual;
    out.times = times;
end

function z = local_standardize(a)
    z = (a - mean(a, 1, 'omitnan')) ./ (std(a, 0, 1, 'omitnan') + 1e-12);
end

function z = local_sanitize(z)
    % findchangepts rejects NaN/Inf (e.g. cone-of-influence-masked WT edges).
    % After standardisation the mean is 0, so filling non-finite entries with 0
    % is a neutral choice that won't create a spurious changepoint.
    z(~isfinite(z)) = 0;
end

function thr = local_autoThreshold(feat)
    % BIC-like floor scaled by variability (mirrors the Python _auto_pen intent;
    % findchangepts MinThreshold is on the residual-energy scale, so this is a
    % behavioural, not numeric, match).
    [T, d] = size(feat);
    base = log(max(T, 2)) * d;
    variability = mean(std(feat, 0, 1), 'omitnan');
    thr = base * (1 + variability);
end
