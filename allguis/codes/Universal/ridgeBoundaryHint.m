function h = ridgeBoundaryHint(ifreq, fmin, fmax, varargin)
%RIDGEBOUNDARYHINT  Flag when a detected ridge hugs the frequency band edges.
%   Mirrors fastmoda.ridge_gpu.ridge_boundary_hint. Frequency edge only.
%
%   h = ridgeBoundaryHint(ifreq, fmin, fmax, Name, Value, ...)
%     ifreq : instantaneous ridge frequency over time (NaN allowed)
%     Name-Value: 'Tol' (0.05), 'LowThr' (0.08), 'HighThr' (0.25)
%
%   h = struct('level','none'|'low'|'high', 'edge','upper'|'lower'|'',
%              'frac',double, 'message',char).

    p = inputParser;
    addParameter(p, 'Tol', 0.05);
    addParameter(p, 'LowThr', 0.08);
    addParameter(p, 'HighThr', 0.25);
    parse(p, varargin{:});
    tol = p.Results.Tol; lowThr = p.Results.LowThr; highThr = p.Results.HighThr;

    ifreq = ifreq(:);
    valid = isfinite(ifreq) & ifreq > 0;
    h = struct('level', 'none', 'edge', '', 'frac', 0, 'message', '');
    if ~any(valid) || fmax <= fmin, return; end

    f = ifreq(valid);
    logspan = log(fmax) - log(fmin);
    dTop = (log(fmax) - log(f)) / logspan;   % 0 at fmax
    dBot = (log(f) - log(fmin)) / logspan;   % 0 at fmin
    fracTop = mean(dTop < tol);
    fracBot = mean(dBot < tol);

    if fracTop >= fracBot
        edge = 'upper'; frac = fracTop; which = 'fmax'; act = 'raising fmax'; edgeHz = fmax;
    else
        edge = 'lower'; frac = fracBot; which = 'fmin'; act = 'lowering fmin'; edgeHz = fmin;
    end

    if frac >= highThr
        level = 'high';
    elseif frac >= lowThr
        level = 'low';
    else
        level = 'none';
    end

    h.frac = frac;
    if strcmp(level, 'none'), return; end
    h.level = level; h.edge = edge;
    h.message = sprintf(['Ridge sits within %d%% of %s (%g Hz) for %.0f%% of its ' ...
        'length — the true ridge may extend beyond the analysed band; consider %s.'], ...
        round(tol*100), which, edgeHz, frac*100, act);
end
