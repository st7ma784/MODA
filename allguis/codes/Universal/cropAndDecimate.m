function [y, fsNew, info] = cropAndDecimate(x, fs, mode, varargin)
%CROPANDDECIMATE  Crop then integer-decimate a signal.
%   Mirrors fastmoda.preprocess.crop_and_decimate (kept at parity with the web
%   app). Decimation is integer-only (fs -> fs/k) and anti-aliased.
%
%   [y, fsNew, info] = cropAndDecimate(x, fs, mode, Name, Value, ...)
%     mode : 'none' | 'range' | 'first' | 'final'
%     Name-Value: 'StartS', 'StopS', 'LengthS', 'DecimateFactor' (int >= 1)
%
%   info is a struct with n_in/n_out, fs_in/fs_out, decimate_factor,
%   t_start/t_stop and dur_in/dur_out for UI display.

    p = inputParser;
    addParameter(p, 'StartS', []);
    addParameter(p, 'StopS', []);
    addParameter(p, 'LengthS', []);
    addParameter(p, 'DecimateFactor', 1);
    parse(p, varargin{:});

    x = x(:).';
    n = numel(x);
    fs = double(fs);

    [i0, i1] = local_cropIndices(n, fs, mode, p.Results.StartS, ...
                                 p.Results.StopS, p.Results.LengthS);
    y = x(i0+1:i1);                          % i0,i1 are 0-based half-open

    k = max(1, round(p.Results.DecimateFactor));
    y = local_decimate(y, k);
    fsNew = fs / k;

    info = struct('n_in', n, 'n_out', numel(y), 'fs_in', fs, 'fs_out', fsNew, ...
        'decimate_factor', k, 't_start', i0/fs, 't_stop', i1/fs, ...
        'dur_in', n/fs, 'dur_out', numel(y)/max(fsNew, eps));
end

function [i0, i1] = local_cropIndices(n, fs, mode, startS, stopS, lengthS)
    switch lower(mode)
        case 'range'
            if isempty(startS), i0 = 0; else, i0 = round(startS*fs); end
            if isempty(stopS),  i1 = n; else, i1 = round(stopS*fs);  end
        case 'first'
            i0 = 0; i1 = round(lengthS*fs);
        case 'final'
            i0 = n - round(lengthS*fs); i1 = n;
        otherwise
            i0 = 0; i1 = n;
    end
    i0 = max(0, min(i0, n));
    i1 = max(0, min(i1, n));
    if i1 <= i0
        error('MODA:crop:empty', ...
            ['Crop produces an empty signal — check the start/stop or length ' ...
             'values against the signal duration.']);
    end
end

function y = local_decimate(x, factor)
    factor = round(factor);
    if factor <= 1, y = double(x); return; end
    y = double(x);
    for f = local_factorize(factor)
        if f == 1 || numel(y) <= 27, break; end
        y = decimate(y, f);                  % anti-alias low-pass then downsample
    end
end

function fac = local_factorize(q)
    q = round(q); fac = [];
    for p = [2 3 5 7 11 13]
        while mod(q, p) == 0, fac(end+1) = p; q = q / p; end %#ok<AGROW>
    end
    if q > 1, fac(end+1) = q; end
    if isempty(fac), fac = 1; end
end
