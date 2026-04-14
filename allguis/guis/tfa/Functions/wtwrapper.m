function [WT, freqarr, wopt] = wtwrapper(sig, fs, fc, fmin, fmax, calc_type, wtype, cutselect, ppselect)
% WTWRAPPER  Compute wavelet or windowed Fourier transform.
%
%   [WT, freqarr, wopt] = wtwrapper(sig, fs, fc, fmin, fmax, calc_type,
%                                    wtype, cutselect, ppselect)
%
%   Inputs
%     sig        - 1×N signal row vector
%     fs         - sampling frequency (Hz)
%     fc         - central frequency / resolution parameter
%     fmin       - minimum frequency (NaN = auto)
%     fmax       - maximum frequency (NaN = auto)
%     calc_type  - 1 = wavelet transform, 2 = windowed Fourier transform
%     wtype      - wavelet / window type string
%     cutselect  - 'CutEdges' option value
%     ppselect   - 'Preprocess' option value
%
%   Outputs
%     WT         - time–frequency representation matrix
%     freqarr    - frequency array
%     wopt       - transform options struct
%
% Converted from GUIDE script to standalone function.
% Compatible with MATLAB R2023a through R2026a.

if calc_type == 1
    tfun = @wt;
    typekey = 'Wavelet';
else
    tfun = @wft;
    typekey = 'Window';
end

base_args = {sig, fs, 'CutEdges', cutselect, 'Preprocess', ppselect, typekey, wtype, 'Padding', 0};

fmin_ok = ~isnan(fmin);
fmax_ok = ~isnan(fmax);
fc_ok   = ~isnan(fc);

if ~fmin_ok && ~fmax_ok
    if ~fc_ok
        [WT, freqarr, wopt] = tfun(base_args{:});
    else
        [WT, freqarr, wopt] = tfun(base_args{:}, 'f0', fc);
    end
elseif ~fmax_ok
    if ~fc_ok
        [WT, freqarr, wopt] = tfun(base_args{:}, 'fmin', fmin);
    else
        [WT, freqarr, wopt] = tfun(base_args{:}, 'fmin', fmin, 'f0', fc);
    end
elseif ~fmin_ok
    if ~fc_ok
        [WT, freqarr, wopt] = tfun(base_args{:}, 'fmax', fmax);
    else
        [WT, freqarr, wopt] = tfun(base_args{:}, 'fmax', fmax, 'f0', fc);
    end
else
    if ~fc_ok
        [WT, freqarr, wopt] = tfun(base_args{:}, 'fmin', fmin, 'fmax', fmax);
    else
        [WT, freqarr, wopt] = tfun(base_args{:}, 'fmin', fmin, 'fmax', fmax, 'f0', fc);
    end
end
