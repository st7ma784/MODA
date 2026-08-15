function [TPC_p, avg_p, surr_p, surrPC_p, freq_p, wopt_p] = coherencePairWorker( ...
    sig1, sig2, fs, fc, fmin, fmax, wtype, cutselect, ppselect, ns, stype_str, under_sample)
% COHERENCEPAIRWORKER  Pure, parfor-safe computation of one signal pair's
% wavelet phase coherence (time-localized + time-averaged) and, if
% requested, its surrogate distribution.
%
% Factored out of CoherenceMulti.doCoherenceCalc's main per-pair loop so it
% can be run in parallel across signal pairs via parfor: it touches only
% plain numeric inputs/outputs, never any UI or handle-class state, which
% is a requirement for parfor loop bodies (workers cannot access waitbars,
% app properties, or other graphics/handle objects).
%
% INPUT:
% sig1, sig2:  the two signals forming this pair (row vectors)
% fs:          sampling frequency
% fc, fmin, fmax, wtype, cutselect, ppselect: wavelet transform parameters,
%              forwarded to wtwrapper exactly as in the serial version
% ns:          number of surrogates (<=1 means "don't compute surrogates")
% stype_str:   surrogate method name, forwarded to surrcalc
% under_sample: downsampling factor applied to the time-localized result,
%              matching the serial version's TPC(:, 1:under_sample:end)
%
% OUTPUT:
% TPC_p:     time-localized phase coherence for this pair (already downsampled)
% avg_p:     time-averaged phase coherence for this pair
% surr_p:    ns x N surrogate signals generated from sig2 ([] if ns<=1)
% surrPC_p:  ns x 1 cell of time-averaged coherence between sig1 and each
%            surrogate (ns x 1 cell of [] if ns<=1)
% freq_p:    frequency array from this pair's transform (identical across
%            pairs in practice, since fmin/fmax/fc/wtype are shared)
% wopt_p:    transform options struct from this pair's transform

[wt_1, freq_p, wopt_p] = wtwrapper(sig1, fs, fc, fmin, fmax, 1, wtype, cutselect, ppselect);
[wt_2, ~, ~] = wtwrapper(sig2, fs, fc, fmin, fmax, 1, wtype, cutselect, ppselect);

TPC_full = tlphcoh(wt_1, wt_2, freq_p, fs);
TPC_p = TPC_full(:, 1:under_sample:end);
avg_p = wphcoh(wt_1, wt_2);

surr_p = [];
surrPC_p = cell(ns, 1);
if ns > 1
    surr_p = surrcalc(sig2, ns, stype_str, 0, fs);
    for k = 1:ns
        [WT_s, ~, ~] = wtwrapper(surr_p(k,:), fs, fc, fmin, fmax, 1, wtype, cutselect, ppselect);
        surrPC_p{k} = wphcoh(wt_1, WT_s);
    end
end
end
