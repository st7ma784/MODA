%----------------Time-localized wavelet phase coherence--------------------
% TPC = tlphcoh(WT1,WT2,freq,fs,Optional:numcycles)
% calculates time-localized wavelet phase coherence TPC.
%
% Input:
% WT1,WT2 - wavelet transforms of two signals
% freq - frequencies used in wavelet transform
% fs - sampling frequency of a signals from which WT1, WT2 were calculated
% numcycles - number of cycles for calculating TPC (determines adaptive
%             window length, i.e. at 0.1 Hz it will be (1/0.1)*numcycles
%             seconds); default=10.
%
% Author: Dmytro Iatsenko (http://www.physics.lancs.ac.uk/research/nbmphysics/diats)
%--------------------------------------------------------------------------


function TPC = tlphcoh(TFR1,TFR2,freq,fs,varargin)

[NF,L]=size(TFR1);
if nargin>4, wsize=varargin{1}; else wsize=10; end

% Instantaneous phase-coherence "unit vector" at each (frequency,time): the
% phase difference between the two signals' wavelet transforms, expressed
% as a complex number of magnitude 1. Averaging THIS (not the phase angle
% itself) over a window and taking its magnitude is the standard circular
% mean used to measure phase-locking: perfectly phase-locked oscillations
% keep pointing the same direction and average to magnitude ~1, while
% randomly-drifting phase differences point every-which-way and average to
% magnitude ~0.
IPC=exp(1i*angle(TFR1.*conj(TFR2)));

% A moving-window mean of IPC could be computed by summing a fresh window
% of `window` values at every time point (cost O(L*window) per frequency),
% but a cumulative sum lets each window's total be read off as a single
% subtraction instead: sum(cs(a:b)) = cumcs(b+1)-cumcs(a). This makes the
% whole per-frequency sliding-window pass O(L) regardless of window size.
ZPC=IPC; ZPC(isnan(ZPC))=0; cumPC=[zeros(NF,1),cumsum(ZPC,2)];
TPC=zeros(NF,L)*NaN;
for fn=1:NF
    cs=IPC(fn,:); cumcs=cumPC(fn,:);
    tn1=find(~isnan(cs),1,'first'); tn2=find(~isnan(cs),1,'last');

    % Window length is frequency-adaptive: `numcycles` (wsize) full cycles
    % of THIS frequency, in samples, forced to be odd so the window has a
    % well-defined centre sample (hw = half-window) to assign the result to.
    window=round((wsize/freq(fn))*fs); window=window+1-mod(window,2); hw=floor(window/2);

    if ~isempty(tn1+tn2) && window<=tn2-tn1
    % locpc(t) = |mean of IPC over the `window`-sample window centred at
    % t+hw| = |cumcs(t+window) - cumcs(t)| / window, evaluated for every
    % valid window start at once (not a loop over t); result is written to
    % the CENTRE of each window, hence the tn1+hw:tn2-hw output range.
    locpc=abs(cumcs(tn1+window:tn2+1)-cumcs(tn1:tn2-window+1))/window;
    TPC(fn,tn1+hw:tn2-hw)=locpc;
    end
end

end

