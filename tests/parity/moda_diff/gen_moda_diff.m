function gen_moda_diff(repoRoot, outDir)
%GEN_MODA_DIFF  Reference outputs from real MODA for the MODA-vs-FastMODA diff.
%
%   Writes one v7 .mat per case into outDir. test_moda_diff.py runs FastMODA's
%   legacy path on exactly the same inputs and reports max|X-Y|/max|X| per case.
%
%   Downstream stages (ridge, coherence, bispectrum) are stored twice over:
%   the MODA output, and the MODA WT that produced it. Feeding that same WT to
%   FastMODA's downstream code isolates the downstream algorithm from the
%   transform, so a difference there is the algorithm's and not inherited.
%
%   Run:  matlab -batch "gen_moda_diff('<repo>', '<repo>/tests/parity/moda_diff/reference')"

addpath(genpath(fullfile(repoRoot,'allguis')));
if ~exist(outDir,'dir'), mkdir(outDir); end

fs = 40; L = 1024; t = (0:L-1)/fs;
rng(1);
% Two tones, a slow chirp and a little noise: something for every wavelet to
% resolve, plus a ridge that actually moves.
sig1 = cos(2*pi*1.0*t) + 0.6*cos(2*pi*3.3*t + 0.4) ...
     + 0.3*cos(2*pi*(0.4*t + 0.02*t.^2)) + 0.05*randn(1,L);
% Phase-coupled partner for coherence, with its own noise.
sig2 = 0.8*cos(2*pi*1.0*t + 0.7) + 0.5*cos(2*pi*3.3*t - 0.3) + 0.05*randn(1,L);

common = {'Display','off','Plot','off'};
n = 0;

% ---- WT: every wavelet x f0 x preprocess x padding x cut-edges ----------
wavelets = {'Lognorm','Morlet','Bump'};
pads     = {'predictive', 0, 'symmetric', 'periodic'};
padname  = {'predictive','zero','symmetric','periodic'};
onoff    = {'on','off'};
for iW=1:3, for f0=[1 2], for iP=1:2, for iD=1:4, for iC=1:2
    args = [common, {'fmin',0.3,'fmax',8,'Wavelet',wavelets{iW},'f0',f0, ...
            'Preprocess',onoff{iP},'Padding',pads{iD},'CutEdges',onoff{iC}}];
    [WT,freq,wopt] = wt(sig1,fs,args{:});
    n=n+1; savecase(outDir,n,'wt',sig1,fs,WT,freq,wopt.nv, ...
        wavelets{iW},f0,onoff{iP},padname{iD},onoff{iC},0.3,8);
end, end, end, end, end

% default band (fmin derived from the wavelet's support, fmax = Nyquist)
for iW=1:3
    [WT,freq,wopt] = wt(sig1,fs,common{:},'Wavelet',wavelets{iW},'f0',1);
    n=n+1; savecase(outDir,n,'wt',sig1,fs,WT,freq,wopt.nv, ...
        wavelets{iW},1,'on','predictive','on',NaN,NaN);
end

% ---- WFT ----------------------------------------------------------------
windows = {'Gaussian','Hann','Blackman','Exp','Rect','Kaiser-3'};
for iN=1:numel(windows), for iD=[1 2 3], for iC=1:2
    [WT,freq] = wft(sig1,fs,common{:},'fmin',0.3,'fmax',8,'Window',windows{iN}, ...
                    'f0',1,'Padding',pads{iD},'CutEdges',onoff{iC});
    n=n+1; savecase(outDir,n,'wft',sig1,fs,WT,freq,NaN, ...
        windows{iN},1,'on',padname{iD},onoff{iC},0.3,8);
end, end, end

% ---- Downstream, from one MODA WT pair (Lognorm f0=1, MODA defaults) ----
% CutEdges off: ecurve and the coherences handle the edges themselves, and
% this is how ridge_extraction.m / the coherence GUI call wt.
args = [common, {'fmin',0.3,'fmax',8,'Wavelet','Lognorm','f0',1,'CutEdges','off'}];
[W1,freq,wopt] = wt(sig1,fs,args{:});
[W2,~,~]       = wt(sig2,fs,args{:});

% ridge_extraction.m: ecurve -> rectfr('direct')
tfsupp = ecurve(W1,freq,wopt);
[iamp,iphi,ifreq] = rectfr(tfsupp,W1,freq,wopt,'direct');
kind='ridge'; save(fullfile(outDir,'ridge.mat'),'kind','sig1','fs','W1','freq', ...
     'tfsupp','iamp','iphi','ifreq','-v7');

% coherence: wphcoh (time-averaged) + tlphcoh (time-localised, 10 cycles)
[phcoh,phdiff] = wphcoh(W1,W2);
TPC = tlphcoh(W1,W2,freq,fs,10);
kind='coherence'; save(fullfile(outDir,'coherence.mat'),'kind','sig1','sig2','fs', ...
     'W1','W2','freq','phcoh','phdiff','TPC','-v7');

% bispectrum 122 at the same settings (bispecWavNew calls wt internally)
[Bisp,bfreq] = bispecWavNew(sig1,sig2,fs,args{:});
kind='bispectrum'; save(fullfile(outDir,'bispectrum.mat'),'kind','sig1','sig2','fs', ...
     'Bisp','bfreq','-v7');

fprintf('wrote %d transform cases + ridge/coherence/bispectrum to %s\n', n, outDir);
end

function savecase(outDir,n,fn,sig,fs,WT,freq,nv,kernel,f0,pre,pad,cut,fmin,fmax)
kind = fn; %#ok<NASGU>
save(fullfile(outDir,sprintf('%s_%03d.mat',fn,n)),'kind','sig','fs','WT','freq','nv', ...
     'kernel','f0','pre','pad','cut','fmin','fmax','-v7');
end
