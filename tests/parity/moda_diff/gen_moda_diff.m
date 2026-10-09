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

% ridge_extraction.m: ecurve -> rectfr('direct'). The reconstruction
% constants (C, D or omg) are saved so the port can be driven by MODA's own
% values; the transform is MODA's too, so a difference is ecurve/rectfr's.
saveridge(outDir,'ridge',W1,freq,wopt,fs);
% Morlet: D is infinite, so rectfr takes its hybrid (phase-derivative) branch
[Wm,fm,wom] = wt(sig1,fs,common{:},'fmin',0.3,'fmax',8,'Wavelet','Morlet','f0',1,'CutEdges','off');
saveridge(outDir,'ridge_morlet',Wm,fm,wom,fs);
% CutEdges on: NaN cells outside the cone of influence
[Wc,fc,woc] = wt(sig1,fs,common{:},'fmin',0.3,'fmax',8,'Wavelet','Lognorm','f0',1,'CutEdges','on');
saveridge(outDir,'ridge_cut',Wc,fc,woc,fs);
% WFT: linear frequency axis, rectfr's other branch
[Wf,ff,wof] = wft(sig1,fs,common{:},'fmin',0.3,'fmax',8,'Window','Gaussian','f0',1,'CutEdges','off');
saveridge(outDir,'ridge_wft',Wf,ff,wof,fs);

% coherence: wphcoh (time-averaged) + tlphcoh (time-localised, 10 cycles)
[phcoh,phdiff] = wphcoh(W1,W2);
TPC = tlphcoh(W1,W2,freq,fs,10);
kind='coherence'; save(fullfile(outDir,'coherence.mat'),'kind','sig1','sig2','fs', ...
     'W1','W2','freq','phcoh','phdiff','TPC','-v7');

% bispectrum (bispecWavNew calls wt internally). Saved with the transforms,
% the padding wt produced and the wavelet's support, so the port can be run
% both end to end and from MODA's own padded signal.
savebisp(outDir,'bispectrum',sig1,sig2,fs,args,1,'predictive','off');          % 122, MODA default padding
zargs = [args, {'Padding',0}];
savebisp(outDir,'bispectrum_zero',sig1,sig2,fs,zargs,1,'zero','off');          % 122, zero padding
savebisp(outDir,'bispectrum_auto',sig1,sig1,fs,zargs,1,'zero','off');          % 111, upper triangle only
gargs = [common, {'fmin',0.3,'fmax',8,'f0',1,'Padding',0}];                     % as the GUI calls it: CutEdges on
savebisp(outDir,'bispectrum_cut',sig1,sig2,fs,gargs,1,'zero','on');

fprintf('wrote %d transform cases + ridge/coherence/bispectrum to %s\n', n, outDir);
end

function saveridge(outDir,name,W1,freq,wopt,fs)
tfsupp = ecurve(W1,freq,wopt,'Display','off');
[iamp,iphi,ifreq] = rectfr(tfsupp,W1,freq,wopt,'direct');
kind='ridge'; C=wopt.wp.C; ompeak=wopt.wp.ompeak; %#ok<NASGU>
if isfield(wopt.wp,'D'), D=wopt.wp.D; omg=NaN; else, D=NaN; omg=wopt.wp.omg; end %#ok<NASGU>
save(fullfile(outDir,[name '.mat']),'kind','fs','W1','freq','tfsupp','iamp','iphi', ...
     'ifreq','C','D','omg','ompeak','-v7');
end

function savebisp(outDir,name,sa,sb,fs,args,f0,pad,cut)
[Bisp,bfreq,opt,WT1,WT2] = bispecWavNew(sa,sb,fs,args{:});
kind='bispectrum'; sig1=sa; sig2=sb; %#ok<NASGU>
padleft=opt.PadLR{1}; padright=opt.PadLR{2}; wp=opt.wp; %#ok<NASGU>
t1e=wp.t1e; t2e=wp.t2e; t1h=wp.t1h; t2h=wp.t2h; ompeak=wp.ompeak; xi1=wp.xi1; xi2=wp.xi2; %#ok<NASGU>
fmin=opt.fmin; fmax=opt.fmax; pre=opt.Preprocess; %#ok<NASGU>
save(fullfile(outDir,[name '.mat']),'kind','sig1','sig2','fs','Bisp','bfreq','WT1','WT2', ...
     'padleft','padright','t1e','t2e','t1h','t2h','ompeak','xi1','xi2','fmin','fmax', ...
     'pre','f0','pad','cut','-v7');
end

function savecase(outDir,n,fn,sig,fs,WT,freq,nv,kernel,f0,pre,pad,cut,fmin,fmax)
kind = fn; %#ok<NASGU>
save(fullfile(outDir,sprintf('%s_%03d.mat',fn,n)),'kind','sig','fs','WT','freq','nv', ...
     'kernel','f0','pre','pad','cut','fmin','fmax','-v7');
end
