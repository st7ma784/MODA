function gen_moda_other(repoRoot, outDir)
%GEN_MODA_OTHER  MODA references for the modules beyond the transforms.
%
%   Bayesian inference (full_bayesian's chain: loop_butter -> hilbert ->
%   bayes_main -> dirc / CFprint), both of MODA's loop_butter variants, the
%   Butterworth design itself, hilbert, and biphaseWavNew. Each stage is
%   saved with its own input, so FastMODA's port of every stage can be fed
%   MODA's input for it and compared on its own.
%
%   Run:  matlab -batch "gen_moda_other('<repo>', '<repo>/tests/parity/moda_diff/reference')"

addpath(genpath(fullfile(repoRoot,'allguis')));
if ~exist(outDir,'dir'), mkdir(outDir); end

% MODA ships two different functions called loop_butter. Which one a call
% by name reaches depends on the path; record it, and make both callable
% under their own names for the comparisons below.
loopButterUsed = which('loop_butter');
tmp = tempname; mkdir(tmp); addpath(tmp);
renameCopy(fullfile(repoRoot,'allguis','guis','bayesian','Functions','loop_butter.m'), ...
           fullfile(tmp,'loop_butter_bayes.m'), 'loop_butter', 'loop_butter_bayes');
renameCopy(fullfile(repoRoot,'allguis','guis','filtering','Functions','loop_butter.m'), ...
           fullfile(tmp,'loop_butter_filt.m'), 'loop_butter', 'loop_butter_filt');

% ---- Butterworth design: [b,a] = butter(n, Wn) --------------------------
Wns = [0.08 0.12; 0.11 0.15; 0.02 0.4; 0.3 0.6; 0.005 0.05];
orders = 1:10;
NB = numel(orders)*size(Wns,1);
BN = zeros(NB,1); BW = zeros(NB,2); BB = NaN(NB,21); BA = NaN(NB,21);
k = 0;
for iw = 1:size(Wns,1)
    for n = orders
        k = k + 1; [b,a] = butter(n, Wns(iw,:));
        BN(k) = n; BW(k,:) = Wns(iw,:); BB(k,1:numel(b)) = b; BA(k,1:numel(a)) = a;
    end
end
kind = 'butter'; save(fullfile(outDir,'other_butter.mat'),'kind','BN','BW','BB','BA');

% ---- Bayesian ----------------------------------------------------------
fs = 20; n = 6000; h = 1/fs;
rng(21);
p = zeros(n,2); p(1,:) = [0 1];
for k = 1:n-1
    d1 = 2*pi*1.0 + 0.3*sin(p(k,2)-p(k,1));
    d2 = 2*pi*1.3 + 0.9*sin(p(k,1)-p(k,2));
    p(k+1,:) = p(k,:) + h*[d1 d2] + sqrt(h)*0.15*randn(1,2);
end
x1 = cos(p(:,1)) + 0.1*randn(n,1); x2 = cos(p(:,2)) + 0.1*randn(n,1);
int1 = [0.8 1.2]; int2 = [1.1 1.5];
win = 50; pr = 0.2; ovr = 0.5;

[bands1, or1] = loop_butter_bayes(x1, int1, fs);
[bands2, or2] = loop_butter_bayes(x2, int2, fs);
sens1 = ulpSensitivity(x1, int1, fs, or1); sens2 = ulpSensitivity(x2, int2, fs, or2);
phi1 = angle(hilbert(bands1)); phi2 = angle(hilbert(bands2));
for bn = 1:2
    [tm, cc, e] = bayes_main(phi1, phi2, win, h, ovr, pr, 0, bn);
    nw = size(cc,1); ng = numel(0:0.13:2*pi);
    cpl1 = zeros(1,nw); cpl2 = zeros(1,nw); drc = zeros(1,nw);
    q21 = zeros(ng,ng,nw); q12 = zeros(ng,ng,nw);
    for m = 1:nw
        [cpl1(m),cpl2(m),drc(m)] = dirc(cc(m,:),bn);
        [~,~,q21(:,:,m),q12(:,:,m)] = CFprint(cc(m,:),bn);
    end
    % the whole chain as MODA's own entry point runs it (one surrogate,
    % which is random; only the deterministic outputs are kept)
    [fb_tm,fb_p1,fb_p2,fb_cpl1,fb_cpl2,fb_cf1,fb_cf2,fb_mcf1,fb_mcf2] = ...
        full_bayesian(x1, x2, int1(1), int1(2), int2(1), int2(2), fs, win, pr, ovr, bn, 1, 95);
    kind = 'bayes';
    save(fullfile(outDir,sprintf('other_bayes_bn%d.mat',bn)),'kind','x1','x2','fs','int1','int2', ...
        'win','pr','ovr','bn','bands1','bands2','or1','or2','phi1','phi2','tm','cc','e', ...
        'cpl1','cpl2','drc','q21','q12','fb_tm','fb_p1','fb_p2','fb_cpl1','fb_cpl2', ...
        'fb_mcf1','fb_mcf2','loopButterUsed','sens1','sens2');
end

% ---- Filtering module's loop_butter ------------------------------------
fsF = 40; tF = (0:4095)'/fsF; rng(4);
xF = cos(2*pi*1.0*tF) + 0.5*cos(2*pi*3.0*tF) + 0.3*cos(2*pi*7.5*tF) + 0.2*randn(size(tF));
bandsF = [0.7 1.3; 2.5 3.5; 0.2 5; 6 9];
outF = zeros(numel(xF), size(bandsF,1)); orF = zeros(1,size(bandsF,1));
outB = outF; orB = orF;
sensF = orF; sensB = orF;
for ib = 1:size(bandsF,1)
    [outF(:,ib), orF(ib)] = loop_butter_filt(xF, bandsF(ib,:), fsF);
    [outB(:,ib), orB(ib)] = loop_butter_bayes(xF, bandsF(ib,:), fsF);
    % MATLAB's own sensitivity at the order it chose: how far the output
    % moves when one coefficient changes by one unit in the last place
    sensF(ib) = ulpSensitivity(xF, bandsF(ib,:), fsF, orF(ib));
    sensB(ib) = ulpSensitivity(xF, bandsF(ib,:), fsF, orB(ib));
end
kind = 'loopbutter';
save(fullfile(outDir,'other_loopbutter.mat'),'kind','xF','fsF','bandsF','outF','orF','outB','orB', ...
    'sensF','sensB');

% ---- hilbert ------------------------------------------------------------
xH = [xF; xF(1:777)];             % odd length too
H1 = hilbert(xF); H2 = hilbert(xH);
kind = 'hilbert'; save(fullfile(outDir,'other_hilbert.mat'),'kind','xF','xH','H1','H2');

% ---- biphaseWavNew ------------------------------------------------------
fsP = 40; LP = 1600; tP = (0:LP-1)/fsP; rng(8);
s1 = cos(2*pi*1.1*tP) + 0.7*cos(2*pi*2.4*tP + 0.3) + 0.1*randn(1,LP);
s2 = cos(2*pi*3.5*tP + 0.9) + cos(2*pi*1.1*tP) + 0.1*randn(1,LP);
args = {'fmin',0.3,'fmax',8,'f0',1};
[~,~,popt] = bispecWavNew(s1, s2, fsP, args{:});   % app.wavopt, as the GUI keeps it
fpair = [1.1 2.4];
[biamp, biphase] = biphaseWavNew(s1, s2, fsP, fpair, popt);
[biampA, biphaseA] = biphaseWavNew(s1, s1, fsP, fpair, popt);
kind = 'biphase';
save(fullfile(outDir,'other_biphase.mat'),'kind','s1','s2','fsP','fpair','biamp','biphase', ...
    'biampA','biphaseA');

rmpath(tmp);
fprintf('loop_butter by name resolves to: %s\n', loopButterUsed);
end

function s = ulpSensitivity(x, band, fs, n)
% Relative change in filtfilt's output when b(1) moves by one ulp
[b,a] = butter(n, band/(fs/2));
y0 = filtfilt(b, a, x);
b(1) = b(1)*(1+eps);
s = max(abs(filtfilt(b, a, x) - y0)) / max(abs(y0));
end

function renameCopy(src, dst, oldName, newName)
txt = fileread(src);
txt = regexprep(txt, ['(function\s*\[[^\]]*\]\s*=\s*)' oldName '(\s*\()'], ['$1' newName '$2'], 'once');
fid = fopen(dst,'w'); fwrite(fid, txt); fclose(fid);
end
