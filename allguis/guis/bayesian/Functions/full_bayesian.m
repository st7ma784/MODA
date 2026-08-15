function [tm,p1,p2,cpl1,cpl2,cf1,cf2,mcf1,mcf2,surr_cpl1,surr_cpl2] = full_bayesian(sig1, sig2, int11, int12, int21, int22, fs, win, pr, ovr, bn, ns, signif)
% Created for use in PyMODA.

int1 = [int11, int12];
int2 = [int21, int22];

[bands1,~] = loop_butter(sig1(:),int1(:),fs);
phi1=angle(hilbert(bands1));

[bands2,~] = loop_butter(sig2(:),int2(:),fs);
phi2=angle(hilbert(bands2));

p1=phi1;
p2=phi2;

[tm, cc, e] = bayes_main(phi1, phi2, win, 1/fs, ovr, pr,0,bn);
% Preallocate at known sizes: size(cc,1) windows, and CFprint always
% returns a fixed numel(0:0.13:2*pi)-by-same grid (see CFprint.m), so the
% dimensions don't depend on loop content.
ng = numel(0:0.13:2*pi);
numWin = size(cc,1);
cpl1 = zeros(1,numWin);
cpl2 = zeros(1,numWin);
q21 = zeros(ng,ng,numWin);
q12 = zeros(ng,ng,numWin);
for m=1:numWin
    [cpl1(m),cpl2(m)]=dirc(cc(m,:),bn); % Direction of coupling
    [~,~,q21(:,:,m),q12(:,:,m)]=CFprint(cc(m,:),bn); % Coupling functions
end

cf1 = q21;
cf2 = q12;
mcf1 = squeeze(mean(q21,3));
mcf2 = squeeze(mean(q12,3));

surr1 = surrcalc(phi1',ns,'CPP',0,fs);
surr2 = surrcalc(phi2',ns,'CPP',0,fs);

% scpl1/scpl2 are the only values needed from each surrogate's bayes_main
% call; the full per-window coupling-coefficient matrix (formerly kept
% for every surrogate in cc_surr{n}) is never used after dirc() extracts
% these two numbers, so it's discarded each iteration instead of retained.
scpl1 = zeros(ns,numWin);
scpl2 = zeros(ns,numWin);
for n=1:ns
    [~,cc_s]=bayes_main(surr1(n,:),surr2(n,:),win,1/fs,ovr,pr,1,bn);
    for idx=1:size(cc_s,1)
        [scpl1(n,idx),scpl2(n,idx)]=dirc(cc_s(idx,:),bn);
    end
end

alph=signif;
alph=(1-(alph/100));
if floor((ns+1)*alph)==0
    surr_cpl1 = max(scpl1);
    surr_cpl2 = max(scpl2);
else
    K=floor((ns+1)*alph);
    s1=sort(scpl1,'descend');
    s2=sort(scpl2,'descend');
    surr_cpl1 = s1(K,:);
    surr_cpl2 = s2(K,:);
    
end

end

