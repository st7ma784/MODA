function [cpl1,cpl2,drc]=dirc(c,bn)
%calculates the net couplings as norms from the relevant inferred parameters

%---inputs---
%c - vector of inferred parameters
%bn - order of Fourier base function

%---outputs---
%cpl1 - coupling from second to first oscillator
%cpl2 - coupling from first to second oscillator
%drc  - direction of coupling drc~[-1,1]

%Note that the input is vector of parameters for one time window - for all
%time windows use 'dirc.m' in loop; see e.g. 'example2_CplPncrPrm.m'

%% ------------------------------------------------------------------------
K=length(c)/2;

% The original triple-nested (ii / ii / ii,jj) loop just walks c(2:K) and
% c(K+2:2*K) in strictly increasing, non-skipping order into q1/q2 — bn
% only determines how many consecutive entries are consumed (which is
% always the full 2:K / K+2:2*K range), never their order. So this is
% exactly a slice, independent of bn.
q1 = c(2:K);
q2 = c(K+2:2*K);

cpl1=norm(q1);
cpl2=norm(q2);
drc=(cpl2-cpl1)/(cpl1+cpl2);
