
function [tm,cc,e]=bayes_main(ph1,ph2,win,h,ovr,pr,s,bn)
%the main function for the inference, propagation and other evaluations
%
%----------------------------- Algorithm overview --------------------------
% This estimates a time-VARYING coupling function between two phase
% oscillators by sliding a window along the phase time-series and, in each
% window, fitting a coupled phase-oscillator model
%   dphi1/dt = f1(phi1,phi2) + noise,  dphi2/dt = f2(phi1,phi2) + noise
% where f1,f2 are expanded in a truncated 2D Fourier series of order `bn`
% (see CFprint.m for how the fitted coefficients turn back into f1,f2 on a
% grid, and dirc.m for how they collapse into single coupling-strength
% numbers). The fit itself (maximum-likelihood / Bayesian inference of the
% Fourier coefficients `Cpt` and their uncertainty `XIpt`, given one
% window's worth of phase data) is done by bayesPhs — treated as a black
% box here, since it is a separate, self-contained inner loop.
%
% What makes this "dynamical" rather than a single one-shot fit is the
% PROPAGATION step between windows: each window's posterior (Cpt, XIpt)
% becomes the PRIOR (Cpr, XIpr) for the NEXT window, but with its
% covariance deliberately inflated by Propagation_function_XIpt (below) —
% this is a random-walk assumption on the coupling parameters, letting the
% inferred coupling drift smoothly between windows instead of being
% re-estimated from scratch (which would be noisy) or frozen at the first
% window's estimate (which would miss genuine time-variation). This
% two-step "infer, then diffuse the belief forward" pattern is exactly the
% predict/update cycle of a Kalman-style recursive Bayesian filter, applied
% here to coupling-function coefficients instead of a physical state.

%---inputs---
%ph1,ph2 - phase time-series vectors
%win - window in seconds
%h - sampling step e.g. h=0.01
%ovr - overlaping of windows; ovr=1 is no overlap; ovr=0.75 will overlap
%      the last 1/4 of each window with the next window
%pr - propagation constant
%s - print progress status if s=1
%bn - order of Fourier base function

%---outputs---
%tm - time vector for plotting
%cc - inferred mean parameters
%e  - inferred noise

%example for default input parameters and call of
%the function >> [tm,cc,e]=bayes_main(ph1,ph2,40,0.01,1,0.2,1,2);
%%

win=win/h;      % window length converted from seconds to samples
w=ovr*win;      % step (in samples) between consecutive window starts; w<win means overlap
ps=ph2-ph1;     % only used below to get the number of usable windows (length(ps)==length(ph1))
pw=win*h*pr;    % propagation constant scaled to this window's actual duration in seconds

M=2+2*((2*bn+1)^2-1);
L=2;

Cpr=zeros(M/L,L);XIpr=zeros(M);


%unwrap the phases if they are not
if (max(ph1)<(2*pi+0.1))
    ph1=unwrap(ph1);
    ph2=unwrap(ph2);
end

%set the right dimensions for the vectors
[m,n]=size(ph1);
if m<n
    ph1=ph1';
    ph2=ph2';
end


%% do the main calculations for each window
% Preallocate cc/e at the known window count and parameter sizes (M from
% line 30, E is always L-by-L per bayesPhs) instead of growing them by
% indexing past the end on every iteration.
numWin=floor((length(ps)-win)/w)+1;
cc=zeros(numWin,M);
e=zeros(numWin,L,L);
for i=0:floor((length(ps)-win)/w)

    % i-th window: `win` samples starting `i*w` samples in (w<win => windows overlap)
    phi1=ph1(i*w+1:i*w+win); phi2=ph2(i*w+1:i*w+win);

    %-----bayesian inference for one window------
    % Cpr/XIpr going in are this window's PRIOR (from the previous
    % iteration's propagation step, or zeros on the very first window);
    % Cpt/XIpt coming out are this window's POSTERIOR mean and inverse-
    % covariance for the Fourier coefficients, E is the inferred noise.
    [Cpt,XIpt,E]=bayesPhs(Cpr,XIpr,h,500,0.00001,phi1',phi2',bn);


    %the propagation for the next window: carry this window's posterior
    %mean forward unchanged, but inflate its covariance (see function below)
    %so the next window's fit isn't overconfident about how much the
    %coupling could have drifted meanwhile.
    [XIpr,Cpr] = Propagation_function_XIpt(Cpt,XIpt,pw);
    
    
    
    e(i+1,:,:)=E;
    cc(i+1,:)=Cpt(:);
    
    %display progress
    if s
        display(['processed so far: t= ' num2str((i+1)*w*h) 's /' num2str(length(ph1)*h) 's ;']);
        %Cpt
        %E
    end
    
end

%time vector for plotting
tm = (win/2:w:length(ph1)-win/2)*h;
%%



function [ XIpr,Cpr ] = Propagation_function_XIpt(Cpt,XIpt,p)
% Propagation function with covariance
% find the new prior for the next block

% The average is not supposed to change
Cpr=Cpt;


% Prepare the diffusion matrix (diagonal, so built directly instead of
% via a loop that only ever touches Inv_Diffusion's own diagonal entries)
invXIpt=inv(XIpt);
Inv_Diffusion = diag(p*p*diag(invXIpt));

% The gaussian of the posterior is convoluted with another
% gaussian which express the diffusion of the parameter.
XIpr=inv(( invXIpt + Inv_Diffusion ));
%%

function [ XIpr,Cpr ] = Propagation_function_Cpt(Cpt,XIpt,p)
% Propagation function with parameters
% find the new prior for the next block

% The average is not supposed to change
Cpr=Cpt;


% Prepare the diffusion matrix (diagonal, so built directly instead of
% via a loop that only ever touches Inv_Diffusion's own diagonal entries)
invXIpt=inv(XIpt);
Inv_Diffusion = diag(p*p*Cpt(:));

% The gaussian of the posterior is convoluted with another
% gaussian which express the diffusion of the parameter.
XIpr=inv(( invXIpt + Inv_Diffusion ));
%%