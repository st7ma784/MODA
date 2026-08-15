function [t1,t2,q1,q2]=CFprint(cc,bn)
%plots the coupling functions from the inferred parameters

%---inputs---
%cc - vector of inferred parameters
%bn - order of Fourier base function

%Note that the input is vector of parameters for one time window
%%
%---evaluating the coupling functions -----
t1=0:0.13:2*pi;t2=0:0.13:2*pi;
u=cc; K=length(u)/2;

% Evaluated as matrices over the full (t1,t2) grid at once instead of a
% scalar loop per grid point — same term-by-term sum and coefficient
% indexing as before (br still walks u in the same order), just summed as
% 49x49 matrix ops instead of 2401 individual (i1,j1) iterations.
[T1,T2] = ndgrid(t1,t2);   % T1(i,j)=t1(i), T2(i,j)=t2(j), matching q1(i1,j1)
q1 = zeros(size(T1));
q2 = zeros(size(T1));

br=2;
for ii=1:bn
    q1 = q1 + u(br)*sin(ii*T1) + u(br+1)*cos(ii*T1);
    q2 = q2 + u(K+br)*sin(ii*T2) + u(K+br+1)*cos(ii*T2);
    br=br+2;
end
for ii=1:bn
    q1 = q1 + u(br)*sin(ii*T2) + u(br+1)*cos(ii*T2);
    q2 = q2 + u(K+br)*sin(ii*T1) + u(K+br+1)*cos(ii*T1);
    br=br+2;
end

for ii=1:bn
    for jj=1:bn
        q1 = q1 + u(br)*sin(ii*T1+jj*T2) + u(br+1)*cos(ii*T1+jj*T2);
        q2 = q2 + u(K+br)*sin(ii*T1+jj*T2) + u(K+br+1)*cos(ii*T1+jj*T2);
        br=br+2;

        q1 = q1 + u(br)*sin(ii*T1-jj*T2) + u(br+1)*cos(ii*T1-jj*T2);
        q2 = q2 + u(K+br)*sin(ii*T1-jj*T2) + u(K+br+1)*cos(ii*T1-jj*T2);
        br=br+2;
    end
end

%---plotting -----
%                         f1=figure;
%
%                         subplot(1,2,1);surf(t1,t2,q1','FaceColor','interp');
%                         view([-40 50])
%                         set(gca,'fontname','Helvetica','fontsize',12,'Xgrid','off','Ygrid','off')
%                         xlabel('\phi_1');ylabel('\phi_2');zlabel('q_1(\phi_1,\phi_2)');axis tight
%
%                         subplot(1,2,2);surf(t1,t2,q2','FaceColor','interp');
%                         view([-40 50])
%                         set(gca,'fontname','Helvetica','fontsize',12,'Xgrid','off','Ygrid','off')
%                         xlabel('\phi_1');ylabel('\phi_2');zlabel('q_2(\phi_1,\phi_2)');axis tight
%
%                         colormap(hot)
%                         set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.

%uncomment this lines for saving the figure
% saveas(f1,'filename','jpg');
% saveas(f1,'filename','fig');

%%
