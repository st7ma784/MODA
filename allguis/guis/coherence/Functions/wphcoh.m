% [phcoh,Optional:phdiff] = wphcoh(WT1,WT2)
% returns time-averaged wavelet phase coherence between two signals;
% WT1 and WT2 are wavelet transforms of these signals.
%
% Author: Dmytro Iatsenko (http://www.physics.lancs.ac.uk/research/nbmphysics/diats)
%--------------------------------------------------------------------------

function [phcoh,varargout] = wphcoh(WT1,WT2)

FN=min([size(WT1,1),size(WT2,1)]);
WT1=WT1(1:FN,:); WT2=WT2(1:FN,:);
phi1=angle(WT1); phi2=angle(WT2);
phexp=exp(1i*(phi1-phi2));

% Per-row (per-frequency) reduction, computed for all rows at once instead
% of looping: CL = count of non-NaN phexp entries in the row, NL = count of
% positions where both WT1 and WT2 are exactly zero. When CL==0, meanVal
% and NL./CL both evaluate via 0/0 or x/0 to produce NaN, matching the
% original loop's untouched NaN-initialized default for that row.
validMask = ~isnan(phexp);
CL = sum(validMask,2);
NL = sum(WT1==0 & WT2==0, 2);

phexpValid = phexp;
phexpValid(~validMask) = 0;
meanVal = sum(phexpValid,2) ./ CL;

phph = meanVal - NL./CL;
phcoh = abs(phph).';
phdiff = angle(phph).';

if nargout>1, varargout{1}=phdiff; end

end
