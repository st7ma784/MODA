function diag_special(repoRoot, refDir)
%DIAG_SPECIAL  Report the outcome of each non-default-path case individually.

harnessDir = fileparts(mfilename('fullpath'));
addpath(harnessDir);
addpath(genpath(fullfile(repoRoot,'allguis')));
addpath(refDir);

C = build_cases();
% the non-default-path cases are the ones whose kernel is not a plain name
sel = find(~ismember({C.kernel}, ...
      {'Lognorm','Morlet','Bump','Morse-3','Gaussian','Hann','Blackman','Exp','Rect','Kaiser-3'}));

fprintf('\n===== NON-DEFAULT PATH CASES =====\n');
for k = sel
    sig = make_signal(C(k).L);
    [okA,WA,errA] = tryrun(C(k).fn,true, sig,C(k).args);
    [okB,WB,errB] = tryrun(C(k).fn,false,sig,C(k).args);

    fprintf('\n%s\n', C(k).id);
    fprintf('  baseline  : %s\n', status(okA,WA,errA));
    fprintf('  production: %s\n', status(okB,WB,errB));
    if okA && okB
        na=isnan(WA); nb=isnan(WB);
        if ~isequal(size(WA),size(WB))
            fprintf('  VERDICT   : SIZE MISMATCH\n');
        elseif ~isequal(na,nb)
            fprintf('  VERDICT   : NaN PATTERN MISMATCH (%d vs %d)\n', nnz(na), nnz(nb));
        else
            Xv=WA(~na); Yv=WB(~nb);
            if isempty(Xv)
                fprintf('  VERDICT   : all-NaN both sides\n');
            else
                sc=max(abs(Xv(:))); if sc==0, sc=1; end
                fprintf('  VERDICT   : MATCH, rel err %.3e\n', max(abs(Xv-Yv))/sc);
            end
        end
    elseif ~okA && ~okB
        if strcmp(errA,errB)
            fprintf('  VERDICT   : both error identically (consistent)\n');
        else
            fprintf('  VERDICT   : DIFFERENT ERRORS\n');
        end
    else
        fprintf('  VERDICT   : ONE SIDE FAILED\n');
    end
end
fprintf('\n===== end =====\n');
end

function s = status(ok,W,err)
if ok, s = sprintf('ok, size %s', mat2str(size(W)));
else,  s = sprintf('ERROR %s', err); end
end

function [ok,W,err] = tryrun(fn, useBase, sig, args)
ok=true; W=[]; err='';
try
    switch [fn '|' num2str(useBase)]
        case 'wt|1',  W = wt_base(sig,40,args{:});
        case 'wt|0',  W = wt(sig,40,args{:});
        case 'wft|1', W = wft_base(sig,40,args{:});
        case 'wft|0', W = wft(sig,40,args{:});
    end
catch ME
    ok=false; err=sprintf('%s: %s', ME.identifier, ME.message);
end
end

function sig = make_signal(L)
rng(12345,'twister');
t = (0:L-1)'/40;
sig = sin(2*pi*(0.5 + 0.02*t).*t) + 0.5*sin(2*pi*3*t) + 0.1*t + 0.05*randn(L,1);
end
