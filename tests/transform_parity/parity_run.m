function parity_run(repoRoot, refDir, variantDir, label, idPattern)
%PARITY_RUN  Run baseline and production on each case and compare in-process.
%   Nothing large is written to disk: each pair of transforms is compared
%   immediately and only scalar statistics are retained.
%
%   variantDir: '' to test the working-tree wt/wft, or a directory holding
%   forced-block-size copies to shadow them (blk=1 / blk=7 runs).

harnessDir = fileparts(mfilename('fullpath'));
addpath(harnessDir);
addpath(genpath(fullfile(repoRoot,'allguis')));
addpath(refDir);                       % wt_base / wft_base (pre-vectorisation)
if ~isempty(variantDir)
    addpath(variantDir,'-begin');      % shadow production wt/wft
end

fprintf('--- %s ---\n', label);
fprintf('wt      -> %s\n', which('wt'));
fprintf('wft     -> %s\n', which('wft'));
fprintf('wt_base -> %s\n', which('wt_base'));

C = build_cases();
if nargin >= 5 && ~isempty(idPattern)
    % Subset for the forced-block runs: the blocking loop is exercised by
    % kernel variety, not by the full option cross, and blk=1 is far too
    % slow to run all 966.
    keep = ~cellfun(@isempty, regexp({C.id}, idPattern, 'once'));
    C = C(keep);
    fprintf('subset: %d cases matching %s\n', numel(C), idPattern);
end
n = numel(C);

matched=0; mismatched=0; botherr=0; onlyerr=0; nanmismatch=0;
worstRel=0; worstId=''; fails={};

for i=1:n
    sig = make_signal(C(i).L);

    [okA, WA, errA] = tryrun(C(i).fn, true,  sig, C(i).args);
    [okB, WB, errB] = tryrun(C(i).fn, false, sig, C(i).args);

    if ~okA && ~okB
        if strcmp(errA, errB)
            botherr=botherr+1; matched=matched+1;
        else
            mismatched=mismatched+1;
            fails{end+1}=sprintf('%s | both errored, different msg:\n      base: %s\n      prod: %s', C(i).id, errA, errB); %#ok<AGROW>
        end
        continue
    end
    if okA ~= okB
        onlyerr=onlyerr+1; mismatched=mismatched+1;
        fails{end+1}=sprintf('%s | ok mismatch (base=%d prod=%d) base_err=%s prod_err=%s', C(i).id, okA, okB, errA, errB); %#ok<AGROW>
        continue
    end
    if ~isequal(size(WA),size(WB))
        mismatched=mismatched+1;
        fails{end+1}=sprintf('%s | size %s vs %s', C(i).id, mat2str(size(WA)), mat2str(size(WB))); %#ok<AGROW>
        continue
    end

    na=isnan(WA); nb=isnan(WB);
    if ~isequal(na,nb)
        nanmismatch=nanmismatch+1; mismatched=mismatched+1;
        fails{end+1}=sprintf('%s | NaN pattern differs (%d vs %d)', C(i).id, nnz(na), nnz(nb)); %#ok<AGROW>
        continue
    end

    Xv=WA(~na); Yv=WB(~nb);
    if isempty(Xv), matched=matched+1; continue, end
    scale=max(abs(Xv(:))); if scale==0, scale=1; end
    rel=max(abs(Xv-Yv))/scale;               % normalised by transform magnitude
    if rel>worstRel, worstRel=rel; worstId=C(i).id; end
    if rel<=1e-12
        matched=matched+1;
    else
        mismatched=mismatched+1;
        fails{end+1}=sprintf('%s | rel err %.3e', C(i).id, rel); %#ok<AGROW>
    end

    if mod(i,100)==0, fprintf('  %d/%d\n', i, n); end
end

fprintf('\n===== %s =====\n', label);
fprintf('cases            : %d\n', n);
fprintf('matched          : %d\n', matched);
fprintf('mismatched       : %d\n', mismatched);
fprintf('  NaN pattern    : %d\n', nanmismatch);
fprintf('  ok/err         : %d\n', onlyerr);
fprintf('both errored     : %d (identical message, counted as matched)\n', botherr);
fprintf('worst rel error  : %.3e\n', worstRel);
fprintf('worst case       : %s\n', worstId);
if ~isempty(fails)
    fprintf('\n-- failures (up to 30 of %d) --\n', numel(fails));
    for i=1:min(30,numel(fails)), fprintf('  %s\n', fails{i}); end
end
fprintf('===== end %s =====\n', label);
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
