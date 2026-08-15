% update for MODA 20.07.2018
% code compatible with Dmytro Iatsenko's wt.m
% author: Aleksandra Pidde a.pidde@gmail.com, a.pidde@lancaster.ac.uk

function [Bisp, freq, opt, wt1, wt2] = bispecWavNew(sig1, sig2, fs, varargin)
% function calculating wavelet bispectrum
%
% INPUT:
% sig1, sig2: 	signals
% fs: 			sampling frequency
% optional:
% opt:			structure of optimal parameters returned by wt.m
%
% OUTPUT:
% Bisp: 		bispectrum, 2 dim matrix of complex values
% freq: 		frequencies
% opt:			structure of optimal parameters returned by wt.m
%
%----------------------------- Algorithm overview --------------------------
% Bisp(j,k) measures quadratic phase coupling between frequencies freq(j)
% and freq(k): it is the (time-)average of wt1(j,:) .* wt2(k,:) .*
% conj(WTdat_at_f3), where f3=freq(j)+freq(k) and WTdat_at_f3 is the
% wavelet transform of sig2 evaluated AT that exact sum frequency (which,
% since freq is a fixed geometric grid, generally falls BETWEEN two of the
% precomputed rows of wt2 — hence needing wtAtf2/wtAtf2_batch, which
% evaluate the transform at an arbitrary frequency rather than only at the
% grid points wt() already produced). If the three components are
% genuinely coupled (phase-locked, not just individually present), the
% product's phase stays roughly constant across time and the time-average
% has large magnitude; if they're independent, the phase varies freely and
% the average tends to cancel toward zero.
%
% For each row j, only k values with freq(j)+freq(k) still inside the
% analysed frequency range are meaningful (validMask/inRange below); the
% additional guardVal check further excludes (j,k) pairs whose sum
% frequency f3 would fall in a part of the grid coarser than the wavelet's
% own frequency resolution at that scale, where the "evaluate wt2 at an
% arbitrary frequency" trick would no longer be accurate. When auto=true
% (bispectrum of a signal with itself), kstart=j exploits Bisp(j,k) having
% no independent information beyond Bisp(k,j) below the diagonal — the sum
% f1+f2 is symmetric in (f1,f2), so only the upper triangle is computed.

% wbar/hObject are set here, before the try block, so the catch handler
% below can always safely check wbar==1 even if an error is thrown before
% the try block's own wbar=0 assignment would otherwise have run (e.g. a
% failure inside wt()/compareMatrix) — otherwise the catch itself throws a
% confusing secondary "undefined variable 'wbar'" error that masks the
% real one.
wbar = 0;
hObject = [];

try
    bstype={'111';'222';'122';'211'};

    if nargin > 3
        if nargin == 4 && iscell(varargin{1})
            in = varargin{1};
        else
            in = varargin;
        end
        
        % Routed through the shared cache (transformCached.m): the
        % Bispectrum module calls this function up to 4 times per button
        % press for the '111'/'222'/'122'/'211' bispectrum types, each pair
        % reusing signal 1 and/or signal 2's plain wavelet transform — this
        % turns those repeats into cache hits instead of redoing the FFT.
        [wt1, freq, opt] = transformCached(@wt, 'Wavelet', sig1, fs, in{:});
        [wt2, freq, opt] = transformCached(@wt, 'Wavelet', sig2, fs, in{:});
    else
        [wt1, freq, opt] = transformCached(@wt, 'Wavelet', sig1, fs);
        [wt2, freq, opt] = transformCached(@wt, 'Wavelet', sig2, fs);
    end
    
    dt = 1 / fs;
    nfreq = length(freq);
    Bisp = NaN * zeros(nfreq, nfreq);
    auto = false;
    if compareMatrix(wt1, wt2)
        auto = true;
    end

    if nargin >= 2 + 3
        for i = 1 : 2 : nargin - 3
            switch varargin{i}

                case 'handles'
                    handles=varargin{i+1};
                case 'hObject'
                    hObject=varargin{i+1};
                case 'num'
                    numb=varargin{i+1};
                case 'wbar'
                    wbar=varargin{i+1};
                    
            end
        end
    end
    
    if wbar==1
        handles.h = waitbar(0,'Calculating bispectrum...',...
            'CreateCancelBtn',...
            'setappdata(gcbf,''canceling'',1)');
        setappdata(handles.h,'canceling',0)
        if ~isempty(hObject); guidata(hObject,handles); end
    else
    end

    % freq is a COLUMN vector (wt.m builds it with a trailing transpose),
    % so freq(idx) follows freq's own (column) orientation regardless of
    % idx's shape — everything below is forced to a consistent ROW
    % orientation explicitly rather than relying on implicit shapes.
    freqRow = freq(:).';
    freqCol = freq(:);

    for j = 1 : nfreq
        if wbar==1
            if getappdata(handles.h,'canceling')
                if ~isempty(hObject); guidata(hObject,handles); end
                break;
            end
        else
        end
        kstart = 1;
        if auto
            kstart = j;
        end

        if wbar==1
            if getappdata(handles.h,'canceling')
                if ~isempty(hObject); guidata(hObject,handles); end
                break;
            end
        else
        end
        % For all k in this row at once: compute f3=freq(j)+freq(k) and the
        % same validity guard the original per-(j,k) loop used, then batch
        % every valid frequency in the row into ONE wtAtf2_batch call
        % instead of one wtAtf2 call per k — the expensive signal FFT/
        % preprocessing wtAtf2 redid on every call happens once per row now.
        kRange = kstart : nfreq;
        if ~isempty(kRange)
            f3all = freqRow(j) + freqRow(kRange);
            biggerAll = max(j, kRange);
            countLess = sum(freqCol < f3all, 1);
            idx3all = countLess + 1;
            inRange = f3all <= freqRow(end);
            safeIdx = max(idx3all-1, 1);
            guardVal = freqRow(safeIdx) > freqRow(biggerAll);
            validMask = inRange & guardVal;

            validK = kRange(validMask);
            if ~isempty(validK)
                f3valid = f3all(validMask);
                WTdatBatch = wtAtf2_batch(sig2, fs, f3valid, opt); % numel(validK) x N
                xxMat = wt1(j,:) .* wt2(validK,:) .* conj(WTdatBatch);
                Bisp(j, validK) = nanmean(xxMat, 2).';
            end
        end
        if wbar==1
            waitbar(j / length(freq),handles.h,sprintf(['Calculating Bispectrum ',bstype{numb},' (%d/%d)'],j,length(freq)));
        else
        end
    end
    if wbar==1
        delete(handles.h);
    else
    end
    
catch e
    errordlg(e.message,'Error');
    if wbar==1
        delete(handles.h);
    else
    end
    rethrow(e)
    
end