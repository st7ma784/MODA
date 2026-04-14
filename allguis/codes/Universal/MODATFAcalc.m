function handles = MODATFAcalc(hObject, eventdata, handles, ty)
% MODATFAcalc  Wavelet / WFT computation for the TFA GUI.
%   ty  - 1 = all signals,  2 = single selected signal
%
% Compatible with MODA App Designer (classdef) interface.
% Replaces old GUIDE script; wtwrapper is now a proper function.
% Compatible with MATLAB R2023a through R2026a.

% ---- UI-agnostic parameter readers ----------------------------------
% Works for both App Designer uicontrols and legacy GUIDE uicontrols.
    function s = readStr(h)
        % Return the string value of an edit-field or the selected item
        % of a listbox/dropdown regardless of UI framework.
        try
            s = get(h, 'String');   % GUIDE uicontrol
        catch
            s = h.Value;            % App Designer uieditfield
        end
        if iscell(s); s = s{1}; end
        s = char(s);
    end

    function idx = readIdx(h)
        % Return 1-based numeric selection index.
        try
            idx = get(h, 'Value');  % GUIDE uicontrol — already numeric
            if ~isnumeric(idx); error('not numeric'); end
        catch
            % App Designer: Value is a string; find it in Items.
            try
                items = h.Items;
                val   = h.Value;
                if iscell(val); val = val{1}; end
                idx   = find(strcmp(items, char(val)), 1);
                if isempty(idx); idx = 1; end
            catch
                idx = 1;
            end
        end
    end

    function items = readItems(h)
        % Return the full cell-array of options.
        try
            items = get(h, 'String');   % GUIDE
        catch
            items = h.Items;            % App Designer
        end
        if ischar(items); items = {items}; end
    end

    function setStr(h, s)
        try
            set(h, 'String', s);        % GUIDE
        catch
            h.Value = s;                % App Designer
        end
    end

    function setEnable(h, s)
        try
            set(h, 'Enable', s);
        catch
            h.Enable = s;
        end
    end
% ---- end helpers ------------------------------------------------------

setEnable(handles.plot_TS,        'on');
setEnable(handles.save_3dplot,    'on');
setEnable(handles.save_both_plot, 'on');
setEnable(handles.save_avg_plot,  'on');
setEnable(handles.save_mm_plot,   'on');
setEnable(handles.wt_single,      'off');
setEnable(handles.wavlet_transform,'off');
if ty == 2
    setEnable(handles.save_WT_coeff, 'on');
else
    setEnable(handles.save_WT_coeff, 'off');
end
setEnable(handles.save_session, 'on');

handles.failed = false;
try
    % ---- Read parameters -----------------------------------------------
    fmax = str2double(readStr(handles.max_freq));
    fmin = str2double(readStr(handles.min_freq));
    fs   = handles.sampling_freq;
    f0   = str2double(readStr(handles.central_freq));
    A    = f0 <= 0.4;

    items          = readItems(handles.wavelet_type);
    index_selected = readIdx(handles.wavelet_type);
    wtype          = items{index_selected};
    B              = strcmp(wtype, 'Bump');

    handles.fc = f0;

    if (A+0) + (B+0) == 2
        errordlg('The bump wavelet requires that f0 > 0.4. Please enter a higher value.','Parameter Error');
        handles.failed = true;
        setEnable(handles.wt_single,       'on');
        setEnable(handles.wavlet_transform,'on');
        return;
    end

    if fmax > fs/2
        errordlg(['Maximum frequency cannot be higher than the Nyquist frequency. Please enter a value ≤ ', num2str(fs/2),' Hz.'],'Parameter Error');
        handles.failed = true;
        setEnable(handles.wt_single,       'on');
        setEnable(handles.wavlet_transform,'on');
        return;
    end

    % WFT requires an explicit fmin
    if handles.calc_type == 2 && isnan(fmin)
        errordlg('Minimum frequency must be specified for WFT.','Parameter Error');
        handles.failed = true;
        setEnable(handles.wt_single,       'on');
        setEnable(handles.wavlet_transform,'on');
        return;
    elseif handles.calc_type == 2
        handles.fc = f0 / fmin;
    end

    if isnan(fs)
        errordlg('Sampling frequency must be specified.','Parameter Error');
        handles.failed = true;
    end

    if handles.calc_type == 1
        setStr(handles.status, 'Calculating Wavelet Transform...');
    else
        setStr(handles.status, 'Calculating Windowed Fourier Transform...');
    end

    % Kaiser window special handling
    if strcmp(wtype, 'Kaiser')
        a     = str2double(readStr(handles.kaisera));
        wtype = ['kaiser-', num2str(a)];
    end

    ppItems   = readItems(handles.preprocess);
    ppIdx     = readIdx(handles.preprocess);
    ppselect  = ppItems{ppIdx};

    cutItems  = readItems(handles.cutedges);
    cutIdx    = readIdx(handles.cutedges);
    cutselect = cutItems{cutIdx};

    if ~isfield(handles,'sig')
        errordlg('Signal not found.','Signal Error');
        handles.failed = true;
    end
    sig = handles.sig;

    xl = csv_to_mvar(readStr(handles.xlim));
    xl = xl .* fs;
    xl(2) = min(xl(2), size(sig, 2));
    xl(1) = max(xl(1), 1);
    xl = xl ./ fs;
    time_axis = xl(1) : 1/fs : xl(2);

    if length(time_axis) >= 2000
        screensize   = max(get(groot,'Screensize'));
        under_sample = floor(size(sig,2) / screensize);
    else
        under_sample = 1;
    end
    if handles.calc_type == 2
        under_sample = ceil(under_sample * 3.5);
    end
    handles.time_axis_us = time_axis(1:under_sample:end);
    n = size(handles.sig, 1);
    handles.WT = cell(n, 1);

    % Crop signal to selected window
    xl2 = csv_to_mvar(readStr(handles.xlim));
    xl2 = xl2 .* fs;
    xl2(2) = min(xl2(2), size(handles.sig, 2));
    xl2(1) = max(xl2(1), 1);
    handles.sig_cut = sig(:, xl2(1):xl2(2));

    if handles.calc_type == 1
        if fmin <= 1/(length(handles.sig_cut)/fs)
            errordlg('WT minimum frequency too low. Leave "Min Freq" blank for auto.','Parameter Error');
            handles.failed = true;
            setEnable(handles.wt_single,       'on');
            setEnable(handles.wavlet_transform,'on');
            return;
        end
    end

    if handles.calc_type == 1
        setStr(handles.status, 'Calculating Wavelet Transform...');
    else
        setStr(handles.status, 'Calculating Windowed Fourier Transform...');
    end

    handles.amp_WT  = cell(n, 1);
    handles.pow_WT  = cell(n, 1);
    handles.pow_arr = cell(n, 1);
    handles.amp_arr = cell(n, 1);

    handles.h = waitbar(0,'Calculating transform...', ...
        'CreateCancelBtn','setappdata(gcbf,''canceling'',1)');
    setappdata(handles.h,'canceling',0);
    try; guidata(hObject,handles); catch; end

    if ty == 2
        I = readIdx(handles.signal_list);
    else
        I = 1:n;
    end

    for p = I
        if getappdata(handles.h,'canceling'); break; end

        if ty == 1; count = n; else; count = 1; end

        if handles.calc_type == 1
            setStr(handles.status, sprintf('Calculating Wavelet Transform of Signal %d/%d', p, count));
        else
            setStr(handles.status, sprintf('Calculating Windowed Fourier Transform of Signal %d/%d', p, count));
        end

        % Call wtwrapper as a proper function (replaces old script call)
        [WT, handles.freqarr, handles.wopt] = wtwrapper( ...
            handles.sig_cut(p,:), fs, handles.fc, fmin, fmax, ...
            handles.calc_type, wtype, cutselect, ppselect);

        handles.WT{p,1} = WT;
        WTamp = abs(WT);
        WTpow = abs(WT).^2;
        handles.pow_arr{p,1} = nanmean(WTpow.');
        handles.amp_arr{p,1} = nanmean(WTamp.');
        handles.amp_WT{p,1}  = WTamp(:, 1:under_sample:end);
        handles.pow_WT{p,1}  = WTpow(:, 1:under_sample:end);
        waitbar(p/n, handles.h);
    end
    try; guidata(hObject,handles); catch; end

    delete(handles.h);
    setEnable(handles.wt_single,        'on');
    setEnable(handles.wavlet_transform, 'on');
    setEnable(handles.mat_save,         'on');
    setEnable(handles.csv_save,         'on');
    setEnable(handles.save_WT_coeff,    'on');
    setEnable(handles.save_session,     'on');

catch e
    errordlg(e.message,'Error');
    handles.failed = true;
    setEnable(handles.wt_single,        'on');
    setEnable(handles.wavlet_transform, 'on');
    try; delete(handles.h); catch; end
    rethrow(e);
end
try; guidata(hObject, handles); catch; end
