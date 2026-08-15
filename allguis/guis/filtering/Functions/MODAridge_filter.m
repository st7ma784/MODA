function handles = MODAridge_filter(hObject, eventdata, handles)
% MODAridge_filter  Ridge extraction and Butterworth band filtering.
% Compatible with MATLAB App Designer (classdef) and legacy GUIDE handles.
% Compatible with MATLAB R2023a through R2026a.

% ---- UI-agnostic helpers ----------------------------------------
    function s = readStr(h)
        try; s = get(h,'String'); catch; s = h.Value; end
        if iscell(s); s = s{1}; end
        s = char(s);
    end

    function idx = readIdx(h)
        try
            idx = get(h,'Value');
            if ~isnumeric(idx); error(''); end
        catch
            try
                items = h.Items; val = h.Value;
                idx = find(strcmp(items, char(val)), 1);
                if isempty(idx); idx = 1; end
            catch; idx = 1; end
        end
    end

    function items = readItems(h)
        try; items = get(h,'String'); catch; items = h.Items; end
        if ischar(items); items = {items}; end
    end

    function setEnable(h, s)
        try; set(h,'Enable',s); catch; h.Enable = s; end
    end

    function setDropdownVal(h, idx)
        % Set dropdown to a 1-based index regardless of UI type.
        try
            set(h,'Value',idx);   % GUIDE numeric index
        catch
            try
                h.Value = h.Items{idx};   % App Designer
            catch; end
        end
    end

    function h = resolve(handles, varargin)
        % Return the first field/property from the name list that exists.
        h = [];
        for k = 1:numel(varargin)
            name = varargin{k};
            try
                if isstruct(handles)
                    if isfield(handles, name); h = handles.(name); return; end
                else
                    if isprop(handles, name); h = handles.(name); return; end
                end
            catch; end
        end
    end
% ---- end helpers -------------------------------------------------

% Resolve component handles that were renamed in App Designer
h_transform     = resolve(handles, 'transform',      'transform_btn');
h_filter_signal = resolve(handles, 'filter_signal',  'filter_signal_btn');
h_ridgecalc     = resolve(handles, 'ridgecalc',      'ridgecalc_btn');
h_save_filt     = resolve(handles, 'save_filtered_sig_plot', 'SaveFiltSigPlotMenu');
h_save_ridge    = resolve(handles, 'save_ridge_plot',        'SaveRidgePlotMenu');
h_save_phase    = resolve(handles, 'save_phase_plot',        'SavePhasePlotMenu');
h_all_filt      = resolve(handles, 'All_filt_plot',          'AllFiltPlotMenu');
h_save_csv      = resolve(handles, 'save_csv',  'SaveCsvMenu',  'csv_save');
h_save_mat      = resolve(handles, 'save_mat',  'SaveMatMenu',  'mat_save');
h_save_session  = resolve(handles, 'save_session', 'SaveSessionMenu');

list = readItems(handles.interval_list);
if isempty(list) || (numel(list)==1 && isempty(list{1}))
    errordlg('Interval list is empty. Please select frequency bands for filtering','Error');
    return;
end

setEnable(h_transform,     'off');
setEnable(h_filter_signal, 'off');
setEnable(h_ridgecalc,     'off');

%% Set up waitbar
handles.h = waitbar(0,'Filtering...', ...
    'CreateCancelBtn','setappdata(gcbf,''canceling'',1)');
setappdata(handles.h,'canceling',0);
try; guidata(hObject,handles); catch; end

try
    list = readItems(handles.interval_list);

    extype = handles.etype;

    if extype == 2   %% Butterworth filter
        for j = 1:size(handles.sig_cut,1)
            for k = 1:size(list,1)
                fl = csv_to_mvar(list{k,1});
                [handles.bands{j,k},~]       = loop_butter(handles.sig_cut(j,:), fl, handles.sampling_freq);
                handles.extract_phase{j,k}   = angle(hilbert(handles.bands{j,k}));
                handles.extract_amp{j,k}     = abs(hilbert(handles.bands{j,k}));
            end
        end

    elseif extype == 1   %% Ridge extraction
        wtypes   = readItems(handles.wind_type);
        wselect  = readIdx(handles.wind_type);
        wtype    = wtypes{wselect};

        ppItems  = readItems(handles.preprocess);
        ppselect = ppItems{readIdx(handles.preprocess)};

        cutItems  = readItems(handles.cutedges);
        cutselect = cutItems{readIdx(handles.cutedges)};

        for j = 1:size(handles.sig_cut,1)
            for k = 1:size(list,1)
                fl = csv_to_mvar(list{k,1});

                if ~isfield(handles,'fc') && ~isprop(handles,'fc')
                    msg = 'An error has occurred. Please re-calculate the transform before proceeding.';
                    errordlg(msg); error(msg);
                end

                if isnan(handles.fc)
                    if handles.calc_type == 1
                        [WT,freqarr,wopt] = wt(handles.sig_cut(j,:), handles.sampling_freq, ...
                            'fmin',fl(1),'fmax',fl(2),'CutEdges','off','Preprocess',ppselect,'Wavelet',wtype);
                    else
                        [WT,freqarr,wopt] = wft(handles.sig_cut(j,:), handles.sampling_freq, ...
                            'fmin',fl(1),'fmax',fl(2),'CutEdges','off','Preprocess',ppselect,'Window',wtype);
                    end
                else
                    if handles.calc_type == 1
                        [WT,freqarr,wopt] = wt(handles.sig_cut(j,:), handles.sampling_freq, ...
                            'fmin',fl(1),'fmax',fl(2),'CutEdges','off','Preprocess',ppselect,'Wavelet',wtype,'f0',handles.fc);
                    else
                        [WT,freqarr,wopt] = wft(handles.sig_cut(j,:), handles.sampling_freq, ...
                            'fmin',fl(1),'fmax',fl(2),'CutEdges','off','Preprocess',ppselect,'Window',wtype,'f0',handles.fc);
                    end
                end

                tfsupp = ecurve(WT, freqarr, wopt);
                [handles.bands_iamp{j,k}, handles.bands_iphi{j,k}, handles.bands_freq{j,k}] = ...
                    rectfr(tfsupp, WT, freqarr, wopt, 'direct');
                handles.recon{j,k}       = handles.bands_iamp{j,k} .* cos(handles.bands_iphi{j,k});
                handles.bands_iphi{j,k}  = mod(handles.bands_iphi{j,k}, 2*pi);
            end
            waitbar(j / size(handles.sig_cut,1), handles.h);
        end

        % Warn about negative frequencies
        shown_error = false;
        for i = 1:size(handles.bands_freq,1)
            for j = 1:size(handles.bands_freq,2)
                if ~isempty(find(handles.bands_freq{i,j} < 0, 1))
                    msg = ['Error: Negative frequencies present in result. ' ...
                           'Try: (1) Lognorm wavelet, (2) narrower frequency interval, ' ...
                           '(3) higher frequency resolution.'];
                    errordlg(msg,'Negative Frequencies');
                    shown_error = true;
                    break;
                end
            end
            if shown_error; break; end
        end
    end

    setEnable(handles.display_type, 'on');
    setDropdownVal(handles.display_type, 2);
    delete(handles.h);
    drawnow;
    try; guidata(hObject,handles); catch; end

    setEnable(h_transform,     'on');
    setEnable(h_filter_signal, 'on');
    setEnable(h_ridgecalc,     'on');
    setEnable(h_save_filt,     'on');
    setEnable(h_save_ridge,    'on');
    setEnable(h_save_phase,    'on');
    setEnable(h_all_filt,      'on');
    setEnable(h_save_csv,      'on');
    setEnable(h_save_mat,      'on');
    setEnable(h_save_session,  'on');

catch e
    errordlg(e.message,'Error');
    setEnable(h_transform,     'on');
    setEnable(h_filter_signal, 'on');
    setEnable(h_ridgecalc,     'on');
    try; delete(handles.h); catch; end
    rethrow(e);
end
