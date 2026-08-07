% MODA data loading function

function [handles,sig,E]=MODAread(handles,type,varargin)

% Parse varargin for whether the number of signals should be even.
% This is used in phase coherence, bispectrum analysis and
% Bayesian inference.
even = false;
n = length(varargin);
for k = 1:n
    if strcmp("even", varargin{k})
        even = true;
        break;
    end
end

E=1;
if hasField(handles,'status')
    setStr(handles.status,'Importing Signal...');  % Update status
end

% Signal-file filter (was '*.*', which gave no indication CSV/text files
% are supported) + MultiSelect so tabs needing several signals (Coherence,
% Bispectrum, Bayesian) can pick one file per signal in a single dialog
% instead of loading a single file that already contains every row.
fileFilter = {'*.mat;*.csv;*.txt;*.dat', 'Signal files (*.mat, *.csv, *.txt, *.dat)'; ...
              '*.*', 'All files (*.*)'};
[filenames,pathname] = uigetfile(fileFilter, 'Select signal file(s)', 'MultiSelect','on');

if isequal(filenames,0)
    sig=0;
    return;
end
if ~iscell(filenames)
    filenames = {filenames};
end

if numel(filenames) == 1
    % ---- Single file: unchanged behavior. The file may itself already
    % contain multiple signal rows (e.g. an existing multi-signal dataset),
    % so orientation is ambiguous and the user is asked to clarify. ----
    name = fullfile(pathname,filenames{1});
    try
        sig = readSignalFile(name);
    catch ME
        errordlg(ME.message, 'Data Import');
        sig = 0; E = 0; return;
    end

    handles.sampling_freq = str2double(cell2mat(newid(['Enter the sampling frequency of the data (',filenames{1},') in Hz'])));
    fs = handles.sampling_freq;

    if isnan(fs)
        errordlg('Sampling frequency must be specified')
        E=0;
        return;
    end

    choice = questdlg('Select Orientation of Data set?', ...
        'Data Import','Column wise','Row wise','Row wise');
    switch choice
        case 'Column wise'
            sig = sig';
    end

    if isempty(choice)
        errordlg('Data set orientation must be specified')
        E=0;
        return;
    end
else
    % ---- Multiple files: one signal per file, stacked as rows. Each
    % file is expected to hold a single signal (a row or column vector);
    % if a file happens to contain more than one row, only its first row
    % is used, since which extra rows would belong to which other file's
    % signal is otherwise ambiguous. ----
    handles.sampling_freq = str2double(cell2mat(newid('Enter the sampling frequency of the data in Hz')));
    fs = handles.sampling_freq;
    if isnan(fs)
        errordlg('Sampling frequency must be specified')
        E=0;
        return;
    end

    rows = cell(numel(filenames),1);
    minLen = Inf;
    for k = 1:numel(filenames)
        try
            s = readSignalFile(fullfile(pathname,filenames{k}));
        catch ME
            errordlg(ME.message, 'Data Import');
            sig = 0; E = 0; return;
        end
        if isvector(s)
            rows{k} = s(:).';
        else
            rows{k} = s(1,:);
        end
        minLen = min(minLen, numel(rows{k}));
    end
    if any(cellfun(@numel,rows) ~= minLen)
        warndlg('Selected files have different lengths; trimming all signals to the shortest one.','Signal length mismatch');
    end
    sig = cell2mat(cellfun(@(r) r(1:minLen), rows, 'UniformOutput', false));
end

num_signals = length(sig(:,1));

% If there are an odd number of signals but an even number must be
% supplied, remove the last one.
% Only do this if there are 3 or more signals, because users may want to
% analyse a single signal.
if even && num_signals > 2 && mod(num_signals, 2) ~= 0
    sig = sig(1:end-1,:);
end

% Assign the loaded data through setProp so this shared reader works whether
% [handles] is a classic GUIDE struct (any field can be added) or an App
% Designer app object (only *declared* properties can be set — a module that
% doesn't declare a legacy field like time_axis_cut/xl simply skips it instead
% of throwing "Unrecognized property").
nSamp = size(sig,2);
time  = linspace(0, nSamp/handles.sampling_freq, nSamp);
handles = setProp(handles,'sig',           sig);
handles = setProp(handles,'sig_cut',       sig);
handles = setProp(handles,'sig_pp',        sig);
handles = setProp(handles,'time_axis',     time);
handles = setProp(handles,'time_axis_cut', time);
handles = setProp(handles,'xl',            [time(1) time(end)]);

if type==1
    N=size(sig);
    if N(1)==1;
        % Paired modules (Coherence/Bispectrum/Bayesian) need an even number of
        % signals; a lone signal is duplicated into a pair. Assign through
        % setProp so a module that doesn't declare every legacy field (e.g.
        % Bayesian has no sig_pp) skips it instead of throwing.
        dup = [sig; sig];
        handles = setProp(handles,'sig',     dup);
        handles = setProp(handles,'sig_cut', dup);
        handles = setProp(handles,'sig_pp',  dup);
    else

        %% Plot time series
        linkaxes([handles.time_series_1 handles.time_series_2],'x'); % Ensures axis limits are identical for both plots
        plot(handles.time_series_1,handles.time_axis,handles.sig(1,:),'color',handles.linecol(1,:));
        xlim(handles.time_series_1,[0 handles.time_axis(end)]);
        plot(handles.time_series_2,handles.time_axis,handles.sig(1+size(handles.sig,1)/2,:),'color',handles.linecol(1,:));
        xlim(handles.time_series_2,[0 handles.time_axis(end)]);
        xlabel(handles.time_series_2,'Time (s)');
        ylabel(handles.time_series_1,'Sig 1');
        ylabel(handles.time_series_2,'Sig 2');

        if   mod(N(1),2)==1;
            errordlg('Number of data sets must be even','Data Error');
            E=0;
            return;

        end

        %% Create signal list
        % hasField (not isfield) so this works when [handles] is an App
        % Designer object — isfield is struct-only and would always be false,
        % leaving the Signal Pair list empty for Coherence/Bispectrum/Bayesian.
        if hasField(handles,'signal_list')
            list = cell(size(sig,1)/2,1);
            list{1,1} = 'Signal Pair 1';

            for i = 2:size(sig,1)/2
                list{i,1} = sprintf('Signal Pair %d',i);
            end

            setListItems(handles.signal_list,list);
        else
        end
    end
else

    %% Plot time series
    plot(handles.time_series,handles.time_axis,sig(1,:),'color',handles.linecol(1,:));
    xlim(handles.time_series,[0 handles.time_axis(end)]);
    xlabel(handles.time_series,'Time (s)');
    ylabel(handles.time_series,'Sig');

end




% plot_TS is a TFA-only control; the paired modules (Bayesian etc.) don't
% declare it, so only touch it when present — otherwise accessing
% handles.plot_TS throws before setEnable can run.
if hasField(handles,'plot_TS')
    setEnable(handles.plot_TS,'on');
end

if hasField(handles,'status')
    setStr(handles.status,'Select data and define parameters');
end

end % MODAread

% ---- UI-agnostic helpers ----------------------------------------
function setStr(h, s)
    try; set(h,'String',s); catch; h.Value = s; end
end

function setEnable(h, s)
    try; set(h,'Enable',s); catch; h.Enable = s; end
end

function setListItems(h, items)
    try; set(h,'String',items); catch; h.Items = items; end
end

function handles = setProp(handles, name, value)
    % Set a data field on [handles], which may be a GUIDE struct (any field can
    % be added) or an App Designer app object (only declared properties can be
    % set). For an object without this property, silently skip: the module that
    % didn't declare the field doesn't use it, so setting it would only throw.
    if isstruct(handles) || isprop(handles, name)
        handles.(name) = value;
    end
end

function tf = hasField(handles, name)
    % True if [name] exists on [handles], whether it's a GUIDE struct (isfield)
    % or an App Designer app object (isprop). isfield alone is always false for
    % an object, which would wrongly skip present controls (e.g. signal_list).
    if isstruct(handles)
        tf = isfield(handles, name);
    else
        tf = isprop(handles, name);
    end
end

function sig = readSignalFile(name)
    % Loads one signal file as a plain numeric array, regardless of whether
    % it's a .mat file (a bare signal matrix, a single stored variable, or a
    % MODA-saved analysis struct such as TFR_data/Filtered_data) or a
    % delimited text format (.csv/.txt/.dat).
    try
        raw = load(name); % .mat -> struct of variables; ascii -> numeric matrix
    catch
        % Excel CSVs with a BOM, or other delimited text load() can't parse.
        sig = readmatrix(name);
        return;
    end

    if ~isstruct(raw)          % ascii load() already returned a numeric matrix
        sig = raw;
        return;
    end

    % .mat files come back as a struct keyed by variable name. Unwrap a single
    % stored variable (the common case); with several, take the largest numeric.
    vars = fieldnames(raw);
    if numel(vars) == 1
        sig = raw.(vars{1});
    else
        sig = pickLargestNumeric(raw);
    end

    % A MODA-saved file stores a struct of analysis results (heterogeneous:
    % numeric matrices mixed with char fields like Wavelet_type), NOT a raw
    % signal — the old struct2array() call turned this into an unusable struct
    % and the caller then failed silently, leaving the entries list empty. Pull
    % the actual signal out of it instead.
    if isstruct(sig)
        sig = extractSignalFromModaStruct(sig);
    end
    if iscell(sig)
        try, sig = cell2mat(sig); catch, sig = cell2mat(sig(:)); end
    end

    if ~isnumeric(sig) || isempty(sig)
        error('MODA:readSignalFile:noSignal', ...
            ['"%s" does not contain a raw signal that can be loaded here.\n', ...
             'If it is a MODA-saved analysis/session file, reopen it with ', ...
             '"Load session" instead of "Load time series".'], name);
    end
    sig = double(sig);
end

function v = pickLargestNumeric(s)
    % Largest numeric field of a struct (by element count); [] if none.
    v = []; best = -1; f = fieldnames(s);
    for k = 1:numel(f)
        c = s.(f{k});
        if isnumeric(c) && numel(c) > best, v = c; best = numel(c); end
    end
end

function sig = extractSignalFromModaStruct(S)
    % Best-effort recovery of the signal from a MODA-saved struct. Known
    % signal-bearing fields are tried in preference order (a TFA save stores
    % Preprocessed_data; a Filtering save stores Ridge_recon), then any raw
    % signal field, then the largest numeric field as a last resort.
    cand = {'Preprocessed_data','Ridge_recon','sig','Signal','signal', ...
            'data','Data','y','x'};
    for k = 1:numel(cand)
        if isfield(S, cand{k})
            v = S.(cand{k});
            if iscell(v) && ~isempty(v), sig = v; return; end
            if isnumeric(v) && ~isempty(v), sig = v; return; end
        end
    end
    sig = pickLargestNumeric(S);
end
