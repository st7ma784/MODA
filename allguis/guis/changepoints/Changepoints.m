classdef Changepoints < matlab.apps.AppBase
    %CHANGEPOINTS  Changepoint detection module: single-frequency and log-binned
    %   full-signal power. Mirrors the FastMODA /changepoints page and uses the
    %   shared engine (changepointsAtFrequency / changepointsLogBinnedPower).
    %
    %   The spectrogram is computed with wt.m (MODA wavelet transform); its
    %   magnitude drives both modes.
    %
    %   NOTE: written to the App Designer module pattern (embed-aware
    %   constructor) but not yet run in MATLAB — verify on load.

    properties (Access = public)
        UIFigure       matlab.ui.Figure
        RootContainer
        OwnsFigure = true
    end

    properties (Access = private)
        Grid
        AxFreq         matlab.ui.control.UIAxes    % single-frequency series
        AxBinned       matlab.ui.control.UIAxes    % binned-power heatmap
        LoadButton     matlab.ui.control.Button
        FsField        matlab.ui.control.NumericEditField
        WinField       matlab.ui.control.NumericEditField
        ModeDrop       matlab.ui.control.DropDown
        TargetField    matlab.ui.control.NumericEditField
        NBinsField     matlab.ui.control.NumericEditField
        ScaleDrop      matlab.ui.control.DropDown
        PowerDrop      matlab.ui.control.DropDown
        RunButton      matlab.ui.control.Button
        StatusLabel    matlab.ui.control.Label

        Signal = []
        Fs     = 40
        Freqs  = []
        Times  = []
        Sxx    = []
    end

    methods (Access = public)
        function app = Changepoints(parentContainer)
            if nargin < 1, parentContainer = []; end
            createComponents(app, parentContainer);
            registerApp(app, app.UIFigure);
            if nargout == 0, clear app; end
        end

        function delete(app)
            if app.OwnsFigure && isvalid(app.UIFigure)
                delete(app.UIFigure);
            end
        end
    end

    methods (Access = private)
        function createComponents(app, parentContainer)
            if isempty(parentContainer)
                app.UIFigure = uifigure('Name', 'MODA — Changepoints', ...
                    'Position', [100 100 1200 720]);
                app.OwnsFigure = true; app.RootContainer = app.UIFigure;
            else
                app.RootContainer = parentContainer;
                app.UIFigure = ancestor(parentContainer, 'figure');
                app.OwnsFigure = false;
            end

            g = uigridlayout(app.RootContainer, [3 1]);
            g.RowHeight = {'1x', '1x', 'fit'};
            g.ColumnWidth = {'1x'};
            app.Grid = g;

            app.AxFreq = uiaxes(g); app.AxFreq.Layout.Row = 1;
            title(app.AxFreq, 'Single-frequency power over time');
            xlabel(app.AxFreq, 'Time (s)'); ylabel(app.AxFreq, 'Power');

            app.AxBinned = uiaxes(g); app.AxBinned.Layout.Row = 2;
            title(app.AxBinned, 'Log-binned power (dB)');
            xlabel(app.AxBinned, 'Time (s)'); ylabel(app.AxBinned, 'Frequency (Hz)');

            ctrl = uigridlayout(g, [2 8]);
            ctrl.Layout.Row = 3;
            ctrl.RowHeight = {'fit','fit'};
            ctrl.ColumnWidth = repmat({'fit'}, 1, 8);

            app.LoadButton = uibutton(ctrl, 'Text', 'Load signal…', ...
                'ButtonPushedFcn', @(s,e) app.loadSignal());
            app.LoadButton.Layout.Row = 1; app.LoadButton.Layout.Column = 1;

            lblFs = uilabel(ctrl, 'Text', 'fs (Hz):'); lblFs.Layout.Row = 1; lblFs.Layout.Column = 2;
            app.FsField = uieditfield(ctrl, 'numeric', 'Value', 40);
            app.FsField.Layout.Row = 1; app.FsField.Layout.Column = 3;

            lblWin = uilabel(ctrl, 'Text', 'Window (s):'); lblWin.Layout.Row = 1; lblWin.Layout.Column = 4;
            app.WinField = uieditfield(ctrl, 'numeric', 'Value', 1);
            app.WinField.Layout.Row = 1; app.WinField.Layout.Column = 5;

            lblMode = uilabel(ctrl, 'Text', 'Mode:'); lblMode.Layout.Row = 1; lblMode.Layout.Column = 6;
            app.ModeDrop = uidropdown(ctrl, 'Items', {'Both','Single frequency','Log-binned power'}, ...
                'ItemsData', {'both','freq','binned'}, ...
                'ValueChangedFcn', @(s,e) app.modeChanged());
            app.ModeDrop.Layout.Row = 1; app.ModeDrop.Layout.Column = [7 8];

            lblT = uilabel(ctrl, 'Text', 'Target Freq (Hz):'); lblT.Layout.Row = 2; lblT.Layout.Column = 1;
            app.TargetField = uieditfield(ctrl, 'numeric', 'Value', 10);
            app.TargetField.Layout.Row = 2; app.TargetField.Layout.Column = 2;

            lblB = uilabel(ctrl, 'Text', 'Bins:'); lblB.Layout.Row = 2; lblB.Layout.Column = 3;
            app.NBinsField = uieditfield(ctrl, 'numeric', 'Value', 12);
            app.NBinsField.Layout.Row = 2; app.NBinsField.Layout.Column = 4;

            app.ScaleDrop = uidropdown(ctrl, 'Items', {'Log','Linear'}, ...
                'ItemsData', {'log','linear'});
            app.ScaleDrop.Layout.Row = 2; app.ScaleDrop.Layout.Column = 5;

            app.PowerDrop = uidropdown(ctrl, 'Items', {'Power','Amplitude'}, ...
                'ItemsData', {true, false});
            app.PowerDrop.Layout.Row = 2; app.PowerDrop.Layout.Column = 6;

            app.RunButton = uibutton(ctrl, 'Text', 'Detect changepoints', ...
                'ButtonPushedFcn', @(s,e) app.runDetection());
            app.RunButton.Layout.Row = 2; app.RunButton.Layout.Column = 7;

            app.StatusLabel = uilabel(ctrl, 'Text', 'Load a signal to begin.');
            app.StatusLabel.Layout.Row = 2; app.StatusLabel.Layout.Column = 8;

            app.modeChanged();
        end

        function modeChanged(app)
            m = app.ModeDrop.Value;
            showFreq = any(strcmp(m, {'both','freq'}));
            showBin  = any(strcmp(m, {'both','binned'}));
            app.AxFreq.Visible = local_onoff(showFreq);
            app.AxBinned.Visible = local_onoff(showBin);
            app.TargetField.Enable = local_onoff(showFreq);
            app.NBinsField.Enable = local_onoff(showBin);
            app.ScaleDrop.Enable = local_onoff(showBin);
        end

        function loadSignal(app)
            filt = {'*.mat;*.csv;*.txt;*.dat', 'Signal files'; '*.*', 'All files'};
            [name, pathn] = uigetfile(filt, 'Select a signal file');
            if isequal(name, 0), return; end
            try
                s = local_readOne(fullfile(pathn, name));
            catch ME
                uialert(app.UIFigure, ME.message, 'Load error'); return;
            end
            if ~isvector(s), s = s(1, :); end
            app.Signal = s(:).';
            app.Fs = app.FsField.Value;
            app.Sxx = [];
            app.StatusLabel.Text = sprintf('%s loaded (%d samples).', name, numel(app.Signal));
        end

        function ensureSpectrogram(app)
            if ~isempty(app.Sxx), return; end
            app.Fs = app.FsField.Value;
            % MODA wavelet transform magnitude as the time-frequency surface.
            % CutEdges off: changepoint detection needs the full time axis, and
            % NaN-masked cone-of-influence edges would break findchangepts.
            [WT, freq] = wt(app.Signal, app.Fs, 'Preprocess', 'off', ...
                            'CutEdges', 'off', 'Display', 'off');
            app.Sxx = abs(WT);
            app.Freqs = freq(:);
            app.Times = (0:size(WT, 2) - 1) / app.Fs;
        end

        function runDetection(app)
            if isempty(app.Signal)
                uialert(app.UIFigure, 'Load a signal first.', 'Changepoints'); return;
            end
            try
                app.ensureSpectrogram();
            catch ME
                uialert(app.UIFigure, ME.message, 'Spectrogram error'); return;
            end
            m = app.ModeDrop.Value;
            usePower = app.PowerDrop.Value;
            msg = '';

            if any(strcmp(m, {'both','freq'}))
                r1 = changepointsAtFrequency(app.Freqs, app.Times, app.Sxx, ...
                    app.TargetField.Value, 'UsePower', usePower);
                cla(app.AxFreq);
                plot(app.AxFreq, app.Times, r1.series, 'Color', [0.76 0.31 0.18]);
                local_vlines(app.AxFreq, r1.changepoint_times);
                title(app.AxFreq, sprintf('Changepoints at %.2f Hz (%d found)', ...
                    r1.actual_freq, numel(r1.changepoint_times)));
                msg = sprintf('freq: %d cps', numel(r1.changepoint_times));
            end

            if any(strcmp(m, {'both','binned'}))
                r2 = changepointsLogBinnedPower(app.Freqs, app.Times, app.Sxx, ...
                    'NBins', app.NBinsField.Value, 'Scale', app.ScaleDrop.Value, ...
                    'UsePower', usePower);
                cla(app.AxBinned);
                imagesc(app.AxBinned, app.Times, r2.bin_centers, ...
                    10*log10(r2.band_power.' + 1e-12));
                app.AxBinned.YDir = 'normal';
                if strcmp(app.ScaleDrop.Value, 'log'), app.AxBinned.YScale = 'log'; end
                local_vlines(app.AxBinned, r2.changepoint_times);
                title(app.AxBinned, sprintf('%s-binned power (%d bins) — %d changepoints', ...
                    app.ScaleDrop.Value, r2.n_bins, numel(r2.changepoint_times)));
                if ~isempty(msg), msg = [msg ' · ']; end
                msg = [msg sprintf('binned: %d cps', numel(r2.changepoint_times))];
            end

            app.StatusLabel.Text = msg;
        end
    end
end

% ---- helpers ----------------------------------------------------------------
function s = local_readOne(name)
    try, raw = load(name); catch, s = readmatrix(name); return; end
    if ~isstruct(raw), s = raw; return; end
    vars = fieldnames(raw);
    if numel(vars) == 1, s = raw.(vars{1}); else
        s = []; best = -1;
        for k = 1:numel(vars)
            c = raw.(vars{k});
            if isnumeric(c) && numel(c) > best, s = c; best = numel(c); end
        end
    end
    if isstruct(s)
        cand = {'Preprocessed_data','sig','Signal','signal','data','y','x'};
        got = [];
        for i = 1:numel(cand)
            if isfield(s, cand{i}) && ~isempty(s.(cand{i})), got = s.(cand{i}); break; end
        end
        s = got;
    end
    if ~isnumeric(s) || isempty(s)
        error('MODA:readOne:noSignal', 'The file does not contain a raw signal.');
    end
    s = double(s);
end

function local_vlines(ax, times)
    yl = ax.YLim;
    hold(ax, 'on');
    for i = 1:numel(times)
        plot(ax, [times(i) times(i)], yl, '--', 'Color', [0.85 0.1 0.1], 'LineWidth', 1.2);
    end
    hold(ax, 'off');
end

function s = local_onoff(tf)
    if tf, s = 'on'; else, s = 'off'; end
end
