classdef Preprocessing < matlab.apps.AppBase
    %PREPROCESSING  Dedicated preprocessing module: clip / bulk-crop / integer
    %   decimate signals before analysis, with a live slice preview. Mirrors the
    %   FastMODA /preprocess page. Uses the shared engine cropAndDecimate().
    %
    %   NOTE: written to match the App Designer module pattern (embed-aware
    %   constructor) but not yet run in MATLAB — verify on load.

    properties (Access = public)
        UIFigure       matlab.ui.Figure
        RootContainer                       % UIFigure (standalone) or a uitab
        OwnsFigure = true
    end

    properties (Access = private)
        Grid
        PreviewAxes    matlab.ui.control.UIAxes
        FileList       matlab.ui.control.ListBox
        FsField        matlab.ui.control.NumericEditField
        ModeDrop       matlab.ui.control.DropDown
        StartField     matlab.ui.control.NumericEditField
        StopField      matlab.ui.control.NumericEditField
        LenField       matlab.ui.control.NumericEditField
        DecimDrop      matlab.ui.control.DropDown
        ApplyButton    matlab.ui.control.Button
        SaveButton     matlab.ui.control.Button
        LoadButton     matlab.ui.control.Button
        StatusLabel    matlab.ui.control.Label
        SlicePatch                          % handle to the shaded overlay

        Signals   = {}                      % raw signals (row vectors)
        FileNames = {}
        Fs        = 1
        Processed = {}                      % cropped+decimated signals
        ProcessedFs = 1
    end

    methods (Access = public)
        function app = Preprocessing(parentContainer)
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
                app.UIFigure = uifigure('Name', 'MODA — Preprocessing', ...
                    'Position', [100 100 1200 720]);
                app.OwnsFigure = true;
                app.RootContainer = app.UIFigure;
            else
                app.RootContainer = parentContainer;
                app.UIFigure = ancestor(parentContainer, 'figure');
                app.OwnsFigure = false;
            end

            g = uigridlayout(app.RootContainer, [2 1]);
            g.RowHeight = {'2x', 'fit'};
            g.ColumnWidth = {'1x'};
            app.Grid = g;

            % --- preview axes ---
            app.PreviewAxes = uiaxes(g);
            app.PreviewAxes.Layout.Row = 1;
            app.PreviewAxes.Layout.Column = 1;
            title(app.PreviewAxes, 'Signal preview — set start/stop to see the slice');
            xlabel(app.PreviewAxes, 'Time (s)'); ylabel(app.PreviewAxes, 'Amplitude');

            % --- controls panel ---
            ctrl = uigridlayout(g, [3 8]);
            ctrl.Layout.Row = 2; ctrl.Layout.Column = 1;
            ctrl.RowHeight = {'fit','fit','fit'};
            ctrl.ColumnWidth = repmat({'fit'}, 1, 8);

            app.LoadButton = uibutton(ctrl, 'Text', 'Load file(s)…', ...
                'ButtonPushedFcn', @(s,e) app.loadFiles());
            app.LoadButton.Layout.Row = 1; app.LoadButton.Layout.Column = 1;

            lblFs = uilabel(ctrl, 'Text', 'fs (Hz):');
            lblFs.Layout.Row = 1; lblFs.Layout.Column = 2;
            app.FsField = uieditfield(ctrl, 'numeric', 'Value', 40, ...
                'ValueChangedFcn', @(s,e) app.fsChanged());
            app.FsField.Layout.Row = 1; app.FsField.Layout.Column = 3;

            lblList = uilabel(ctrl, 'Text', 'Loaded:');
            lblList.Layout.Row = 1; lblList.Layout.Column = 4;
            app.FileList = uilistbox(ctrl, 'Items', {}, ...
                'ValueChangedFcn', @(s,e) app.previewSelected());
            app.FileList.Layout.Row = 1; app.FileList.Layout.Column = [5 8];

            lblMode = uilabel(ctrl, 'Text', 'Crop:');
            lblMode.Layout.Row = 2; lblMode.Layout.Column = 1;
            app.ModeDrop = uidropdown(ctrl, ...
                'Items', {'None', 'Range (start–stop)', 'First N seconds', 'Final N seconds'}, ...
                'ItemsData', {'none','range','first','final'}, ...
                'ValueChangedFcn', @(s,e) app.modeChanged());
            app.ModeDrop.Layout.Row = 2; app.ModeDrop.Layout.Column = 2;

            lblStart = uilabel(ctrl, 'Text', 'Start (s):');
            lblStart.Layout.Row = 2; lblStart.Layout.Column = 3;
            app.StartField = uieditfield(ctrl, 'numeric', 'Value', 0, ...
                'ValueChangedFcn', @(s,e) app.drawSlice());
            app.StartField.Layout.Row = 2; app.StartField.Layout.Column = 4;

            lblStop = uilabel(ctrl, 'Text', 'Stop (s):');
            lblStop.Layout.Row = 2; lblStop.Layout.Column = 5;
            app.StopField = uieditfield(ctrl, 'numeric', 'Value', 0, ...
                'ValueChangedFcn', @(s,e) app.drawSlice());
            app.StopField.Layout.Row = 2; app.StopField.Layout.Column = 6;

            lblLen = uilabel(ctrl, 'Text', 'Length N (s):');
            lblLen.Layout.Row = 2; lblLen.Layout.Column = 7;
            app.LenField = uieditfield(ctrl, 'numeric', 'Value', 0, ...
                'ValueChangedFcn', @(s,e) app.drawSlice());
            app.LenField.Layout.Row = 2; app.LenField.Layout.Column = 8;

            lblDec = uilabel(ctrl, 'Text', 'Downsample:');
            lblDec.Layout.Row = 3; lblDec.Layout.Column = 1;
            app.DecimDrop = uidropdown(ctrl, 'Items', {'1× — no change'}, ...
                'ItemsData', {1});
            app.DecimDrop.Layout.Row = 3; app.DecimDrop.Layout.Column = 2;

            app.ApplyButton = uibutton(ctrl, 'Text', 'Apply to file(s)', ...
                'ButtonPushedFcn', @(s,e) app.applyProcessing());
            app.ApplyButton.Layout.Row = 3; app.ApplyButton.Layout.Column = 3;

            app.SaveButton = uibutton(ctrl, 'Text', 'Save cropped…', ...
                'ButtonPushedFcn', @(s,e) app.saveProcessed());
            app.SaveButton.Layout.Row = 3; app.SaveButton.Layout.Column = 4;

            app.StatusLabel = uilabel(ctrl, 'Text', 'Load one or more signal files to begin.');
            app.StatusLabel.Layout.Row = 3; app.StatusLabel.Layout.Column = [5 8];

            app.modeChanged();
        end

        function loadFiles(app)
            filt = {'*.mat;*.csv;*.txt;*.dat', 'Signal files'; '*.*', 'All files'};
            [names, pathn] = uigetfile(filt, 'Select signal file(s)', 'MultiSelect', 'on');
            if isequal(names, 0), return; end
            if ~iscell(names), names = {names}; end
            app.Signals = {}; app.FileNames = {};
            for k = 1:numel(names)
                try
                    s = app.readOne(fullfile(pathn, names{k}));
                catch ME
                    uialert(app.UIFigure, ME.message, 'Load error'); return;
                end
                if ~isvector(s), s = s(1, :); end
                app.Signals{end+1} = s(:).';
                app.FileNames{end+1} = names{k};
            end
            app.Fs = app.FsField.Value;
            app.FileList.Items = app.FileNames;
            if ~isempty(app.FileNames), app.FileList.Value = app.FileNames{1}; end
            app.populateDecim();
            app.previewSelected();
            app.StatusLabel.Text = sprintf('%d file(s) loaded.', numel(app.Signals));
        end

        function s = readOne(~, name)
            % Robust single-signal reader (mirrors MODAread.readSignalFile).
            try
                raw = load(name);
            catch
                s = readmatrix(name); return;
            end
            if ~isstruct(raw), s = raw; return; end
            vars = fieldnames(raw);
            if numel(vars) == 1, s = raw.(vars{1}); else, s = local_largestNumeric(raw); end
            if isstruct(s)
                cand = {'Preprocessed_data','Ridge_recon','sig','Signal','signal','data','y','x'};
                got = [];
                for i = 1:numel(cand)
                    if isfield(s, cand{i}) && ~isempty(s.(cand{i})), got = s.(cand{i}); break; end
                end
                if isempty(got), got = local_largestNumeric(s); end
                s = got;
            end
            if iscell(s), s = cell2mat(s); end
            if ~isnumeric(s) || isempty(s)
                error('MODA:readOne:noSignal', ...
                    'The file does not contain a raw signal (is it a saved session?).');
            end
            s = double(s);
        end

        function idx = selectedIndex(app)
            idx = find(strcmp(app.FileNames, app.FileList.Value), 1);
            if isempty(idx), idx = 1; end
        end

        function previewSelected(app)
            if isempty(app.Signals), return; end
            x = app.Signals{app.selectedIndex()};
            t = (0:numel(x)-1) / app.Fs;
            plot(app.PreviewAxes, t, x, 'Color', [0.76 0.31 0.18]);
            app.PreviewAxes.XLim = [0, max(t)];
            app.SlicePatch = [];
            % sensible default window
            if app.StopField.Value == 0, app.StopField.Value = round(max(t)); end
            if app.LenField.Value == 0, app.LenField.Value = round(max(t)/2); end
            app.drawSlice();
        end

        function w = currentWindow(app)
            if isempty(app.Signals), w = []; return; end
            dur = numel(app.Signals{app.selectedIndex()}) / app.Fs;
            switch app.ModeDrop.Value
                case 'range', w = [max(0, app.StartField.Value), min(dur, app.StopField.Value)];
                case 'first', w = [0, min(dur, app.LenField.Value)];
                case 'final', w = [max(0, dur - app.LenField.Value), dur];
                otherwise, w = [];
            end
        end

        function drawSlice(app)
            if ~isempty(app.SlicePatch) && isvalid(app.SlicePatch), delete(app.SlicePatch); end
            app.SlicePatch = [];
            w = app.currentWindow();
            if isempty(w) || w(2) <= w(1), return; end
            yl = app.PreviewAxes.YLim;
            app.SlicePatch = patch(app.PreviewAxes, ...
                [w(1) w(2) w(2) w(1)], [yl(1) yl(1) yl(2) yl(2)], ...
                [0.91 0.58 0.18], 'FaceAlpha', 0.22, 'EdgeColor', 'none');
            uistack(app.SlicePatch, 'bottom');
        end

        function modeChanged(app)
            m = app.ModeDrop.Value;
            isRange = strcmp(m, 'range');
            isLen = any(strcmp(m, {'first','final'}));
            app.StartField.Enable = local_onoff(isRange);
            app.StopField.Enable = local_onoff(isRange);
            app.LenField.Enable = local_onoff(isLen);
            app.drawSlice();
        end

        function fsChanged(app)
            app.Fs = app.FsField.Value;
            app.populateDecim();
            app.previewSelected();
        end

        function populateDecim(app)
            fs = app.FsField.Value;
            items = {}; data = {};
            for k = 1:32
                r = fs / k;
                if r < 0.5 && k > 1, break; end
                if k == 1, items{end+1} = '1× — no change';
                else, items{end+1} = sprintf('%d× → %.2f Hz', k, r); end %#ok<AGROW>
                data{end+1} = k; %#ok<AGROW>
            end
            app.DecimDrop.Items = items;
            app.DecimDrop.ItemsData = data;
        end

        function applyProcessing(app)
            if isempty(app.Signals)
                uialert(app.UIFigure, 'Load a file first.', 'Preprocessing'); return;
            end
            app.Fs = app.FsField.Value;
            k = app.DecimDrop.Value;
            app.Processed = cell(1, numel(app.Signals));
            try
                for i = 1:numel(app.Signals)
                    [y, fsNew] = cropAndDecimate(app.Signals{i}, app.Fs, app.ModeDrop.Value, ...
                        'StartS', app.StartField.Value, 'StopS', app.StopField.Value, ...
                        'LengthS', app.LenField.Value, 'DecimateFactor', k);
                    app.Processed{i} = y;
                    app.ProcessedFs = fsNew;
                end
            catch ME
                uialert(app.UIFigure, ME.message, 'Preprocessing error'); return;
            end
            % preview the processed version of the selected file
            y = app.Processed{app.selectedIndex()};
            t = (0:numel(y)-1) / app.ProcessedFs;
            plot(app.PreviewAxes, t, y, 'Color', [0.76 0.31 0.18]);
            app.PreviewAxes.XLim = [0, max(t)];
            title(app.PreviewAxes, sprintf('Processed — %d samples @ %.3g Hz (%.2f s)', ...
                numel(y), app.ProcessedFs, numel(y)/app.ProcessedFs));
            app.SlicePatch = [];
            app.StatusLabel.Text = sprintf(['Applied to %d file(s). New rate %.3g Hz. ' ...
                'Use "Save cropped…" to export.'], numel(app.Processed), app.ProcessedFs);
        end

        function saveProcessed(app)
            if isempty(app.Processed)
                uialert(app.UIFigure, 'Apply preprocessing first.', 'Preprocessing'); return;
            end
            [fn, pth] = uiputfile('*.mat', 'Save preprocessed signals');
            if isequal(fn, 0), return; end
            data = app.Processed; %#ok<NASGU>
            fs = app.ProcessedFs; %#ok<NASGU>
            names = app.FileNames; %#ok<NASGU>
            save(fullfile(pth, fn), 'data', 'fs', 'names');
            app.StatusLabel.Text = sprintf('Saved %d signal(s) to %s.', numel(app.Processed), fn);
        end
    end
end

function v = local_largestNumeric(s)
    v = []; best = -1; f = fieldnames(s);
    for k = 1:numel(f)
        c = s.(f{k});
        if isnumeric(c) && numel(c) > best, v = c; best = numel(c); end
    end
end

function s = local_onoff(tf)
    if tf, s = 'on'; else, s = 'off'; end
end
