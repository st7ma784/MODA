% CoherenceMulti — App Designer migration
% Wavelet phase coherence for signal pairs.
% Compatible with MATLAB R2023a through R2026a.

classdef CoherenceMulti < matlab.apps.AppBase

    %% UI component properties
    properties (Access = public)
        UIFigure            matlab.ui.Figure
        RootContainer   % parent for built components: UIFigure (standalone) or a uitab (embedded)
        OwnsFigure = true   % false when embedded into a shell app's uitab

        % Menus
        FileMenu            matlab.ui.container.Menu
        LoadMenu            matlab.ui.container.Menu
        SaveAvgCsvMenu      matlab.ui.container.Menu
        SaveAvgMatMenu      matlab.ui.container.Menu
        SaveSessionMenu     matlab.ui.container.Menu
        LoadSessionMenu     matlab.ui.container.Menu
        ResetGUIMenu        matlab.ui.container.Menu
        PlotMenu            matlab.ui.container.Menu
        ExportViewMenu      matlab.ui.container.Menu
        OpenViewMenu        matlab.ui.container.Menu
        ExportReportMenu    matlab.ui.container.Menu

        % Logos
        logo                matlab.ui.control.Image
        nbmplogo            matlab.ui.control.Image

        % Time-series panel
        TimeSeriesPanel     matlab.ui.container.Panel
        time_series_1       matlab.ui.control.UIAxes
        time_series_2       matlab.ui.control.UIAxes

        % Main plot pane (overlapping axes)
        wt_pane             matlab.ui.container.Panel
        plot3d              matlab.ui.control.UIAxes
        plot_pow            matlab.ui.control.UIAxes
        cum_avg             matlab.ui.control.UIAxes

        % Signal list
        signal_list         matlab.ui.control.ListBox

        % Status / info
        status              matlab.ui.control.EditField
        signal_length       matlab.ui.control.EditField

        % Buttons
        wavlet_transform    matlab.ui.control.Button
        wt_single           matlab.ui.control.Button
        open_file_btn       matlab.ui.control.Button
        save_preset_btn     matlab.ui.control.Button
        load_preset_btn     matlab.ui.control.Button
        refresh_limits      matlab.ui.control.Button
        supdate             matlab.ui.control.Button

        % WPC parameters
        max_freq            matlab.ui.control.EditField
        min_freq            matlab.ui.control.EditField
        central_freq        matlab.ui.control.EditField
        wavelet_type        matlab.ui.control.DropDown
        preprocess          matlab.ui.control.DropDown
        cutedges            matlab.ui.control.DropDown

        % Surrogate parameters (hidden by default; see enableSurrogatesChanged)
        enableSurrogatesCheckbox matlab.ui.control.CheckBox
        surrogateControls   cell
        surrogate_count     matlab.ui.control.EditField
        surrogate_type      matlab.ui.control.DropDown
        surrogate_analysis  matlab.ui.control.DropDown
        surrogate_percentile matlab.ui.control.EditField
        subtract_surrogates matlab.ui.control.CheckBox

        % Limits
        xlim                matlab.ui.control.EditField
        ylim                matlab.ui.control.EditField
        length              matlab.ui.control.EditField

        % Intervals
        intervals           matlab.ui.control.EditField

        % Plot type radio group
        plot_type_bg        matlab.ui.container.ButtonGroup
        power_rb            matlab.ui.control.RadioButton
        amp_rb              matlab.ui.control.RadioButton
    end

    %% Data properties (replaces handles struct)
    properties (Access = public)
        % Signal data
        sig             = []
        sig_cut         = []
        time_axis       = []
        time_axis_cut   = []
        time_axis_ds    = []
        sampling_freq   = NaN
        it              = 0   % load counter (MODAreadcheck: confirm re-load)

        % Results
        freqarr         = []
        wopt            = []
        TPC             = {}
        time_avg_wpc    = {}
        TPC_surr_avg_arr = {}
        TPC_surr_avg_max = {}
        surrogates      = {}
        thresh          = 1
        nscalc          = 0
        stype           = ''
        currsig         = []

        % Appearance
        cmap            = []
        linecol         = []
        line2width      = 2
        plot_type       = 1     % 1=power, 2=amp
        leg1            = {}
        leg             = {}
        h_wait          = []
    end

    %% Helper methods
    methods (Access = private)

        function idx = listboxIndex(~, lb)
            % Return 1-based numeric index of current listbox selection.
            items = lb.Items;
            val   = lb.Value;
            if ischar(val) || isstring(val)
                val = {char(val)};
            end
            idx = find(strcmp(items, val{1}), 1);
            if isempty(idx); idx = 1; end
        end

        function idx = dropdownIndex(~, dd)
            items = dd.Items;
            val   = dd.Value;
            idx   = find(strcmp(items, val), 1);
            if isempty(idx); idx = 1; end
        end

        function showPlotMode(app, mode)
            % mode 1 = TF single/pair  (plot3d + plot_pow, hide cum_avg)
            % mode 2 = average (cum_avg only)
            if mode == 1
                app.plot3d.Visible   = 'on';
                app.plot_pow.Visible = 'on';
                app.cum_avg.Visible  = 'off';
            else
                app.plot3d.Visible   = 'off';
                app.plot_pow.Visible = 'off';
                app.cum_avg.Visible  = 'on';
            end
        end

        function setStatus(app, msg)
            app.status.Value = msg;
            drawnow;
        end

        function initSettings(app)
            % Logos are already loaded via uiimage at creation time (see
            % anchorBrandingLogos), so handles.logo/nbmplogo are
            % deliberately omitted here — MODAsettings' logo-loading
            % section is guarded to skip cleanly when those fields are absent.
            handles = MODAsettings([], struct());
            app.cmap     = handles.cmap;
            app.linecol  = handles.linecol;
            app.line2width = handles.line2width;
        end

        function anchorBrandingLogos(app)
            % Consistent branding placement across every module screen —
            % identical Position/size in every module file, see
            % TimeFrequencyAnalysis.m's anchorBrandingLogos for the full
            % rationale (uiimage avoids the stretch/warp uiaxes+image() had).
            W = 1600; H = 860;
            bg = app.UIFigure.Color;

            app.nbmplogo = uiimage(app.RootContainer,'Position',[W-370 H-65 360 55]);
            app.nbmplogo.ScaleMethod = 'fit';
            app.nbmplogo.BackgroundColor = bg;
            imgPath = which('MODAbanner5.png');
            if ~isempty(imgPath), app.nbmplogo.ImageSource = imgPath; end

            app.logo = uiimage(app.RootContainer,'Position',[W-140 10 130 55]);
            app.logo.ScaleMethod = 'fit';
            app.logo.BackgroundColor = bg;
            imgPath = which('physicslogo.png');
            if ~isempty(imgPath), app.logo.ImageSource = imgPath; end
        end
    end

    %% Callbacks
    methods (Access = private)

        function startupFcn(app)
            initSettings(app);
            app.plot_type = 1;

            % Disable menus and buttons that need data (menus only exist
            % when this module owns its figure — see createComponents)
            if app.OwnsFigure
                app.SaveAvgCsvMenu.Enable  = 'off';
                app.SaveAvgMatMenu.Enable  = 'off';
                app.ExportViewMenu.Enable  = 'off';
                app.OpenViewMenu.Enable    = 'off';
                app.ExportReportMenu.Enable = 'off';
            end

            % Axes initial visibility
            showPlotMode(app, 1);
            app.status.Value = 'Please Import Signal';
        end

        %------------------------------------------------------------------
        function fileReadMenuSelected(app, ~)
            % Load data via MODAreadcheck / MODAread
            handles.cmap         = app.cmap;
            handles.linecol      = app.linecol;
            handles.sampling_freq = app.sampling_freq;
            [handles, A] = MODAreadcheck(handles);
            if A ~= 1; return; end

            % Clear old results
            cla(app.plot3d,  'reset');
            cla(app.plot_pow,'reset');
            cla(app.cum_avg, 'reset');
            cla(app.time_series_1,'reset');
            cla(app.time_series_2,'reset');
            showPlotMode(app, 1);
            app.plot3d.Visible  = 'on';
            app.plot_pow.Visible = 'on';
            app.cum_avg.Visible  = 'off';

            app.freqarr      = [];
            app.sig          = [];
            app.time_avg_wpc = {};
            app.leg1         = {};
            app.TPC          = {};
            app.TPC_surr_avg_arr = {};
            app.TPC_surr_avg_max = {};
            app.surrogates   = {};
            app.wopt         = [];
            app.freqarr      = [];

            app.signal_list.Value = app.signal_list.Items{1};

            [handles, sig] = MODAread(handles, 1, "even");
            if isequal(sig, 0); return; end

            app.sig          = handles.sig;
            app.sampling_freq = handles.sampling_freq;
            app.time_axis    = handles.time_axis;

            n = size(sig, 1) / 2;
            list = cell(n + 1, 1);
            for i = 1:n
                list{i} = sprintf('Signal Pair %d', i);
            end
            list{n+1} = 'Average Plot (All)';
            app.signal_list.Items = list;
            app.signal_list.Value = list{1};

            linkaxes([app.time_series_1 app.time_series_2], 'x');
            plot(app.time_series_1, app.time_axis, app.sig(1,:),   'color', app.linecol(1,:));
            plot(app.time_series_2, app.time_axis, app.sig(1+n,:), 'color', app.linecol(1,:));
            xlim(app.time_series_1, [0, size(sig,2)/app.sampling_freq]);
            xlim(app.time_series_2, [0, size(sig,2)/app.sampling_freq]);
            ylabel(app.time_series_1, 'Sig 1');
            ylabel(app.time_series_2, 'Sig 2');
            xlabel(app.time_series_2, 'Time (s)');

            refreshLimitsCallback(app, []);
            setStatus(app, 'Select Data And Continue With Wavelet Transform');
            app.signal_length.Value = sprintf('%g minutes', size(sig,2)/app.sampling_freq/60);
        end

        %------------------------------------------------------------------
        function refreshLimitsCallback(app, ~)
            x = app.time_series_1.XLim;
            y = app.time_series_1.YLim;
            t = x(2) - x(1);
            app.xlim.Value   = sprintf('%g, %g', x(1), x(2));
            app.ylim.Value   = sprintf('%g, %g', y(1), y(2));
            app.length.Value = sprintf('%g', t);

            if isempty(app.sig); return; end
            fs = app.sampling_freq;
            xi = round(x .* fs);
            xi(2) = min(xi(2), size(app.sig, 2));
            xi(1) = max(xi(1), 1);
            app.sig_cut      = app.sig(:, xi(1):xi(2));
            app.time_axis_cut = app.time_axis(xi(1):xi(2));
        end

        %------------------------------------------------------------------
        function xlimFieldChanged(app, ~)
            xl = csv_to_mvar(app.xlim.Value);
            if numel(xl) < 2; return; end
            xlim(app.time_series_1, xl);
            xlim(app.time_series_2, xl);
            app.length.Value = sprintf('%g', xl(2)-xl(1));
        end

        function ylimFieldChanged(app, ~)
            yl = csv_to_mvar(app.ylim.Value);
            if numel(yl) < 2; return; end
            ylim(app.time_series_1, yl);
            ylim(app.time_series_2, yl);
        end

        %------------------------------------------------------------------
        function signalListChanged(app, ~)
            if isempty(app.sig); return; end
            n = size(app.sig, 1) / 2;
            sel = listboxIndex(app, app.signal_list);
            if sel == n + 1
                % Average selected — show all pairs, trigger xyplot
                xyplotCallback(app, []);
                return;
            end
            % Single pair
            xl = csv_to_mvar(app.xlim.Value);
            plot(app.time_series_1, app.time_axis, app.sig(sel,:), 'color', app.linecol(1,:));
            plot(app.time_series_2, app.time_axis, app.sig(sel+n,:), 'color', app.linecol(1,:));
            xlim(app.time_series_1, xl);
            xlim(app.time_series_2, xl);
            ylabel(app.time_series_1, 'Sig 1');
            ylabel(app.time_series_2, 'Sig 2');
            xlabel(app.time_series_2, 'Time (s)');
            refreshLimitsCallback(app, []);
            setStatus(app, 'Select Data And Continue With Wavelet Transform');
            if ~isempty(app.TPC)
                xyplotCallback(app, []);
            end
            intervalsCallback(app, []);
        end

        %------------------------------------------------------------------
        function waveletTransformButtonPushed(app, ~)
            app.currsig = [];
            try
                app.wavlet_transform.Enable = 'off';
                app.wt_single.Enable        = 'off';
                doCoherenceCalc(app, 1);
                xyplotCallback(app, []);
                intervalsCallback(app, []);
            catch e
                errordlg(e.message, 'Error');
                app.wt_single.Enable        = 'on';
                app.wavlet_transform.Enable = 'on';
                rethrow(e);
            end
        end

        % ---- Analysis presets (save/load parameter values only) -----
        function savePresetButtonPushed(app, ~)
            [fname, fpath] = uiputfile('*.mat', 'Save Coherence preset as...', 'coherence_preset.mat');
            if isequal(fname, 0), return; end
            params = struct('max_freq', app.max_freq.Value, 'min_freq', app.min_freq.Value, ...
                'central_freq', app.central_freq.Value, 'wavelet_type', app.wavelet_type.Value, ...
                'preprocess', app.preprocess.Value, 'cutedges', app.cutedges.Value);
            ok = savePreset(fullfile(fpath, fname), 'CoherenceMulti', params);
            if ok
                app.status.Value = ['Preset saved: ', fname];
            else
                uialert(app.UIFigure, 'Failed to save preset.', 'Save Preset Error');
            end
        end

        function loadPresetButtonPushed(app, ~)
            [fname, fpath] = uigetfile('*.mat', 'Load Coherence preset...');
            if isequal(fname, 0), return; end
            [params, savedModule, ok] = loadPreset(fullfile(fpath, fname));
            if ~ok
                uialert(app.UIFigure, 'Selected file is not a valid MODA preset.', 'Load Preset Error');
                return;
            end
            if ~strcmpi(savedModule, 'CoherenceMulti')
                uialert(app.UIFigure, sprintf('This preset was saved from "%s" — applying it anyway, but some fields may not match.', savedModule), ...
                    'Preset From Different Module', 'Icon', 'warning');
            end
            if isfield(params,'max_freq'), app.max_freq.Value = params.max_freq; end
            if isfield(params,'min_freq'), app.min_freq.Value = params.min_freq; end
            if isfield(params,'central_freq'), app.central_freq.Value = params.central_freq; end
            if isfield(params,'wavelet_type'), app.wavelet_type.Value = params.wavelet_type; end
            if isfield(params,'preprocess'), app.preprocess.Value = params.preprocess; end
            if isfield(params,'cutedges'), app.cutedges.Value = params.cutedges; end
            app.status.Value = ['Preset loaded: ', fname];
        end

        function wtSingleButtonPushed(app, ~)
            app.currsig = listboxIndex(app, app.signal_list);
            try
                app.wavlet_transform.Enable = 'off';
                app.wt_single.Enable        = 'off';
                doCoherenceCalc(app, 0);
                xyplotCallback(app, []);
                intervalsCallback(app, []);
            catch e
                errordlg(e.message, 'Error');
                app.wt_single.Enable        = 'on';
                app.wavlet_transform.Enable = 'on';
                rethrow(e);
            end
        end

        function doCoherenceCalc(app, calcAll)
            % Core WPC calculation (replaces MODAwpc).
            fmax = str2double(app.max_freq.Value);
            fmin = str2double(app.min_freq.Value);
            fc   = str2double(app.central_freq.Value);
            fs   = app.sampling_freq;

            if isnan(fmax); fmax = fs/2; end

            wtypeIdx = dropdownIndex(app, app.wavelet_type);
            wtype    = app.wavelet_type.Items{wtypeIdx};
            ppIdx    = dropdownIndex(app, app.preprocess);
            ppselect = app.preprocess.Items{ppIdx};
            cutIdx   = dropdownIndex(app, app.cutedges);
            cutselect = app.cutedges.Items{cutIdx};

            ns       = floor(str2double(app.surrogate_count.Value));
            stypeIdx = dropdownIndex(app, app.surrogate_type);
            stype_str = app.surrogate_type.Items{stypeIdx};
            if stypeIdx == 1; stype_str = 'RP'; end
            app.stype = stype_str;

            if fmax > fs/2
                errordlg(['Max freq must be ≤ Nyquist (', num2str(fs/2), ' Hz).'], 'Parameter Error');
                return;
            end
            if isnan(fs)
                errordlg('Sampling frequency must be specified.', 'Parameter Error');
                return;
            end
            if isempty(app.sig)
                errordlg('Signal not found.', 'Signal Error');
                return;
            end

            n    = size(app.sig_cut, 1) / 2;
            app.nscalc = ns;

            if length(app.sig_cut) >= 2000
                screensize   = max(get(groot,'Screensize'));
                under_sample = floor(size(app.sig_cut, 2) / screensize);
            else
                under_sample = 1;
            end
            app.time_axis_ds = app.time_axis_cut(1:under_sample:end);

            if calcAll
                inds = 1:n;
            else
                inds = listboxIndex(app, app.signal_list);
            end

            app.time_avg_wpc     = cell(size(app.sig_cut, 1), 1);
            app.TPC_surr_avg_arr = cell(ns, size(app.sig_cut, 1) / 2);
            app.surrogates       = cell(size(app.sig_cut, 1) / 2, 1);

            app.h_wait = waitbar(0, 'Calculating coherence...', ...
                'CreateCancelBtn', 'setappdata(gcbf,''canceling'',1)');
            setappdata(app.h_wait, 'canceling', 0);

            % Per-pair (and per-surrogate) computation is pushed out to
            % coherencePairWorker.m and run via parfor across signal pairs
            % when Parallel Computing Toolbox is available, falling back to
            % a plain serial loop otherwise (or if anything about setting
            % up the parallel pool/DataQueue fails). A local (non-handle)
            % copy of the signal matrix is used inside the loop so parfor
            % workers never need to broadcast the whole app object.
            %
            % Trade-off versus the previous serial loop: that loop polled
            % the waitbar's cancel button between every pair AND every
            % surrogate, so cancelling stopped work almost immediately.
            % parfor workers cannot touch this figure's waitbar/appdata at
            % all, so cancellation can now only be checked once, before the
            % whole batch of pairs starts — once running, the batch runs to
            % completion. Live progress (which pair just finished) is still
            % reported during the batch via a DataQueue, which workers CAN
            % safely send() through.
            if getappdata(app.h_wait, 'canceling')
                delete(app.h_wait);
                setStatus(app, 'Calculation interrupted by user');
                app.wt_single.Enable        = 'on';
                app.wavlet_transform.Enable = 'on';
                return;
            end

            sigCutLocal = app.sig_cut; % plain matrix copy; avoids broadcasting the handle-class app into parfor workers
            nInds = numel(inds);
            resultsTPC    = cell(nInds, 1);
            resultsAvg    = cell(nInds, 1);
            resultsSurr   = cell(nInds, 1);
            resultsSurrPC = cell(ns, nInds);
            resultsFreq   = cell(nInds, 1);
            resultsWopt   = cell(nInds, 1);

            setStatus(app, sprintf('Calculating WPC for %d signal pair(s)...', nInds));

            useParfor = false;
            try
                useParfor = license('test', 'Distrib_Computing_Toolbox') && ~isempty(ver('parallel'));
            catch
                useParfor = false;
            end

            completed = 0;
            try
                if useParfor
                    dq = parallel.pool.DataQueue;
                    afterEach(dq, @(idx) updateWaitbarSafe(app.h_wait, idx, nInds, 'Calculating WPC'));
                    parfor idx = 1:nInds
                        p = inds(idx); %#ok<PFBNS>
                        [resultsTPC{idx}, resultsAvg{idx}, resultsSurr{idx}, resultsSurrPC(:,idx), resultsFreq{idx}, resultsWopt{idx}] = ...
                            coherencePairWorker(sigCutLocal(p,:), sigCutLocal(p+n,:), fs, fc, fmin, fmax, wtype, cutselect, ppselect, ns, stype_str, under_sample);
                        send(dq, idx);
                    end
                else
                    error('MODA:noParallelToolbox', 'fall through to serial path below');
                end
            catch
                % Either Parallel Computing Toolbox isn't available, or
                % something about starting the pool/DataQueue failed —
                % fall back to the plain serial loop, still with live
                % waitbar progress per pair.
                for idx = 1:nInds
                    p = inds(idx);
                    [resultsTPC{idx}, resultsAvg{idx}, resultsSurr{idx}, resultsSurrPC(:,idx), resultsFreq{idx}, resultsWopt{idx}] = ...
                        coherencePairWorker(sigCutLocal(p,:), sigCutLocal(p+n,:), fs, fc, fmin, fmax, wtype, cutselect, ppselect, ns, stype_str, under_sample);
                    updateWaitbarSafe(app.h_wait, idx, nInds, 'Calculating WPC');
                end
            end

            % Assign results back onto the handle-class app on the client,
            % indexed by the ORIGINAL pair index p (not idx), matching what
            % the previous serial loop wrote directly.
            for idx = 1:nInds
                p = inds(idx);
                app.TPC{p,1}          = resultsTPC{idx};
                app.time_avg_wpc{p,1} = resultsAvg{idx};
                if ns > 1
                    app.surrogates{p,1} = resultsSurr{idx};
                    for k = 1:ns
                        app.TPC_surr_avg_arr{k,p} = resultsSurrPC{k,idx};
                    end
                end
            end
            if nInds > 0
                app.freqarr = resultsFreq{end};
                app.wopt    = resultsWopt{end};
                completed = 1;
            end

            if ishandle(app.h_wait); delete(app.h_wait); end

            if completed
                surr_analysis = dropdownIndex(app, app.surrogate_analysis);
                alph          = str2double(app.surrogate_percentile.Value);
                app.thresh    = surr_analysis;
                app.TPC_surr_avg_max = cell(size(app.sig_cut,1)/2, 1);

                if calcAll
                    loop_inds = 1:size(app.sig_cut,1)/2;
                else
                    loop_inds = inds;
                end
                for i = loop_inds
                    if ns > 1
                        if calcAll
                            t = cell2mat(app.TPC_surr_avg_arr);
                            t = t(:, length(app.freqarr)*(i-1)+1 : length(app.freqarr)*i);
                        else
                            t = cell2mat(app.TPC_surr_avg_arr);
                            t = t(:, 1:length(app.freqarr));
                        end
                        if surr_analysis == 2
                            if floor((ns+1)*alph) == 0
                                app.TPC_surr_avg_max{i,1} = max(t);
                            else
                                K = floor((ns+1)*alph);
                                s1 = sort(t, 'descend');
                                app.TPC_surr_avg_max{i,1} = s1(K,:);
                            end
                        elseif surr_analysis == 1 && ns > 1
                            app.TPC_surr_avg_max{i,1} = max(t);
                        end
                    end
                end

                app.intervals.Enable         = 'on';
                app.wt_single.Enable         = 'on';
                app.wavlet_transform.Enable  = 'on';
                if app.OwnsFigure
                    app.ExportViewMenu.Enable    = 'on';
                    app.OpenViewMenu.Enable      = 'on';
                    app.ExportReportMenu.Enable  = 'on';
                    app.SaveAvgCsvMenu.Enable    = 'on';
                    app.SaveAvgMatMenu.Enable    = 'on';
                end
                if ns > 1
                    app.subtract_surrogates.Enable = 'on';
                end
                setStatus(app, 'Calculation complete');
            else
                setStatus(app, 'Calculation interrupted by user');
                app.wt_single.Enable        = 'on';
                app.wavlet_transform.Enable = 'on';
            end
        end

        %------------------------------------------------------------------
        function xyplotCallback(app, ~)
            if isempty(app.sig); return; end
            n = size(app.sig, 1) / 2;
            signal_selected = listboxIndex(app, app.signal_list);
            gfs = 12;

            if isempty(app.time_avg_wpc); return; end
            if signal_selected <= n && isempty(app.time_avg_wpc{signal_selected,1}); return; end

            if signal_selected == n + 1 && ~isempty(app.freqarr)
                % Average plot
                showPlotMode(app, 2);
                cla(app.cum_avg, 'reset');
                hold(app.cum_avg, 'on');

                if n > 1
                    plot(app.cum_avg, app.freqarr, mean(cell2mat(app.time_avg_wpc)), '-',  'Linewidth', 3, 'color', app.linecol(1,:));
                    plot(app.cum_avg, app.freqarr, median(cell2mat(app.time_avg_wpc)), '--','Linewidth', 3, 'color', app.linecol(2,:));
                else
                    d = cell2mat(app.time_avg_wpc);
                    plot(app.cum_avg, app.freqarr, d, '-',  'Linewidth', 3, 'color', app.linecol(1,:));
                    plot(app.cum_avg, app.freqarr, d, '--', 'Linewidth', 3, 'color', app.linecol(2,:));
                end
                app.leg1 = {'Mean','Median'};
                ylabel(app.cum_avg, 'Overall Coherence', 'FontUnits','Points','FontSize',gfs);
                xlabel(app.cum_avg, 'Frequency (Hz)',    'FontUnits','Points','FontSize',gfs);
                legend(app.cum_avg, app.leg1);
                set(app.cum_avg, 'xscale','log');
                idx_first = find(sum(~isnan(app.time_avg_wpc{1,1}),1) > 0, 1, 'first');
                idx_last  = find(sum(~isnan(app.time_avg_wpc{1,1}),1) > 0, 1, 'last');
                if ~isempty(idx_first)
                    xlim(app.cum_avg, [app.freqarr(idx_first) app.freqarr(idx_last)]);
                end
                grid(app.cum_avg, 'off');
                box(app.cum_avg, 'on');

            elseif ~isempty(app.freqarr)
                % Single-pair TF plot
                showPlotMode(app, 1);
                cla(app.plot3d,  'reset');
                cla(app.plot_pow,'reset');

                pcolor(app.plot3d, app.time_axis_ds, app.freqarr, app.TPC{signal_selected,1});
                colormap(app.plot3d, app.cmap);
                shading(app.plot3d, 'interp');
                set(app.plot3d, 'yscale','log');
                xlabel(app.plot3d, 'Time (s)',      'FontUnits','Points','FontSize',gfs);
                ylabel(app.plot3d, 'Frequency (Hz)','FontUnits','Points','FontSize',gfs);

                plot(app.plot_pow, app.time_avg_wpc{signal_selected,1}, app.freqarr, 'LineWidth', 2, 'color', app.linecol(1,:));
                hold(app.plot_pow, 'on');
                app.leg = {'Original Signal'};
                if ~isempty(app.TPC_surr_avg_max) && ~isempty(app.TPC_surr_avg_max{signal_selected,1})
                    plot(app.plot_pow, app.TPC_surr_avg_max{signal_selected,1}, app.freqarr, 'LineWidth',2,'color',app.linecol(2,:));
                    app.leg = {'Original Signal','Surrogate'};
                end
                hold(app.plot_pow,'off');
                set(app.plot_pow, 'yscale','log');
                xlabel(app.plot_pow, 'Overall Coherence', 'FontUnits','Points','FontSize',gfs);
                ylabel(app.plot_pow, 'Frequency (Hz)',    'FontUnits','Points','FontSize',gfs);
                legend(app.plot_pow, app.leg);

                idx_first = find(sum(~isnan(app.time_avg_wpc{signal_selected,1}),1) > 0, 1, 'first');
                idx_last  = find(sum(~isnan(app.time_avg_wpc{signal_selected,1}),1) > 0, 1, 'last');
                if ~isempty(idx_first)
                    ylim(app.plot3d,  [app.freqarr(idx_first) app.freqarr(idx_last)]);
                    ylim(app.plot_pow,[app.freqarr(idx_first) app.freqarr(idx_last)]);
                    xlim(app.plot3d,  [app.time_axis_ds(1) app.time_axis_ds(end)]);
                end
                grid(app.plot3d,  'on');
                grid(app.plot_pow,'off');
                setStatus(app, 'Done Plotting');
            end
        end

        %------------------------------------------------------------------
        function intervalsCallback(app, ~)
            intervals = csv_to_mvar(app.intervals.Value);
            if isempty(intervals); return; end
            intervals = sort(intervals);

            signal_selected = listboxIndex(app, app.signal_list);
            n = size(app.sig, 1) / 2;
            if signal_selected == n + 1
                ax = app.cum_avg;
                hold(ax, 'on');
                xl = ax.YLim;
                for j = 1:numel(intervals)
                    x = [xl(1) xl(2)];
                    z = ones(1,2);
                    y = intervals(j) * ones(1,2);
                    plot3(ax, y, x, z, '--k');
                end
            else
                for ax = [app.plot3d app.plot_pow]
                    if strcmp(ax.Visible, 'on')
                        hold(ax, 'on');
                        xl = ax.XLim;
                        for j = 1:numel(intervals)
                            y = [intervals(j) intervals(j)];
                            plot(ax, xl, y, '--k');
                        end
                        hold(ax, 'off');
                    end
                end
            end
        end

        %------------------------------------------------------------------
        function enableSurrogatesChanged(app, ~)
            vis = 'off';
            if app.enableSurrogatesCheckbox.Value, vis = 'on'; end
            for c = app.surrogateControls, c{1}.Visible = vis; end
        end

        %------------------------------------------------------------------
        function subtractSurrogatesChanged(app, ~)
            if isempty(app.time_avg_wpc); return; end
            sel = listboxIndex(app, app.signal_list);
            n   = size(app.sig,1)/2;
            if sel == n+1; return; end
            gfs = 12;
            cla(app.plot_pow);
            hold(app.plot_pow,'on');
            if app.subtract_surrogates.Value && ~isempty(app.TPC_surr_avg_max) && ~isempty(app.TPC_surr_avg_max{sel,1})
                cc = subplus(app.time_avg_wpc{sel,1} - app.TPC_surr_avg_max{sel,1});
                plot(app.plot_pow, cc, app.freqarr, 'LineWidth',2,'color',app.linecol(1,:));
                app.leg = {'Surrogate Subtracted'};
            else
                plot(app.plot_pow, app.time_avg_wpc{sel,1}, app.freqarr, 'LineWidth',2,'color',app.linecol(1,:));
                app.leg = {'Original Signal'};
                if ~isempty(app.TPC_surr_avg_max) && ~isempty(app.TPC_surr_avg_max{sel,1})
                    plot(app.plot_pow, app.TPC_surr_avg_max{sel,1}, app.freqarr, 'LineWidth',2,'color',app.linecol(2,:));
                    app.leg = {'Original Signal','Surrogate'};
                end
            end
            hold(app.plot_pow,'off');
            set(app.plot_pow,'yscale','log');
            idx_first = find(sum(~isnan(app.time_avg_wpc{sel,1}),1)>0,1,'first');
            idx_last  = find(sum(~isnan(app.time_avg_wpc{sel,1}),1)>0,1,'last');
            if ~isempty(idx_first)
                ylim(app.plot_pow,[app.freqarr(idx_first) app.freqarr(idx_last)]);
            end
            xlabel(app.plot_pow,'Overall Coherence','FontUnits','Points','FontSize',gfs);
            ylabel(app.plot_pow,'Frequency (Hz)',    'FontUnits','Points','FontSize',gfs);
            legend(app.plot_pow, app.leg);
            grid(app.plot_pow,'off');
        end

        %------------------------------------------------------------------
        function plotTypeChanged(app, ~)
            if app.power_rb.Value
                app.plot_type = 1;
            else
                app.plot_type = 2;
            end
        end

        %------------------------------------------------------------------
        % Save/Plot menu callbacks
        function tf = isAverageView(app)
            n = size(app.sig, 1) / 2;
            tf = listboxIndex(app, app.signal_list) == n + 1 && ~isempty(app.freqarr);
        end

        function fig = buildViewFigure(app)
            % Builds a hidden figure reproducing whichever result view is
            % currently showing: the single-pair TF+coherence pair, or the
            % all-pair mean/median coherence plot.
            fig = figure('Visible','off');
            if ~app.isAverageView()
                ax1 = copyobj(app.plot3d,  fig);
                ax2 = copyobj(app.plot_pow, fig);
                h = colorbar(ax1); ylabel(h,'Wavelet coherence');
                colormap(fig, app.cmap);
                set(ax1,'Units','normalized','Position',[0.07,0.2,.55,.7]);
                set(ax2,'Units','normalized','Position',[0.8, 0.2,.18,.7],'YTickMode','auto','YTickLabelMode','auto');
                set(fig,'Units','normalized','Position',[0.2 0.2 0.6 0.5]);
            else
                ax = copyobj(app.cum_avg, fig);
                set(ax,'Units','normalized','Position',[0.1,0.2,.85,.7]);
                set(fig,'Units','normalized','Position',[0.2 0.2 0.5 0.5]);
                legend(ax, app.leg1);
            end
        end

        function exportViewMenuSelected(app, ~)
            [FileName,PathName] = uiputfile({'*.png';'*.pdf';'*.fig'}, 'Export current view as');
            if isequal(FileName,0), return; end
            fig = app.buildViewFigure();
            try
                dest = fullfile(PathName,FileName);
                if endsWith(FileName,'.fig')
                    savefig(fig, dest);
                else
                    exportgraphics(fig, dest);
                end
            catch e
                delete(fig); errordlg(e.message,'Error'); rethrow(e);
            end
            delete(fig);
        end

        function openViewMenuSelected(app, ~)
            fig = app.buildViewFigure();
            fig.Visible = 'on';
        end

        function exportReportMenuSelected(app, ~)
            [FileName,PathName] = uiputfile('*.pdf', 'Export report as', 'coherence_report.pdf');
            if isequal(FileName,0), return; end
            fig = app.buildViewFigure();
            params = struct('max_freq', app.max_freq.Value, 'min_freq', app.min_freq.Value, ...
                'central_freq', app.central_freq.Value, 'wavelet_type', app.wavelet_type.Value, ...
                'preprocess', app.preprocess.Value, 'cutedges', app.cutedges.Value);
            ok = exportReportPDF(fullfile(PathName,FileName), fig, 'Wavelet Phase Coherence', params);
            delete(fig);
            if ~ok
                errordlg('Failed to export report.', 'Error');
            end
        end

        %------------------------------------------------------------------
        function saveAvgCsvMenuSelected(app, ~)
            try
                [FileName, PathName] = uiputfile('.csv','Save Coherence Data');
                if isequal(FileName,0); return; end
                save_location = [PathName, FileName];
                avg_coh = cell2mat(app.time_avg_wpc)';
                xl = csv_to_mvar(app.xlim.Value);
                L  = xl(2)*app.wopt.fs - xl(1)*app.wopt.fs;

                D.Coherence         = avg_coh;
                D.Frequency         = app.freqarr;
                D.Time              = linspace(xl(1),xl(2),L);
                D.Sampling_frequency = app.wopt.fs;
                D.fmax              = app.wopt.fmax;
                D.fmin              = app.wopt.fmin;
                D.fr                = app.wopt.f0;
                D.Preprocessing     = app.wopt.Preprocess;
                D.Cut_Edges         = app.wopt.CutEdges;
                D.Wavelet_type      = app.wopt.Wavelet;
                ns = str2double(app.surrogate_count.Value);
                if ns > 1
                    D.Surrogates         = cell2mat(app.TPC_surr_avg_max)';
                    D.Surrogate_type     = app.stype;
                    D.Surrogate_number   = app.surrogate_count.Value;
                    D.Surrogate_threshold = buildSurrThreshStr(app);
                end
                data = buildCsvData(app, D);
                cell2csv(save_location, data, ',');
            catch e
                errordlg(e.message,'Error'); rethrow(e);
            end
        end

        function saveAvgMatMenuSelected(app, ~)
            try
                [FileName, PathName] = uiputfile('.mat','Save Coherence Data');
                if isequal(FileName,0); return; end
                save_location = [PathName, FileName];
                avg_coh = cell2mat(app.time_avg_wpc)';
                xl = csv_to_mvar(app.xlim.Value);
                L  = xl(2)*app.wopt.fs - xl(1)*app.wopt.fs;

                Coherence_data.Coherence          = avg_coh;
                Coherence_data.Frequency          = app.freqarr;
                Coherence_data.Time               = linspace(xl(1),xl(2),L);
                Coherence_data.Sampling_frequency = app.wopt.fs;
                Coherence_data.fmax               = app.wopt.fmax;
                Coherence_data.fmin               = app.wopt.fmin;
                Coherence_data.fr                 = app.wopt.f0;
                Coherence_data.Preprocessing      = app.wopt.Preprocess;
                Coherence_data.Cut_Edges          = app.wopt.CutEdges;
                Coherence_data.Wavelet_type       = app.wopt.Wavelet;
                if ~isempty(app.currsig)
                    Coherence_data.Selected_sig   = app.currsig;
                end
                ns = str2double(app.surrogate_count.Value);
                if ns > 1
                    Coherence_data.Surrogates         = cell2mat(app.TPC_surr_avg_max)';
                    Coherence_data.Surrogate_type     = app.stype;
                    Coherence_data.Surrogate_number   = app.surrogate_count.Value;
                    Coherence_data.Surrogate_threshold = buildSurrThreshStr(app);
                end
                save(save_location, 'Coherence_data');
            catch e
                errordlg(e.message,'Error'); rethrow(e);
            end
        end

        function s = buildSurrThreshStr(app)
            if app.thresh == 2
                s = ['Significance ', app.surrogate_percentile.Value];
            else
                s = 'Maximum';
            end
        end

        function data = buildCsvData(app, D)
            L = length(D.Frequency);
            N = size(D.Coherence, 2);
            if isfield(D,'Surrogates')
                dstart = 18;
                data = cell(L+dstart, (N*2)+1);
                data{14,1} = 'Surrogate type';   data{14,2} = D.Surrogate_type;
                data{15,1} = 'Surrogate number'; data{15,2} = D.Surrogate_number;
                data{16,1} = 'Surrogate threshold'; data{16,2} = D.Surrogate_threshold;
            else
                dstart = 15;
                data = cell(L+dstart, N+1);
            end
            data{1,1}='Wavelet phase coherence toolbox'; data{2,1}=date; data{3,1}=[];
            data{4,1}='PARAMETERS';
            data{5,1}='Sampling frequency (Hz)'; data{5,2}=D.Sampling_frequency;
            data{6,1}='Maximum frequency (Hz)';  data{6,2}=D.fmax;
            data{7,1}='Minimum frequency (Hz)';  data{7,2}=D.fmin;
            data{8,1}='Central frequency';        data{8,2}=D.fr;
            data{9,1}='Preprocessing';             data{9,2}=D.Preprocessing;
            data{10,1}='Wavelet type';             data{10,2}=D.Wavelet_type;
            data{11,1}='Cut Edges';                data{11,2}=D.Cut_Edges;
            data{12,1}='Time start (s)';           data{12,2}=min(D.Time);
            data{13,1}='Time end (s)';             data{13,2}=max(D.Time);
            data{dstart,1}='Frequency';
            for l = 1:L; data{l+dstart,1} = D.Frequency(l); end
            if isfield(D,'Surrogates')
                for j = 1:N
                    data{dstart,j+1}   = ['Coherence ',  num2str(j)];
                    data{dstart,j+N+1} = ['Surrogate ',  num2str(j)];
                    for k = 1:L
                        data{k+dstart,j+1}   = D.Coherence(k,j);
                        data{k+dstart,j+N+1} = D.Surrogates(k,j);
                    end
                end
            else
                for j = 1:N
                    data{dstart,j+1} = ['Coherence ', num2str(j)];
                    for k = 1:L; data{k+dstart,j+1} = D.Coherence(k,j); end
                end
            end
        end

        %------------------------------------------------------------------
        function saveSessionMenuSelected(app, ~)
            handles.sig          = app.sig;
            handles.sig_cut      = app.sig_cut;
            handles.time_axis    = app.time_axis;
            handles.sampling_freq = app.sampling_freq;
            handles.freqarr      = app.freqarr;
            handles.wopt         = app.wopt;
            handles.TPC          = app.TPC;
            handles.time_avg_wpc = app.time_avg_wpc;
            MODAsave(handles);
        end

        function resetGUIMenuSelected(app, ~)
            delete(app);
            CoherenceMulti;
        end

        function supupdateButtonPushed(app, ~)
            % Save surrogate settings (legacy compatibility)
            stype = dropdownIndex(app, app.surrogate_type);
            nsurr = str2double(app.surrogate_count.Value);
            save('stype','stype','nsurr');
        end

    end  % private methods

    %% Component creation
    methods (Access = private)

        function createComponents(app, parentContainer)
            % parentContainer: optional. Omit for a standalone window
            % (legacy behavior); pass a uitab to build onto it instead.
            W = 1600; H = 860;

            if nargin < 2 || isempty(parentContainer)
                app.UIFigure = uifigure('Visible','off');
                app.UIFigure.Position = [0 0 W H];
                app.UIFigure.Resize = 'off';
                app.UIFigure.Name = 'MODA — Wavelet Phase Coherence';
                app.UIFigure.CloseRequestFcn = @(~,~) MODAclose(app.UIFigure, struct());
                app.OwnsFigure    = true;
                app.RootContainer = app.UIFigure;
            else
                app.RootContainer = parentContainer;
                app.UIFigure      = ancestor(parentContainer, 'figure');
                app.OwnsFigure    = false;
            end

            % Components below use absolute pixels on a WxH canvas; a
            % scrolling viewport keeps the top of that layout reachable in a
            % smaller window/tab. See attachScrollCanvas.
            [app.RootContainer, sidebarView] = attachScrollCanvas(app.RootContainer, W, H, 330);

            % Menu bar (figure-level; only when this module owns the figure)
            if app.OwnsFigure
                app.FileMenu = uimenu(app.UIFigure, 'Text','File');
                app.LoadMenu = uimenu(app.FileMenu, 'Text','Load Time Series', ...
                    'MenuSelectedFcn', @(s,e) fileReadMenuSelected(app,e));
                app.SaveAvgCsvMenu = uimenu(app.FileMenu,'Text','Save Coherence (.csv)', ...
                    'MenuSelectedFcn',@(s,e) saveAvgCsvMenuSelected(app,e));
                app.SaveAvgMatMenu = uimenu(app.FileMenu,'Text','Save Coherence (.mat)', ...
                    'MenuSelectedFcn',@(s,e) saveAvgMatMenuSelected(app,e));
                uimenu(app.FileMenu,'Separator','on');
                app.SaveSessionMenu = uimenu(app.FileMenu,'Text','Save Session', ...
                    'MenuSelectedFcn',@(s,e) saveSessionMenuSelected(app,e));
                app.LoadSessionMenu = uimenu(app.FileMenu,'Text','Load Session');
                app.ResetGUIMenu    = uimenu(app.FileMenu,'Text','New Workspace', ...
                    'MenuSelectedFcn',@(s,e) resetGUIMenuSelected(app,e));

                % Replaces the old 5-item Plot menu with two actions acting
                % on whichever view (single-pair or all-pair average) shows.
                app.PlotMenu = uimenu(app.UIFigure, 'Text','Plot');
                app.ExportViewMenu = uimenu(app.PlotMenu,'Text','Export current view...', ...
                    'MenuSelectedFcn',@(s,e) exportViewMenuSelected(app,e));
                app.OpenViewMenu = uimenu(app.PlotMenu,'Text','Open current view in new figure', ...
                    'MenuSelectedFcn',@(s,e) openViewMenuSelected(app,e));
                app.ExportReportMenu = uimenu(app.PlotMenu,'Text','Export report (plot + parameters)...', ...
                    'MenuSelectedFcn',@(s,e) exportReportMenuSelected(app,e));
            end

            % Only when this module owns its figure — embedded in MODAApp's
            % tab, the logos would overlap the results panel instead of
            % adding anything (MODAApp already shows its own top-bar banner).
            if app.OwnsFigure
                app.anchorBrandingLogos();
            end

            % ---- Left control panel ----
            ctrlPanel = uipanel(sidebarView,'Position',[0 0 330 795],'Title','');

            % See TimeFrequencyAnalysis: embedded tabs have no File menu, so
            % this button is the only way to load data there.
            app.open_file_btn = uibutton(ctrlPanel,'push','Position',[5 790 320 28],'Text','📂 Open File...', ...
                'Tooltip','Load a time series (.mat, .csv, .txt, or any format MATLAB can read).', ...
                'ButtonPushedFcn',@(s,e)app.fileReadMenuSelected(e));

            yl = 750;
            uilabel(ctrlPanel,'Position',[5 yl 100 20],'Text','Signal Pairs:');
            app.signal_list = uilistbox(ctrlPanel,'Position',[5 yl-110 320 110], ...
                'Items',{'Signal Pair 1'}, ...
                'ValueChangedFcn',@(s,e) signalListChanged(app,e));

            yl = yl - 140;
            uilabel(ctrlPanel,'Position',[5 yl 320 20],'Text','Status:');
            app.status = uieditfield(ctrlPanel,'text','Position',[5 yl-25 320 22],'Value','Please Import Signal','Editable','off');

            yl = yl - 60;
            app.wavlet_transform = uibutton(ctrlPanel,'push','Position',[5 yl 155 30],'Text','WPC All Pairs', ...
                'ButtonPushedFcn',@(s,e) waveletTransformButtonPushed(app,e));
            app.wt_single = uibutton(ctrlPanel,'push','Position',[165 yl 155 30],'Text','WPC Single Pair', ...
                'ButtonPushedFcn',@(s,e) wtSingleButtonPushed(app,e));

            yl = yl - 36;
            app.save_preset_btn = uibutton(ctrlPanel,'push','Position',[5 yl 155 26],'Text','Save Preset', ...
                'Tooltip','Save the current Max/Min/Central Freq, Wavelet Type, Preprocess, and Cut Edges settings to a file.', ...
                'ButtonPushedFcn',@(s,e) savePresetButtonPushed(app,e));
            app.load_preset_btn = uibutton(ctrlPanel,'push','Position',[165 yl 155 26],'Text','Load Preset', ...
                'Tooltip','Load previously-saved Max/Min/Central Freq, Wavelet Type, Preprocess, and Cut Edges settings from a file.', ...
                'ButtonPushedFcn',@(s,e) loadPresetButtonPushed(app,e));

            yl = yl - 50;
            uilabel(ctrlPanel,'Position',[5 yl 160 20],'Text','Signal Length:');
            app.signal_length = uieditfield(ctrlPanel,'text','Position',[170 yl 155 22],'Value','','Editable','off');

            % WPC params
            yl = yl - 40;
            uilabel(ctrlPanel,'Position',[5 yl 100 20],'Text','Max Freq (Hz):');
            app.max_freq = uieditfield(ctrlPanel,'text','Position',[110 yl 100 22],'Value','', ...
                'Tooltip','Maximum frequency for which to calculate the wavelet transform (default: Nyquist, fs/2).');
            yl = yl - 30;
            uilabel(ctrlPanel,'Position',[5 yl 100 20],'Text','Min Freq (Hz):');
            app.min_freq = uieditfield(ctrlPanel,'text','Position',[110 yl 100 22],'Value','', ...
                'Tooltip','Minimum frequency for which to calculate the wavelet transform.');
            yl = yl - 30;
            uilabel(ctrlPanel,'Position',[5 yl 100 20],'Text','Central Freq:');
            app.central_freq = uieditfield(ctrlPanel,'text','Position',[110 yl 100 22],'Value','', ...
                'Tooltip','Wavelet resolution parameter (f0). Higher values give better frequency resolution but coarser time resolution, and vice versa.');
            yl = yl - 30;
            uilabel(ctrlPanel,'Position',[5 yl 100 20],'Text','Wavelet Type:');
            app.wavelet_type = uidropdown(ctrlPanel,'Position',[110 yl 155 22], ...
                'Items',{'Lognorm','Morlet','Bump'}, ...
                'Tooltip','Shape of the wavelet used for both signals'' transforms before computing their phase coherence.');
            yl = yl - 30;
            uilabel(ctrlPanel,'Position',[5 yl 100 20],'Text','Preprocess:');
            app.preprocess = uidropdown(ctrlPanel,'Position',[110 yl 155 22], ...
                'Items',{'off','on'}, ...
                'Tooltip','When on, detrends and bandpass-filters each signal to [Min Freq, Max Freq] before transforming.');
            yl = yl - 30;
            uilabel(ctrlPanel,'Position',[5 yl 100 20],'Text','Cut Edges:');
            app.cutedges = uidropdown(ctrlPanel,'Position',[110 yl 155 22], ...
                'Items',{'on','off'}, ...
                'Tooltip','When on, excludes transform values outside the cone of influence (near the signal''s start/end) from the coherence calculation.');

            % Surrogate params — collapsed by default behind a toggle so the
            % sidebar isn't cluttered with a 5-control block most users won't touch.
            yl = yl - 40;
            app.enableSurrogatesCheckbox = uicheckbox(ctrlPanel,'Position',[5 yl 250 22], ...
                'Text','Enable surrogate testing', ...
                'ValueChangedFcn',@(s,e) enableSurrogatesChanged(app,e));
            yl = yl - 30;
            lbl1 = uilabel(ctrlPanel,'Position',[5 yl 140 20],'Text','Surrogate Count:','Visible','off');
            app.surrogate_count = uieditfield(ctrlPanel,'text','Position',[148 yl 100 22],'Value','0','Visible','off');
            yl = yl - 30;
            lbl2 = uilabel(ctrlPanel,'Position',[5 yl 140 20],'Text','Surrogate Type:','Visible','off');
            app.surrogate_type = uidropdown(ctrlPanel,'Position',[148 yl 155 22], ...
                'Items',{'RP','IAAFT1','IAAFT2','AAFT'},'Visible','off');
            yl = yl - 30;
            lbl3 = uilabel(ctrlPanel,'Position',[5 yl 130 20],'Text','Surr. Analysis:','Visible','off');
            app.surrogate_analysis = uidropdown(ctrlPanel,'Position',[140 yl 155 22], ...
                'Items',{'Maximum','Percentile'},'Visible','off');
            yl = yl - 30;
            lbl4 = uilabel(ctrlPanel,'Position',[5 yl 130 20],'Text','Surr. Percentile:','Visible','off');
            app.surrogate_percentile = uieditfield(ctrlPanel,'text','Position',[140 yl 100 22],'Value','0.95','Visible','off');
            yl = yl - 30;
            app.subtract_surrogates = uicheckbox(ctrlPanel,'Position',[5 yl 250 22], ...
                'Text','Subtract Surrogates','Visible','off', ...
                'ValueChangedFcn',@(s,e) subtractSurrogatesChanged(app,e));

            app.supdate = uibutton(ctrlPanel,'push','Position',[5 yl-35 150 25],'Text','Update Surrogates','Visible','off', ...
                'ButtonPushedFcn',@(s,e) supupdateButtonPushed(app,e));

            app.surrogateControls = {lbl1, lbl2, lbl3, lbl4, app.surrogate_count, app.surrogate_type, ...
                app.surrogate_analysis, app.surrogate_percentile, app.subtract_surrogates, app.supdate};

            % Limits
            yl = yl - 80;
            uilabel(ctrlPanel,'Position',[5 yl 75 20],'Text','X Limits:');
            app.xlim = uieditfield(ctrlPanel,'text','Position',[85 yl 235 22],'Value','', ...
                'ValueChangedFcn',@(s,e) xlimFieldChanged(app,e));
            yl = yl - 30;
            uilabel(ctrlPanel,'Position',[5 yl 75 20],'Text','Y Limits:');
            app.ylim = uieditfield(ctrlPanel,'text','Position',[85 yl 235 22],'Value','', ...
                'ValueChangedFcn',@(s,e) ylimFieldChanged(app,e));
            yl = yl - 30;
            uilabel(ctrlPanel,'Position',[5 yl 60 20],'Text','Length:');
            app.length = uieditfield(ctrlPanel,'text','Position',[70 yl 100 22],'Value','','Editable','off');
            app.refresh_limits = uibutton(ctrlPanel,'push','Position',[180 yl 100 22],'Text','Refresh', ...
                'ButtonPushedFcn',@(s,e) refreshLimitsCallback(app,e));

            % Intervals
            yl = yl - 35;
            uilabel(ctrlPanel,'Position',[5 yl 80 20],'Text','Intervals:');
            app.intervals = uieditfield(ctrlPanel,'text','Position',[90 yl 230 22],'Value','', ...
                'ValueChangedFcn',@(s,e) intervalsCallback(app,e));

            % Plot type radio — panel tall enough that the title bar
            % doesn't overlap the radio row.
            yl = yl - 60;
            app.plot_type_bg = uibuttongroup(ctrlPanel,'Position',[5 yl 200 55],'Title','Plot Type', ...
                'SelectionChangedFcn',@(s,e) plotTypeChanged(app,e));
            app.power_rb = uiradiobutton(app.plot_type_bg,'Position',[5 10 90 20],'Text','Power','Value',true);
            app.amp_rb   = uiradiobutton(app.plot_type_bg,'Position',[100 10 90 20],'Text','Amplitude');

            % ---- Time series panel (right top) ----
            app.TimeSeriesPanel = uipanel(app.RootContainer,'Position',[330 500 1270 355],'Title','Time Series');
            app.time_series_1 = uiaxes(app.TimeSeriesPanel,'Position',[5 185 1255 155]);
            app.time_series_2 = uiaxes(app.TimeSeriesPanel,'Position',[5 5   1255 175]);

            % ---- WT pane (right bottom) ----
            app.wt_pane = uipanel(app.RootContainer,'Position',[330 0 1270 500],'Title','Wavelet Phase Coherence');
            app.plot3d   = uiaxes(app.wt_pane,'Position',[5   5 870 480]);
            app.plot_pow = uiaxes(app.wt_pane,'Position',[885 5 380 480]);
            app.cum_avg  = uiaxes(app.wt_pane,'Position',[5   5 1255 480]);
            app.cum_avg.Visible = 'off';

            % Sidebar must always end inside the visible area (it scrolls
            % internally) rather than running off the bottom of the window.
            fitSidebarPanel(ctrlPanel);
        end
    end

    %% Constructor / destructor
    methods (Access = public)
        function app = CoherenceMulti(parentContainer)
            % parentContainer: optional. Omit for a standalone window
            % (legacy behavior); pass a uitab to build onto it instead.
            if nargin < 1
                parentContainer = [];
            end
            createComponents(app, parentContainer);
            registerApp(app, app.UIFigure);
            runStartupFcn(app, @startupFcn);
            if nargout == 0; clear app; end
        end

        function delete(app)
            if app.OwnsFigure && isvalid(app.UIFigure)
                delete(app.UIFigure);
            end
        end
    end
end
