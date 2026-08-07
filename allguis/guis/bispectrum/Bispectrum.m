% Bispectrum — App Designer migration
% Wavelet bispectrum analysis for signal pairs.
% Compatible with MATLAB R2023a through R2026a.

classdef Bispectrum < matlab.apps.AppBase

    %% UI component properties
    properties (Access = public)
        UIFigure            matlab.ui.Figure
        RootContainer   % parent for built components: UIFigure (standalone) or a uitab (embedded)
        OwnsFigure = true   % false when embedded into a shell app's uitab

        % Menus
        FileMenu            matlab.ui.container.Menu
        LoadMenu            matlab.ui.container.Menu
        SaveMatMenu         matlab.ui.container.Menu
        SaveCsvMenu         matlab.ui.container.Menu
        SaveSessionMenu     matlab.ui.container.Menu
        LoadSessionMenu     matlab.ui.container.Menu
        ResetGUIMenu        matlab.ui.container.Menu
        PlotMenu            matlab.ui.container.Menu
        ExportViewMenu      matlab.ui.container.Menu
        OpenViewMenu        matlab.ui.container.Menu

        % Logos
        logo                matlab.ui.control.Image
        nbmplogo            matlab.ui.control.Image

        % Time series axes
        time_series_1       matlab.ui.control.UIAxes
        time_series_2       matlab.ui.control.UIAxes
        plot_pp             matlab.ui.control.UIAxes

        % WT pane — overlapping sets of axes
        wt_pane             matlab.ui.container.Panel
        % WT display (mode 1 or 2)
        plot3d              matlab.ui.control.UIAxes
        plot_pow            matlab.ui.control.UIAxes
        % Bispectrum single view (modes 3-6)
        bisp                matlab.ui.control.UIAxes
        bisp_amp_axis       matlab.ui.control.UIAxes
        bisp_phase_axis     matlab.ui.control.UIAxes
        % All bispectra view (mode 7)
        bispxxx_axis        matlab.ui.control.UIAxes
        bispppp_axis        matlab.ui.control.UIAxes
        bispxpp_axis        matlab.ui.control.UIAxes
        bisppxx_axis        matlab.ui.control.UIAxes
        wt_1                matlab.ui.control.UIAxes
        wt_2                matlab.ui.control.UIAxes

        % Controls
        status              matlab.ui.control.EditField
        bisp_calc           matlab.ui.control.Button
        biph_calc           matlab.ui.control.Button
        mark_freq           matlab.ui.control.Button
        bisp_clear          matlab.ui.control.Button
        refresh_limits      matlab.ui.control.Button

        display_type        matlab.ui.control.DropDown
        detrend_signal_popup matlab.ui.control.DropDown
        preprocess          matlab.ui.control.DropDown

        max_freq            matlab.ui.control.EditField
        min_freq            matlab.ui.control.EditField
        central_freq        matlab.ui.control.EditField
        xlim                matlab.ui.control.EditField
        ylim                matlab.ui.control.EditField
        length              matlab.ui.control.EditField

        surr_num            matlab.ui.control.EditField
        alpha               matlab.ui.control.EditField
        surr_plot           matlab.ui.control.CheckBox

        freq_1              matlab.ui.control.EditField
        freq_2              matlab.ui.control.EditField
        frequency_select    matlab.ui.control.ListBox

        % Plot type radio
        plot_type_bg        matlab.ui.container.ButtonGroup
        power_rb            matlab.ui.control.RadioButton
        amp_rb              matlab.ui.control.RadioButton
    end

    %% Data properties
    properties (Access = public)
        sig             = []
        sig_cut         = []
        sig_pp          = []
        time_axis       = []
        time_axis_cut   = []
        sampling_freq   = NaN
        freqarr         = []
        wavopt          = []
        WT              = {}
        amp_WT          = {}
        pow_WT          = {}
        amp_arr         = {}
        pow_arr         = {}
        xl              = []

        bispxxx         = []
        bispppp         = []
        bispxpp         = []
        bisppxx         = []
        surrxxx         = []
        surrppp         = []
        surrxpp         = []
        surrpxx         = []
        surrxxxT        = []
        surrpppT        = []
        surrxppT        = []
        surrpxxT        = []
        bispxxxS        = []
        bisppppS        = []
        bispxppS        = []
        bisppxxS        = []

        biamp           = {}
        biphase         = {}
        ns              = 0
        alph            = 0.05
        stop            = 0

        freq_plot_list  = {}
        leg_bisp        = {}
        f1list          = {}
        f2list          = {}

        cmap            = []
        linecol         = []
        line2width      = 2
        plot_type       = 1   % 1=power, 2=amp
        h_wait          = []
    end

    %% Helpers
    methods (Access = private)

        function idx = dropdownIndex(~, dd)
            items = dd.Items;
            val   = dd.Value;
            idx   = find(strcmp(items, val), 1);
            if isempty(idx); idx = 1; end
        end

        function setStatus(app, msg)
            app.status.Value = msg;
            drawnow;
        end

        function initSettings(app)
            % Logos are already loaded via uiimage at creation time (see
            % anchorBrandingLogos); handles.logo/nbmplogo deliberately
            % omitted so MODAsettings' guarded logo-loading section skips.
            handles = MODAsettings([], struct());
            app.cmap        = handles.cmap;
            app.linecol     = handles.linecol;
            app.line2width  = handles.line2width;
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

        function showPlot(app, mode)
            % mode 1 = WT power/amp  (plot3d + plot_pow)
            % mode 2 = single bisp   (bisp + bisp_amp_axis + bisp_phase_axis)
            % mode 3 = all bisp      (bispxxx/bispppp/bispxpp/bisppxx + wt_1/wt_2)
            all_axes = [app.plot3d, app.plot_pow, ...
                        app.bisp, app.bisp_amp_axis, app.bisp_phase_axis, ...
                        app.bispxxx_axis, app.bispppp_axis, app.bispxpp_axis, app.bisppxx_axis, ...
                        app.wt_1, app.wt_2];
            for ax = all_axes; ax.Visible = 'off'; end
            switch mode
                case 1
                    app.plot3d.Visible   = 'on';
                    app.plot_pow.Visible = 'on';
                case 2
                    app.bisp.Visible           = 'on';
                    app.bisp_amp_axis.Visible   = 'on';
                    app.bisp_phase_axis.Visible = 'on';
                case 3
                    app.bispxxx_axis.Visible = 'on';
                    app.bispppp_axis.Visible = 'on';
                    app.bispxpp_axis.Visible = 'on';
                    app.bisppxx_axis.Visible = 'on';
                    app.wt_1.Visible         = 'on';
                    app.wt_2.Visible         = 'on';
            end
        end

    end

    %% Callbacks
    methods (Access = private)

        function startupFcn(app)
            initSettings(app);
            app.plot_type = 1;
            app.surr_plot.Enable  = 'off';
            app.display_type.Enable = 'off';
            showPlot(app, 1);
            app.plot3d.Visible  = 'off';
            app.plot_pow.Visible = 'off';
            app.status.Value = 'Please Import Signal';
        end

        %------------------------------------------------------------------
        function fileReadMenuSelected(app, ~)
            handles.cmap         = app.cmap;
            handles.linecol      = app.linecol;
            handles.sampling_freq = app.sampling_freq;
            [handles, A] = MODAreadcheck(handles);
            if A ~= 1; return; end

            % Clear axes and data
            showPlot(app, 1);
            app.plot3d.Visible  = 'off';
            app.plot_pow.Visible = 'off';
            cla(app.time_series_1,'reset');
            cla(app.time_series_2,'reset');
            cla(app.plot_pp,'reset');
            app.display_type.Enable = 'off';

            app.freqarr = []; app.sig = []; app.sig_cut = [];
            app.biamp = {}; app.biphase = {}; app.WT = {};
            app.bispxxx=[]; app.bispppp=[]; app.bispxpp=[]; app.bisppxx=[];
            app.amp_WT={}; app.pow_WT={}; app.amp_arr={}; app.pow_arr={};

            [handles] = MODAread(handles, 1, "even");
            if ~isfield(handles,'sig'); return; end

            app.sig          = handles.sig;
            app.sampling_freq = handles.sampling_freq;
            app.time_axis    = handles.time_axis;
            gfs = 12;

            plot(app.time_series_1, app.time_axis, app.sig(1,:), 'color', app.linecol(1,:));
            xlim(app.time_series_1, [0, size(app.sig,2)/app.sampling_freq]);
            ylabel(app.time_series_1,'Sig 1','FontUnits','points','FontSize',gfs);
            set(app.time_series_1,'XTickLabels',[]);

            if sum(abs(app.sig(1,:) - app.sig(2,:))) ~= 0
                plot(app.time_series_2, app.time_axis, app.sig(2,:), 'color', app.linecol(1,:));
                xlim(app.time_series_2, [0, size(app.sig,2)/app.sampling_freq]);
            end
            ylabel(app.time_series_2,'Sig 2','FontUnits','points','FontSize',gfs);
            xlabel(app.time_series_2,'Time (s)','FontUnits','points','FontSize',gfs);
            set(app.time_series_2,'FontUnits','points','FontSize',gfs);
            linkaxes([app.time_series_1 app.time_series_2],'x');

            refreshLimitsCallback(app,[]);
            preprocessCallback(app,[]);
            app.bisp_calc.Enable = 'on';
            setStatus(app,'Data loaded. Continue with bispectrum calculation.');
        end

        %------------------------------------------------------------------
        function refreshLimitsCallback(app, ~)
            if isempty(app.sig); return; end
            x = app.time_series_1.XLim;
            y = app.time_series_1.YLim;
            t = x(2) - x(1);
            app.xlim.Value = sprintf('%g, %g', x(1), x(2));
            app.ylim.Value = sprintf('%g, %g', y(1), y(2));
            app.length.Value = sprintf('%g', t);

            xlim(app.plot_pp, x);
            fs = app.sampling_freq;
            xi = round(x .* fs);
            xi(2) = min(xi(2), size(app.sig, 2));
            xi(1) = max(xi(1), 1);
            app.sig_cut      = app.sig(:, xi(1):xi(2));
            app.time_axis_cut = app.time_axis(xi(1):xi(2));

            preprocessCallback(app,[]);
            app.xl = [app.time_axis_cut(1) app.time_axis_cut(end)];
            displayTypeChanged(app,[]);
        end

        %------------------------------------------------------------------
        function preprocessCallback(app, ~)
            ppIdx = dropdownIndex(app, app.preprocess);
            if ppIdx == 2   % 'on'
                app.plot_pp.Visible = 'on';
                cla(app.plot_pp,'reset');
                app.detrend_signal_popup.Enable = 'on';

                fmax = str2double(app.max_freq.Value);
                fmin = str2double(app.min_freq.Value);
                if isnan(fmax); fmax = app.sampling_freq/2; end
                if isnan(fmin); fmin = 0; end

                N = size(app.sig_cut);
                L = N(2);
                app.sig_pp = NaN(N);
                sig_sel = dropdownIndex(app, app.detrend_signal_popup);

                for j = 1:2
                    sig = app.sig_cut(j,:);
                    X = (1:length(sig))'/app.sampling_freq;
                    XM = ones(length(X),4);
                    for pn = 1:3
                        CX = X.^pn;
                        XM(:,pn+1) = (CX-mean(CX))/std(CX);
                    end
                    sig = sig(:);
                    new_sig = sig - XM*(pinv(XM)*sig);
                    fx = fft(new_sig, L);
                    Nq = ceil((L+1)/2);
                    ff = [(0:Nq-1), -fliplr(1:L-Nq)] * app.sampling_freq / L;
                    ff = ff(:);
                    fx(abs(ff) <= max([fmin, app.sampling_freq/L]) | abs(ff) >= fmax) = 0;
                    app.sig_pp(j,:) = ifft(fx)';
                end

                gfs = 12;
                plot(app.plot_pp, app.time_axis_cut, app.sig_cut(sig_sel,:), 'color', app.linecol(1,:));
                hold(app.plot_pp,'on');
                plot(app.plot_pp, app.time_axis_cut, app.sig_pp(sig_sel,:),  'color', app.linecol(2,:));
                legend(app.plot_pp,{'Original','Pre-Processed'},'FontSize',gfs-2,'Location','Best','units','points');
                xlim(app.plot_pp,[app.time_axis_cut(1) app.time_axis_cut(end)]);
                xlabel(app.plot_pp,'Time (s)');
                drawnow;
            else
                cla(app.plot_pp,'reset');
                app.plot_pp.Visible = 'off';
                app.detrend_signal_popup.Enable = 'off';
            end
        end

        %------------------------------------------------------------------
        function bispCalcButtonPushed(app, ~)
            app.bisp_calc.Enable = 'off';
            try
                fmax = str2double(app.max_freq.Value);
                fmin = str2double(app.min_freq.Value);
                fc   = str2double(app.central_freq.Value);
                fs   = app.sampling_freq;
                if isnan(fmax); fmax = fs/2; end

                ppIdx = dropdownIndex(app, app.preprocess);
                ppselect = app.preprocess.Items{ppIdx};

                if fmax > fs/2
                    errordlg(['Max freq must be ≤ Nyquist (', num2str(fs/2),' Hz).'],'Parameter Error');
                    app.bisp_calc.Enable = 'on'; return;
                end
                if fmin <= 1/(length(app.sig_cut)/fs)
                    errordlg('Min freq too low. Leave blank for auto.','Parameter Error');
                    app.bisp_calc.Enable = 'on'; return;
                end

                setStatus(app,'Calculating wavelet bispectrum');
                app.ns = str2double(app.surr_num.Value);
                app.bispxxx=[]; app.bispppp=[]; app.bispxpp=[]; app.bisppxx=[];
                app.surrxxx=[]; app.surrppp=[]; app.surrxpp=[]; app.surrpxx=[];
                app.WT = cell(size(app.sig_cut,1),1);

                sigcheck = sum(abs(app.sig_cut(1,:) - app.sig_cut(2,:)));

                % Build argument list for bispecWavNew
                base_args = {'fmin',fmin,'fmax',fmax,'f0',fc};
                if strcmp(ppselect,'off')
                    pp_arg = {'preprocess','off'};
                else
                    pp_arg = {};
                end

                if sigcheck ~= 0
                    % Two distinct signals — compute all 4 cross-bispectra
                    [app.bispxxx,~,~,WT1,~] = bispecWavNew(app.sig_cut(1,:),app.sig_cut(1,:),fs,base_args{:},pp_arg{:},'num',1,'wbar',1);
                    app.bispxxx = abs(app.bispxxx);
                    [app.bispppp] = bispecWavNew(app.sig_cut(2,:),app.sig_cut(2,:),fs,base_args{:},pp_arg{:},'num',2,'wbar',1);
                    app.bispppp = abs(app.bispppp);
                    [app.bispxpp,app.freqarr,app.wavopt,WT1,WT2] = bispecWavNew(app.sig_cut(1,:),app.sig_cut(2,:),fs,base_args{:},pp_arg{:},'num',3,'wbar',1);
                    app.bispxpp = abs(app.bispxpp);
                    [app.bisppxx] = bispecWavNew(app.sig_cut(2,:),app.sig_cut(1,:),fs,base_args{:},pp_arg{:},'num',4,'wbar',1);
                    app.bisppxx = abs(app.bisppxx);

                    if app.ns > 0
                        setStatus(app,'Calculating wavelet bispectrum surrogates');
                        app.h_wait = waitbar(0,'Calculating bispectrum surrogates...','CreateCancelBtn','setappdata(gcbf,''canceling'',1)');
                        setappdata(app.h_wait,'canceling',0);
                        for j = 1:app.ns
                            waitbar(j/app.ns, app.h_wait, sprintf('Calculating surrogate (%d/%d)',j,app.ns));
                            if getappdata(app.h_wait,'canceling'); break; end
                            s1 = wavsurrogate(app.sig_cut(1,:),'IAAFT2',1);
                            if getappdata(app.h_wait,'canceling'); break; end
                            s2 = wavsurrogate(app.sig_cut(2,:),'IAAFT2',1);
                            if getappdata(app.h_wait,'canceling'); break; end
                            app.surrxxx(:,:,j) = abs(bispecWavNew(s1,s1,fs,base_args{:},pp_arg{:},'num',1));
                            if getappdata(app.h_wait,'canceling'); break; end
                            app.surrppp(:,:,j) = abs(bispecWavNew(s2,s2,fs,base_args{:},pp_arg{:},'num',2));
                            if getappdata(app.h_wait,'canceling'); break; end
                            app.surrxpp(:,:,j) = abs(bispecWavNew(s1,s2,fs,base_args{:},pp_arg{:},'num',3));
                            if getappdata(app.h_wait,'canceling'); break; end
                            app.surrpxx(:,:,j) = abs(bispecWavNew(s2,s1,fs,base_args{:},pp_arg{:},'num',4));
                        end
                        delete(app.h_wait);
                    end
                else
                    % Same signal — only b111
                    [app.bispxxx,app.freqarr,app.wavopt,WT1,WT2] = bispecWavNew(app.sig_cut(1,:),app.sig_cut(1,:),fs,base_args{:},pp_arg{:},'num',1,'wbar',1);
                    app.bispxxx = abs(app.bispxxx);
                    WT2 = NaN(size(WT1));
                    app.bispppp = NaN(size(app.bispxxx));
                    app.bispxpp = NaN(size(app.bispxxx));
                    app.bisppxx = NaN(size(app.bispxxx));
                    if app.ns > 0
                        setStatus(app,'Calculating wavelet bispectrum surrogates');
                        app.h_wait = waitbar(0,'Calculating bispectrum surrogates...','CreateCancelBtn','setappdata(gcbf,''canceling'',1)');
                        setappdata(app.h_wait,'canceling',0);
                        for j = 1:app.ns
                            waitbar(j/app.ns, app.h_wait, sprintf('Calculating surrogate (%d/%d)',j,app.ns));
                            if getappdata(app.h_wait,'canceling'); break; end
                            s1 = wavsurrogate(app.sig_cut(1,:),'IAAFT2',1);
                            if getappdata(app.h_wait,'canceling'); break; end
                            app.surrxxx(:,:,j) = abs(bispecWavNew(s1,s1,fs,base_args{:},pp_arg{:},'num',1));
                            app.surrppp(:,:,j) = NaN(size(app.surrxxx(:,:,j)));
                            app.surrxpp(:,:,j) = NaN(size(app.surrxxx(:,:,j)));
                            app.surrpxx(:,:,j) = NaN(size(app.surrxxx(:,:,j)));
                        end
                        delete(app.h_wait);
                    end
                end

                app.amp_WT{1} = abs(WT1); app.amp_arr{1} = nanmean(app.amp_WT{1},2);
                app.pow_WT{1} = abs(WT1).^2; app.pow_arr{1} = nanmean(app.pow_WT{1},2);
                app.amp_WT{2} = abs(WT2); app.amp_arr{2} = nanmean(app.amp_WT{2},2);
                app.pow_WT{2} = abs(WT2).^2; app.pow_arr{2} = nanmean(app.pow_WT{2},2);

                app.display_type.Enable = 'on';
                if app.OwnsFigure
                    app.SaveMatMenu.Enable  = 'on';
                    app.SaveCsvMenu.Enable  = 'on';
                    app.ExportViewMenu.Enable = 'on';
                    app.OpenViewMenu.Enable   = 'on';
                end
                if app.ns > 0; app.surr_plot.Enable = 'on'; end
                app.bisp_calc.Enable = 'on';
                setStatus(app,'Calculation complete');
                displayTypeChanged(app,[]);

            catch e
                errordlg(e.message,'Error');
                app.bisp_calc.Enable = 'on';
                rethrow(e);
            end
        end

        %------------------------------------------------------------------
        function displayTypeChanged(app, ~)
            if isempty(app.freqarr); return; end
            disp_select = dropdownIndex(app, app.display_type);
            setStatus(app,'Plotting data');
            gfs = 12;

            if length(app.time_axis_cut) >= 2000
                screensize = max(get(groot,'Screensize'));
                n = floor(size(app.sig_cut,2)/screensize);
            else
                n = 1;
            end

            % Modes 1-2: WT display
            if (disp_select == 1 || disp_select == 2) && ~isempty(app.amp_WT)
                if length(app.pow_WT{1}) ~= length(app.time_axis_cut)
                    errordlg('Time axis changed — please recalculate.','Error');
                    showPlot(app, 1);
                    app.plot3d.Visible  = 'off';
                    app.plot_pow.Visible = 'off';
                    return;
                end
                showPlot(app, 1);
                cla(app.plot3d,'reset'); cla(app.plot_pow,'reset');
                if app.plot_type == 1
                    pcolor(app.plot3d, app.time_axis_cut(1:n:end), app.freqarr, app.pow_WT{disp_select}(:,1:n:end));
                    zlabel(app.plot3d,'Power');
                    plot(app.plot_pow, app.pow_arr{disp_select}, app.freqarr, '-k','LineWidth',3);
                    xlabel(app.plot_pow,'Average Power');
                else
                    pcolor(app.plot3d, app.time_axis_cut(1:n:end), app.freqarr, app.amp_WT{disp_select}(:,1:n:end));
                    zlabel(app.plot3d,'Amplitude');
                    plot(app.plot_pow, app.amp_arr{disp_select}, app.freqarr, '-k','LineWidth',3);
                    xlabel(app.plot_pow,'Average Amplitude');
                end
                colormap(app.plot3d, app.cmap); shading(app.plot3d,'interp');
                set(app.plot3d,'yscale','log','ylim',[min(app.freqarr) max(app.freqarr)],'xlim',app.xl);
                set(app.plot_pow,'yscale','log','ylim',[min(app.freqarr) max(app.freqarr)]);
                xlabel(app.plot3d,'Time (s)'); ylabel(app.plot3d,'Frequency (Hz)');
                ylabel(app.plot_pow,'Frequency (Hz)');

            % Modes 3-6: single bispectrum
            elseif disp_select >= 3 && disp_select <= 6 && ~isempty(app.bispxxx)
                if length(app.pow_WT{1}) ~= length(app.time_axis_cut)
                    errordlg('Time axis changed — please recalculate.','Error');
                    return;
                end
                showPlot(app, 2);
                cla(app.bisp,'reset'); cla(app.bisp_amp_axis,'reset'); cla(app.bisp_phase_axis,'reset');

                surrsel = app.surr_plot.Value;
                if ~surrsel
                    % Raw bispectra
                    switch disp_select
                        case 3; pcolor(app.bisp,app.freqarr,app.freqarr,app.bispxxx); title(app.bisp,'Bispectrum 111'); xlabel(app.bisp,'Freq - Sig 1 (Hz)'); ylabel(app.bisp,'Freq - Sig 1 (Hz)');
                        case 4; pcolor(app.bisp,app.freqarr,app.freqarr,app.bispppp); title(app.bisp,'Bispectrum 222'); xlabel(app.bisp,'Freq - Sig 2 (Hz)'); ylabel(app.bisp,'Freq - Sig 2 (Hz)');
                        case 5; pcolor(app.bisp,app.freqarr,app.freqarr,app.bispxpp); title(app.bisp,'Bispectrum 122'); xlabel(app.bisp,'Freq - Sig 2 (Hz)'); ylabel(app.bisp,'Freq - Sig 1 (Hz)');
                        case 6; pcolor(app.bisp,app.freqarr,app.freqarr,app.bisppxx); title(app.bisp,'Bispectrum 211'); xlabel(app.bisp,'Freq - Sig 1 (Hz)'); ylabel(app.bisp,'Freq - Sig 2 (Hz)');
                    end
                else
                    % Surrogate-subtracted
                    app.alph = str2double(app.alpha.Value);
                    K = max(1, floor((app.ns+1)*app.alph));
                    switch disp_select
                        case 3
                            S = sort(app.surrxxx,3,'descend'); app.surrxxxT=S(:,:,K);
                            app.bispxxxS = app.bispxxx-app.surrxxxT; app.bispxxxS(app.bispxxxS<0)=NaN;
                            pcolor(app.bisp,app.freqarr,app.freqarr,app.bispxxxS); title(app.bisp,'Bispectrum 111');
                        case 4
                            S = sort(app.surrppp,3,'descend'); app.surrpppT=S(:,:,K);
                            app.bisppppS = app.bispppp-app.surrpppT; app.bisppppS(app.bisppppS<0)=NaN;
                            pcolor(app.bisp,app.freqarr,app.freqarr,app.bisppppS); title(app.bisp,'Bispectrum 222');
                        case 5
                            S = sort(app.surrxpp,3,'descend'); app.surrxppT=S(:,:,K);
                            app.bispxppS = app.bispxpp-app.surrxppT; app.bispxppS(app.bispxppS<0)=NaN;
                            pcolor(app.bisp,app.freqarr,app.freqarr,app.bispxppS); title(app.bisp,'Bispectrum 122');
                        case 6
                            S = sort(app.surrpxx,3,'descend'); app.surrpxxT=S(:,:,K);
                            app.bisppxxS = app.bisppxx-app.surrpxxT; app.bisppxxS(app.bisppxxS<0)=NaN;
                            pcolor(app.bisp,app.freqarr,app.freqarr,app.bisppxxS); title(app.bisp,'Bispectrum 211');
                    end
                end
                set(app.bisp,'yscale','log','xscale','log');
                colormap(app.bisp, app.cmap); shading(app.bisp,'interp');
                idx_first = find(sum(~isnan(app.bispxxx),1)>0,1,'first');
                idx_last  = find(sum(~isnan(app.bispxxx),1)>0,1,'last');
                if ~isempty(idx_first)
                    xlim(app.bisp,[app.freqarr(idx_first) app.freqarr(idx_last)]);
                    ylim(app.bisp,[app.freqarr(idx_first) app.freqarr(idx_last)]);
                end
                grid(app.bisp,'on');

            % Mode 7: all bispectra + WT
            elseif disp_select == 7 && ~isempty(app.WT)
                showPlot(app, 3);

                surrsel = app.surr_plot.Value;
                if ~surrsel
                    pcolor(app.bispxxx_axis,app.freqarr,app.freqarr,app.bispxxx);
                    pcolor(app.bispppp_axis,app.freqarr,app.freqarr,app.bispppp);
                    pcolor(app.bisppxx_axis,app.freqarr,app.freqarr,app.bisppxx);
                    pcolor(app.bispxpp_axis,app.freqarr,app.freqarr,app.bispxpp);
                else
                    app.alph = str2double(app.alpha.Value);
                    K = max(1, floor((app.ns+1)*app.alph));
                    S = sort(app.surrxxx,3,'descend'); app.surrxxxT=S(:,:,K);
                    app.bispxxxS = app.bispxxx-app.surrxxxT; app.bispxxxS(app.bispxxxS<0)=NaN;
                    pcolor(app.bispxxx_axis,app.freqarr,app.freqarr,app.bispxxxS);
                    sigcheck = sum(abs(app.sig_cut(1,:)-app.sig_cut(2,:)));
                    if sigcheck ~= 0
                        S = sort(app.surrppp,3,'descend'); app.surrpppT=S(:,:,K);
                        app.bisppppS=app.bispppp-app.surrpppT; app.bisppppS(app.bisppppS<0)=NaN;
                        pcolor(app.bispppp_axis,app.freqarr,app.freqarr,app.bisppppS);
                        S = sort(app.surrxpp,3,'descend'); app.surrxppT=S(:,:,K);
                        app.bispxppS=app.bispxpp-app.surrxppT; app.bispxppS(app.bispxppS<0)=NaN;
                        pcolor(app.bispxpp_axis,app.freqarr,app.freqarr,app.bispxppS);
                        S = sort(app.surrpxx,3,'descend'); app.surrpxxT=S(:,:,K);
                        app.bisppxxS=app.bisppxx-app.surrpxxT; app.bisppxxS(app.bisppxxS<0)=NaN;
                        pcolor(app.bisppxx_axis,app.freqarr,app.freqarr,app.bisppxxS);
                    end
                end

                if length(app.pow_WT{1}) == length(app.time_axis_cut)
                    if app.plot_type == 1
                        pcolor(app.wt_1, app.time_axis_cut(1:n:end), app.freqarr, app.pow_WT{1}(:,1:n:end));
                        pcolor(app.wt_2, app.time_axis_cut(1:n:end), app.freqarr, app.pow_WT{2}(:,1:n:end));
                    else
                        pcolor(app.wt_1, app.time_axis_cut(1:n:end), app.freqarr, app.amp_WT{1}(:,1:n:end));
                        pcolor(app.wt_2, app.time_axis_cut(1:n:end), app.freqarr, app.amp_WT{2}(:,1:n:end));
                    end
                    colormap(app.wt_1,app.cmap); colormap(app.wt_2,app.cmap);
                    shading(app.wt_1,'interp'); shading(app.wt_2,'interp');
                    title(app.wt_1,'Wavelet Transform - Signal 1');
                    title(app.wt_2,'Wavelet Transform - Signal 2');
                    ylabel(app.wt_1,'Frequency (Hz)');
                    ylabel(app.wt_2,'Frequency (Hz)');
                    xlabel(app.wt_2,'Time (s)');
                    idx_first = find(sum(~isnan(app.bispxxx),1)>0,1,'first');
                    idx_last  = find(sum(~isnan(app.bispxxx),1)>0,1,'last');
                    if ~isempty(idx_first)
                        set(app.wt_1,'yscale','log','ylim',[app.freqarr(idx_first) app.freqarr(idx_last)],'xlim',app.xl,'xticklabel',[]);
                        set(app.wt_2,'yscale','log','ylim',[app.freqarr(idx_first) app.freqarr(idx_last)],'xlim',app.xl);
                    end
                end

                bisp_list = [app.bispxxx_axis; app.bispppp_axis; app.bispxpp_axis; app.bisppxx_axis];
                titles = {'b111','b222','b122','b211'};
                idx_first = find(sum(~isnan(app.bispxxx),1)>0,1,'first');
                idx_last  = find(sum(~isnan(app.bispxxx),1)>0,1,'last');
                for i = 1:4
                    colormap(bisp_list(i),app.cmap); shading(bisp_list(i),'interp');
                    set(bisp_list(i),'yscale','log','xscale','log');
                    title(bisp_list(i),titles{i});
                    if ~isempty(idx_first)
                        xlim(bisp_list(i),[app.freqarr(idx_first) app.freqarr(idx_last)]);
                        ylim(bisp_list(i),[app.freqarr(idx_first) app.freqarr(idx_last)]);
                    end
                end
            end
            setStatus(app,'Plotting complete');
        end

        %------------------------------------------------------------------
        function markFreqButtonPushed(app, ~)
            disp_select = dropdownIndex(app, app.display_type);
            if disp_select <= 2 || disp_select >= 7; return; end
            [x, y] = ginput(1);
            % Clear old temporary marker
            ch = allchild(app.bisp);
            for j = 1:numel(ch)
                if strcmp(get(ch(j),'Type'),'line')
                    if length(get(ch(j),'XData'))==1 && strcmp(get(ch(j),'Marker'),'*')
                        delete(ch(j));
                    end
                end
            end
            hold(app.bisp,'on'); plot(app.bisp,x,y,'k*');
            app.freq_1.Value = num2str(x);
            app.freq_2.Value = num2str(y);
        end

        %------------------------------------------------------------------
        function biphCalcButtonPushed(app, ~)
            disp_select = dropdownIndex(app, app.display_type);
            if disp_select <= 2 || disp_select >= 7; return; end

            f1_str = app.freq_1.Value; f2_str = app.freq_2.Value;
            if isempty(f1_str) || isempty(f2_str); return; end
            f1 = str2double(f1_str); f2 = str2double(f2_str);
            fmin_v = min(app.freqarr); fmax_v = max(app.freqarr);

            if f1 < fmin_v || f1 > fmax_v || f2 < fmin_v || f2 > fmax_v
                errordlg('Selected point is outside the allowable range','Parameter Error'); return;
            end

            list = app.frequency_select.Items;
            new_entry = sprintf('%f, %f', f1, f2);
            list{end+1} = new_entry;
            app.frequency_select.Items = list;
            app.frequency_select.Value = list{end};

            % Mark permanent point
            hold(app.bisp,'on');
            ch = allchild(app.bisp);
            for j = 1:numel(ch)
                if strcmp(get(ch(j),'Type'),'line')
                    if length(get(ch(j),'XData'))==1 && strcmp(get(ch(j),'Marker'),'*')
                        set(ch(j),'Marker','o','MarkerEdgeColor','k');
                    end
                end
            end
            plot(app.bisp, f1, f2, 'ok');

            ppIdx = dropdownIndex(app, app.preprocess);
            ppselect = app.preprocess.Items{ppIdx};
            fc = str2double(app.central_freq.Value);

            j = numel(list);
            if strcmp(ppselect,'off')
                switch disp_select
                    case 3; [app.biamp{j}, app.biphase{j}] = biphaseWavNew(app.sig_cut(1,:),app.sig_cut(1,:),app.sampling_freq,[f2 f1],app.wavopt);
                    case 4; [app.biamp{j}, app.biphase{j}] = biphaseWavNew(app.sig_cut(2,:),app.sig_cut(2,:),app.sampling_freq,[f2 f1],app.wavopt);
                    case 5; [app.biamp{j}, app.biphase{j}] = biphaseWavNew(app.sig_cut(1,:),app.sig_cut(2,:),app.sampling_freq,[f2 f1],app.wavopt);
                    case 6; [app.biamp{j}, app.biphase{j}] = biphaseWavNew(app.sig_cut(2,:),app.sig_cut(1,:),app.sampling_freq,[f2 f1],app.wavopt);
                end
            else
                switch disp_select
                    case 3; [app.biamp{j}, app.biphase{j}] = biphaseWavNew(app.sig_pp(1,:),app.sig_pp(1,:),app.sampling_freq,[f2 f1],app.wavopt);
                    case 4; [app.biamp{j}, app.biphase{j}] = biphaseWavNew(app.sig_pp(2,:),app.sig_pp(2,:),app.sampling_freq,[f2 f1],app.wavopt);
                    case 5; [app.biamp{j}, app.biphase{j}] = biphaseWavNew(app.sig_pp(1,:),app.sig_pp(2,:),app.sampling_freq,[f2 f1],app.wavopt);
                    case 6; [app.biamp{j}, app.biphase{j}] = biphaseWavNew(app.sig_pp(2,:),app.sig_pp(1,:),app.sampling_freq,[f2 f1],app.wavopt);
                end
            end
            frequencySelectChanged(app,[]);
        end

        %------------------------------------------------------------------
        function frequencySelectChanged(app, ~)
            disp_select = dropdownIndex(app, app.display_type);
            if disp_select <= 2 || disp_select >= 7; return; end
            gfs = 12;

            cla(app.bisp_amp_axis); cla(app.bisp_phase_axis);
            hold(app.bisp_amp_axis,'on'); hold(app.bisp_phase_axis,'on'); hold(app.bisp,'on');

            list = app.frequency_select.Items;
            if isempty(list); return; end

            sel_val = app.frequency_select.Value;
            frequency_selected = find(strcmp(list, sel_val));
            if isempty(frequency_selected); frequency_selected = 1:numel(list); end

            app.leg_bisp = {};
            for j = 1:numel(frequency_selected)
                idx = frequency_selected(j);
                if idx > numel(app.biamp); continue; end
                col = min(j, 8);
                plot(app.bisp_amp_axis,   app.time_axis_cut, app.biamp{idx},   'Linewidth',1,'color',app.linecol(col,:));
                plot(app.bisp_phase_axis, app.time_axis_cut, app.biphase{idx}, 'Linewidth',1,'color',app.linecol(col,:));
                fl = csv_to_mvar(list{idx});
                plot(app.bisp, fl(1), fl(2), 'ok');
                app.leg_bisp{j} = sprintf('%g - %g Hz', round(fl(1),3), round(fl(2),3));
            end
            ylabel(app.bisp_amp_axis,'Biamplitude');
            ylabel(app.bisp_phase_axis,'Biphase');
            xlabel(app.bisp_phase_axis,'Time (s)');
            if ~isempty(app.leg_bisp)
                legend(app.bisp_amp_axis, app.leg_bisp,'FontSize',gfs);
            end
            grid(app.bisp_amp_axis,'off'); grid(app.bisp_phase_axis,'off');
            setStatus(app,'Plotting complete');
        end

        %------------------------------------------------------------------
        function bispClearButtonPushed(app, ~)
            disp_select = dropdownIndex(app, app.display_type);
            if disp_select <= 2 || disp_select >= 7; return; end
            cla(app.bisp_amp_axis,'reset'); cla(app.bisp_phase_axis,'reset');
            % Clear point markers on bisp
            ch = allchild(app.bisp);
            for j = 1:numel(ch)
                if strcmp(get(ch(j),'Type'),'line')
                    mk = get(ch(j),'Marker');
                    if strcmp(mk,'o') || strcmp(mk,'*'); delete(ch(j)); end
                end
            end
        end

        %------------------------------------------------------------------
        function plotTypeChanged(app, ~)
            app.plot_type = 1 + ~app.power_rb.Value;
            displayTypeChanged(app,[]);
        end

        function surrPlotChanged(app, ~)
            displayTypeChanged(app,[]);
        end

        function alphaChanged(app, ~)
            displayTypeChanged(app,[]);
        end

        %------------------------------------------------------------------
        % Export current view (replaces the old 9-item Save/plot menu list)
        function axs = currentViewAxes(app)
            % Whichever result axes are currently visible, in display order —
            % works across all 7 display_type modes without hardcoding each one.
            candidates = {app.plot3d, app.plot_pow, app.bisp, app.bisp_amp_axis, app.bisp_phase_axis, ...
                          app.bispxxx_axis, app.bispxpp_axis, app.bisppxx_axis, app.bispppp_axis, ...
                          app.wt_1, app.wt_2};
            axs = {};
            for c = candidates
                if isvalid(c{1}) && strcmp(c{1}.Visible,'on')
                    axs{end+1} = c{1}; %#ok<AGROW>
                end
            end
        end

        function fig = buildViewFigure(app)
            axs = app.currentViewAxes();
            n = numel(axs);
            cols = max(1, min(n,3));
            rows = max(1, ceil(n/cols));
            fig = figure('Visible','off','Position',[100 100 380*cols 380*rows]);
            for i = 1:n
                newAx = copyobj(axs{i}, fig);
                subplot(rows, cols, i, newAx);
                colormap(newAx, app.cmap);
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

        %------------------------------------------------------------------
        function saveMatMenuSelected(app, ~)
            try
                [FileName,PathName] = uiputfile('.mat','Save data as');
                if isequal(FileName,0); return; end
                Bisp_data = buildSaveStruct(app);
                save([PathName,FileName],'Bisp_data');
            catch e; errordlg(e.message,'Error'); rethrow(e); end
        end

        function saveCsvMenuSelected(app, ~)
            try
                Bisp_data = buildSaveStruct(app);
                csvsavefolder(app, Bisp_data);
            catch e; errordlg(e.message,'Error'); rethrow(e); end
        end

        function D = buildSaveStruct(app)
            ppIdx    = dropdownIndex(app, app.preprocess);
            ppselect = app.preprocess.Items{ppIdx};
            sigcheck = sum(abs(app.sig_cut(1,:)-app.sig_cut(2,:)));

            D.b111 = app.bispxxx;
            if sigcheck ~= 0
                D.b222 = app.bispppp; D.b122 = app.bispxpp; D.b211 = app.bisppxx;
            end
            if ~isempty(app.alph) && app.alph ~= 0
                D.b111surr = app.surrxxx;
                if sigcheck ~= 0
                    D.b222surr = app.surrppp; D.b122surr = app.surrxpp; D.b211surr = app.surrpxx;
                end
                K = max(1, floor((app.ns+1)*app.alph));
                S = sort(app.surrxxx,3,'descend'); D.b111surr_threshold = S(:,:,K);
                if sigcheck ~= 0
                    S2=sort(app.surrppp,3,'descend'); D.b222surr_threshold=S2(:,:,K);
                    S3=sort(app.surrxpp,3,'descend'); D.b122surr_threshold=S3(:,:,K);
                    S4=sort(app.surrpxx,3,'descend'); D.b211surr_threshold=S4(:,:,K);
                end
                D.alpha  = app.alph;
                D.surrnum = app.ns;
            end
            freq_list = app.frequency_select.Items;
            if ~isempty(freq_list)
                D.selected_points = freq_list';
                disp_sel = dropdownIndex(app, app.display_type);
                names = {'111','222','122','211'};
                if disp_sel >= 3 && disp_sel <= 6
                    D.selected_plot = names{disp_sel-2};
                end
                D.biamp   = app.biamp;
                D.biphase = app.biphase;
            end
            if app.plot_type == 1
                D.WTPower = cell2mat(app.pow_arr)';
            else
                D.WTAmplitude = cell2mat(app.amp_arr)';
            end
            D.Frequency         = app.freqarr;
            D.Time              = app.time_axis_cut;
            D.Sampling_frequency = app.sampling_freq;
            D.fmax              = app.freqarr(end);
            D.fmin              = app.freqarr(1);
            D.fr                = str2double(app.central_freq.Value);
            D.Preprocessing     = ppselect;
        end

        function csvsavefolder(app, D)
            curr = pwd;
            [FileName, PathName] = uiputfile('.csv','Save as');
            if isequal(FileName,0); return; end
            cd(PathName);
            foldername = FileName(1:end-4);
            mkdir(foldername);

            L = length(D.Time);
            dstart = 15;
            data{1,1}='MODA v1.0 - Wavelet Bispectrum Analysis'; data{2,1}=date; data{3,1}=[];
            data{4,1}='PARAMETERS';
            data{5,1}='Sampling frequency (Hz)'; data{5,2}=D.Sampling_frequency;
            data{6,1}='Maximum frequency (Hz)';  data{6,2}=D.fmax;
            data{7,1}='Minimum frequency (Hz)';  data{7,2}=D.fmin;
            data{8,1}='Frequency resolution';     data{8,2}=D.fr;
            data{9,1}='Preprocessing';             data{9,2}=D.Preprocessing;
            data{10,1}='Cut Edges'; data{10,2}='on';
            data{11,1}='Time start (s)'; data{11,2}=min(D.Time);
            data{12,1}='Time end (s)';   data{12,2}=max(D.Time);

            if isfield(D,'selected_plot')
                Np = size(D.selected_points,1);
                data{13,1}='Selected plot'; data{13,2}=D.selected_plot;
                for j=1:Np; data{13+j,1}=['Selected point ',num2str(j),' (Hz)']; data{13+j,2}=D.selected_points{j}; dstart=dstart+1; end
                for k=1:Np
                    data{dstart,k*2}=['Biamp - point ',num2str(k)];
                    data{dstart,(k*2)+1}=['Biphase - point ',num2str(k)];
                    for nn=1:L; data{nn+dstart,k*2}=D.biamp{k}(nn); data{nn+dstart,(k*2)+1}=D.biphase{k}(nn); end
                end
                data{dstart,1}='Time (s)';
                for l=1:L; data{l+dstart,1}=D.Time(l); end
                cell2csv([foldername,'\params_biphase_biamp.csv'],data,',');
            else
                cell2csv([foldername,'\params.csv'],data,',');
            end

            sigcheck = sum(abs(app.sig_cut(1,:)-app.sig_cut(2,:)));
            S = size(D.b111);
            bisp_list = {'b111','b222','b122','b211'};
            file_list = {'Bispectrum_111','Bispectrum_222','Bispectrum_122','Bispectrum_211'};
            for bi = 1:4
                if bi > 1 && sigcheck == 0; continue; end
                bdat = D.(bisp_list{bi});
                data2 = cell(S(1)+1, S(2)+1);
                data2{1,1} = 'Freq';
                for ni=1:length(D.Frequency); data2{ni+1,1}=D.Frequency(ni); data2{1,ni+1}=D.Frequency(ni); end
                for j=1:S(1); for k=1:S(2); data2{j+1,k+1}=bdat(j,k); end; end
                cell2csv([foldername,'\',file_list{bi},'.csv'],data2,',');
            end
            cd(curr);
        end

        function saveSessionMenuSelected(app, ~)
            handles.sig          = app.sig;
            handles.sig_cut      = app.sig_cut;
            handles.time_axis    = app.time_axis;
            handles.sampling_freq = app.sampling_freq;
            handles.freqarr      = app.freqarr;
            handles.wavopt       = app.wavopt;
            handles.bispxxx      = app.bispxxx;
            handles.bispppp      = app.bispppp;
            MODAsave(handles);
        end

        function resetGUIMenuSelected(app, ~)
            delete(app); Bispectrum;
        end

    end  % private callbacks

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
                app.UIFigure.Name = 'MODA — Wavelet Bispectrum';
                app.UIFigure.CloseRequestFcn = @(~,~) MODAclose(app.UIFigure, struct());
                app.OwnsFigure    = true;
                app.RootContainer = app.UIFigure;
            else
                app.RootContainer = parentContainer;
                app.UIFigure      = ancestor(parentContainer, 'figure');
                app.OwnsFigure    = false;
            end

            % Menus (figure-level; only when this module owns the figure)
            if app.OwnsFigure
                app.FileMenu = uimenu(app.UIFigure,'Text','File');
                app.LoadMenu = uimenu(app.FileMenu,'Text','Load Time Series','MenuSelectedFcn',@(s,e) fileReadMenuSelected(app,e));
                app.SaveMatMenu = uimenu(app.FileMenu,'Text','Save Data (.mat)','MenuSelectedFcn',@(s,e) saveMatMenuSelected(app,e));
                app.SaveCsvMenu = uimenu(app.FileMenu,'Text','Save Data (.csv)','MenuSelectedFcn',@(s,e) saveCsvMenuSelected(app,e));
                uimenu(app.FileMenu,'Separator','on');
                app.SaveSessionMenu = uimenu(app.FileMenu,'Text','Save Session','MenuSelectedFcn',@(s,e) saveSessionMenuSelected(app,e));
                app.LoadSessionMenu = uimenu(app.FileMenu,'Text','Load Session');
                app.ResetGUIMenu    = uimenu(app.FileMenu,'Text','New Workspace','MenuSelectedFcn',@(s,e) resetGUIMenuSelected(app,e));

                % Replaces the old 9-item Plot menu (heaviest popup-window
                % offender in the app) with two actions that act on
                % whichever axes are currently visible in the results pane.
                app.PlotMenu = uimenu(app.UIFigure,'Text','Plot');
                app.ExportViewMenu = uimenu(app.PlotMenu,'Text','Export current view...', ...
                    'MenuSelectedFcn',@(s,e) exportViewMenuSelected(app,e));
                app.OpenViewMenu = uimenu(app.PlotMenu,'Text','Open current view in new figure', ...
                    'MenuSelectedFcn',@(s,e) openViewMenuSelected(app,e));
            end

            % Only when this module owns its figure — embedded in MODAApp's
            % tab, the logos would overlap the results panel instead of
            % adding anything (MODAApp already shows its own top-bar banner).
            if app.OwnsFigure
                app.anchorBrandingLogos();
            end

            % ---- Left control panel ----
            ctrlPanel = uipanel(app.RootContainer,'Position',[0 0 330 795],'Title','');

            yl = 760;
            uilabel(ctrlPanel,'Position',[5 yl 320 20],'Text','Status:');
            app.status = uieditfield(ctrlPanel,'text','Position',[5 yl-25 320 22],'Value','Please Import Signal','Editable','off');

            yl = yl - 55;
            app.bisp_calc = uibutton(ctrlPanel,'push','Position',[5 yl 155 30],'Text','Calculate Bispectrum', ...
                'ButtonPushedFcn',@(s,e) bispCalcButtonPushed(app,e));
            app.bisp_calc.Enable = 'off';

            yl = yl - 45;
            uilabel(ctrlPanel,'Position',[5 yl 100 20],'Text','Display:');
            app.display_type = uidropdown(ctrlPanel,'Position',[110 yl 200 22], ...
                'Items',{'WT Signal 1','WT Signal 2','Bispectrum 111','Bispectrum 222','Bispectrum 122','Bispectrum 211','All Bispectra'}, ...
                'ValueChangedFcn',@(s,e) displayTypeChanged(app,e));

            yl = yl - 35;
            uilabel(ctrlPanel,'Position',[5 yl 100 20],'Text','Max Freq (Hz):');
            app.max_freq = uieditfield(ctrlPanel,'text','Position',[110 yl 100 22],'Value','', ...
                'ValueChangedFcn',@(s,e) preprocessCallback(app,e));
            yl = yl - 30;
            uilabel(ctrlPanel,'Position',[5 yl 100 20],'Text','Min Freq (Hz):');
            app.min_freq = uieditfield(ctrlPanel,'text','Position',[110 yl 100 22],'Value','', ...
                'ValueChangedFcn',@(s,e) preprocessCallback(app,e));
            yl = yl - 30;
            uilabel(ctrlPanel,'Position',[5 yl 100 20],'Text','Central Freq:');
            app.central_freq = uieditfield(ctrlPanel,'text','Position',[110 yl 100 22],'Value','');

            yl = yl - 35;
            uilabel(ctrlPanel,'Position',[5 yl 100 20],'Text','Preprocess:');
            app.preprocess = uidropdown(ctrlPanel,'Position',[110 yl 155 22], ...
                'Items',{'off','on'}, ...
                'ValueChangedFcn',@(s,e) preprocessCallback(app,e));
            yl = yl - 30;
            uilabel(ctrlPanel,'Position',[5 yl 110 20],'Text','Detrend Signal:');
            app.detrend_signal_popup = uidropdown(ctrlPanel,'Position',[120 yl 155 22], ...
                'Items',{'Sig 1','Sig 2'}, ...
                'ValueChangedFcn',@(s,e) preprocessCallback(app,e));

            % Surrogate params
            yl = yl - 40;
            uilabel(ctrlPanel,'Position',[5 yl 140 20],'Text','Surrogate Count:');
            app.surr_num = uieditfield(ctrlPanel,'text','Position',[148 yl 100 22],'Value','0');
            yl = yl - 30;
            uilabel(ctrlPanel,'Position',[5 yl 140 20],'Text','Alpha:');
            app.alpha = uieditfield(ctrlPanel,'text','Position',[148 yl 100 22],'Value','0.05', ...
                'ValueChangedFcn',@(s,e) alphaChanged(app,e));
            yl = yl - 30;
            app.surr_plot = uicheckbox(ctrlPanel,'Position',[5 yl 200 22],'Text','Show Surrogate Threshold', ...
                'ValueChangedFcn',@(s,e) surrPlotChanged(app,e));

            % Limits
            yl = yl - 50;
            uilabel(ctrlPanel,'Position',[5 yl 75 20],'Text','X Limits:');
            app.xlim = uieditfield(ctrlPanel,'text','Position',[85 yl 235 22],'Value','','Editable','off');
            yl = yl - 30;
            uilabel(ctrlPanel,'Position',[5 yl 75 20],'Text','Y Limits:');
            app.ylim = uieditfield(ctrlPanel,'text','Position',[85 yl 235 22],'Value','','Editable','off');
            yl = yl - 30;
            uilabel(ctrlPanel,'Position',[5 yl 60 20],'Text','Length:');
            app.length = uieditfield(ctrlPanel,'text','Position',[70 yl 100 22],'Value','','Editable','off');
            app.refresh_limits = uibutton(ctrlPanel,'push','Position',[180 yl 100 22],'Text','Refresh', ...
                'ButtonPushedFcn',@(s,e) refreshLimitsCallback(app,e));

            % Frequency selection
            yl = yl - 50;
            uilabel(ctrlPanel,'Position',[5 yl 95 20],'Text','Freq 1 (Hz):');
            app.freq_1 = uieditfield(ctrlPanel,'text','Position',[105 yl 100 22],'Value','');
            yl = yl - 30;
            uilabel(ctrlPanel,'Position',[5 yl 95 20],'Text','Freq 2 (Hz):');
            app.freq_2 = uieditfield(ctrlPanel,'text','Position',[105 yl 100 22],'Value','');
            yl = yl - 30;
            app.mark_freq = uibutton(ctrlPanel,'push','Position',[5 yl 100 25],'Text','Select Point', ...
                'ButtonPushedFcn',@(s,e) markFreqButtonPushed(app,e));
            app.biph_calc = uibutton(ctrlPanel,'push','Position',[110 yl 100 25],'Text','Calc Biphase', ...
                'ButtonPushedFcn',@(s,e) biphCalcButtonPushed(app,e));
            yl = yl - 30;
            app.bisp_clear = uibutton(ctrlPanel,'push','Position',[5 yl 100 25],'Text','Clear', ...
                'ButtonPushedFcn',@(s,e) bispClearButtonPushed(app,e));
            yl = yl - 80;
            uilabel(ctrlPanel,'Position',[5 yl+55 150 20],'Text','Selected Frequencies:');
            app.frequency_select = uilistbox(ctrlPanel,'Position',[5 yl 320 80], ...
                'Items',{}, ...
                'ValueChangedFcn',@(s,e) frequencySelectChanged(app,e));

            % Plot type radio — panel tall enough that the title bar
            % doesn't overlap the radio row.
            yl = yl - 65;
            app.plot_type_bg = uibuttongroup(ctrlPanel,'Position',[5 yl 200 55],'Title','Plot Type', ...
                'SelectionChangedFcn',@(s,e) plotTypeChanged(app,e));
            app.power_rb = uiradiobutton(app.plot_type_bg,'Position',[5 10 90 20],'Text','Power','Value',true);
            app.amp_rb   = uiradiobutton(app.plot_type_bg,'Position',[100 10 90 20],'Text','Amplitude');

            % ---- Time series area (top right) ----
            app.time_series_1 = uiaxes(app.RootContainer,'Position',[335 640 780 155]);
            app.time_series_2 = uiaxes(app.RootContainer,'Position',[335 480 780 155]);
            app.plot_pp       = uiaxes(app.RootContainer,'Position',[1120 480 475 315]);
            app.plot_pp.Visible = 'off';

            % ---- WT / bispectrum pane ----
            app.wt_pane = uipanel(app.RootContainer,'Position',[335 0 1265 478],'Title','Analysis');

            % WT axes (mode 1-2)
            app.plot3d   = uiaxes(app.wt_pane,'Position',[5   5 870 460]);
            app.plot_pow = uiaxes(app.wt_pane,'Position',[885 5 375 460]);

            % Single bispectrum axes (mode 3-6) — placed at same x,y as plot3d
            app.bisp             = uiaxes(app.wt_pane,'Position',[5   5 560 460]);
            app.bisp_amp_axis    = uiaxes(app.wt_pane,'Position',[575 240 685 220]);
            app.bisp_phase_axis  = uiaxes(app.wt_pane,'Position',[575   5 685 230]);

            % All bispectra axes (mode 7)
            app.bispxxx_axis = uiaxes(app.wt_pane,'Position',[5   240 300 215]);
            app.bispppp_axis = uiaxes(app.wt_pane,'Position',[315 240 300 215]);
            app.bispxpp_axis = uiaxes(app.wt_pane,'Position',[5     5 300 230]);
            app.bisppxx_axis = uiaxes(app.wt_pane,'Position',[315   5 300 230]);
            app.wt_1         = uiaxes(app.wt_pane,'Position',[625 240 635 215]);
            app.wt_2         = uiaxes(app.wt_pane,'Position',[625   5 635 230]);
        end
    end

    %% Constructor / destructor
    methods (Access = public)
        function app = Bispectrum(parentContainer)
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
