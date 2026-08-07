%Version 1.02
%********************************************************************************
%************************** Filtering & Ridge Extraction GUI ********************
%********************************************************************************
%
% Migrated from GUIDE to App Designer (classdef).
% Compatible with MATLAB R2023a through R2026a.

classdef Filtering < matlab.apps.AppBase

    % ------------------------------------------------------------------ %
    %  UI component properties                                             %
    % ------------------------------------------------------------------ %
    properties (Access = public)
        UIFigure
        RootContainer   % parent for built components: UIFigure (standalone) or a uitab (embedded)
        OwnsFigure = true   % false when embedded into a shell app's uitab

        % Menus
        FileMenu, ResetGUIMenu, FileReadMenu, LoadSessionMenu
        SavePlotMenu, ExportViewMenu, OpenViewMenu
        SaveMenu, SaveCsvMenu, SaveMatMenu, SaveSessionMenu

        % Logos
        logo, nbmplogo

        % Panels
        TimeSeriesPanel, WtPane

        % Results tabs (replaces manual axes show/hide) and the axes within them
        ResultsTabGroup, TFTab, BandsTab, FourierTab
        time_series, plot_pp
        plot3d, plot_pow, cum_avg
        fourier_plot, amp_axis, phase_axis, freq_axis

        % Controls
        signal_list, interval_list
        status, transform_btn, filter_signal_btn, ridgecalc_btn
        xlim_field, ylim_field, length_field
        max_freq, min_freq, central_freq
        wind_type, preprocess, cutedges, kaisera
        kaiseraLabel   % small sub-panel wrapping the "a" label+field, shown only for Kaiser
        refresh_limits_btn, mark_interval_btn, add_interval_btn
        freq_1, freq_2
        fourier_scale

        % Button groups
        plot_type_bg, power_rb, amp_rb
        calc_type_bg, wav_rb, four_rb
    end

    % ------------------------------------------------------------------ %
    %  Data properties                                                     %
    % ------------------------------------------------------------------ %
    properties (Access = public)
        cmap, linecol, line2width = 2
        calc_type = 1
        plot_type = 2
        etype = 2   % 1=ridge, 2=bands

        sig, sig_cut, sig_pp
        sampling_freq
        freqarr, wopt
        amp_WT, pow_WT, amp_av, pow_av
        time_axis, time_axis_cut, time_axis_ds
        xl
        bands, recon
        extract_phase, extract_amp
        bands_iphi
        f1_cell, f2_cell
        peak_value, fc
        leg1, leg2

        c = 0
        h_wait
        it = 0
    end

    methods (Access = private)
        function idx = listboxIndex(~, lb)
            items = lb.Items;
            sel   = lb.Value;
            if ischar(sel), sel = {sel}; end
            idx = find(ismember(items, sel));
            if isempty(idx), idx = 1; end
        end

        function idx = dropdownIndex(~, dd)
            idx = find(strcmp(dd.Items, dd.Value), 1);
            if isempty(idx), idx = 1; end
        end

        function idx = resultTabIndex(app)
            % 1=Time-frequency, 2=Bands, 3=Fourier — mirrors the old
            % display_type dropdown's index, now driven by ResultsTabGroup.
            if app.ResultsTabGroup.SelectedTab == app.BandsTab
                idx = 2;
            elseif app.ResultsTabGroup.SelectedTab == app.FourierTab
                idx = 3;
            else
                idx = 1;
            end
        end

        function selectResultTab(app, idx)
            tabs = {app.TFTab, app.BandsTab, app.FourierTab};
            app.ResultsTabGroup.SelectedTab = tabs{idx};
        end

        function setListboxByIndex(~, lb, idx)
            if idx < 1 || idx > numel(lb.Items), return; end
            lb.Value = lb.Items{idx};
        end

        function initSettings(app)
            load('cmap.mat','cmap');
            app.cmap    = cmap;
            app.linecol = cmap([1,18,40,50,60,64,15],:);
            if app.OwnsFigure
                ss = get(groot,'Screensize');
                sw = ss(3); sh = ss(4);
                if sw < 1600 || sh < 860
                    app.UIFigure.Position = [0 0 sw sh];
                else
                    app.UIFigure.Position = [round((sw-1600)/2) round((sh-860)/2) 1600 860];
                end
            end
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
            % Which top-level view (Time-frequency/Bands/Fourier) is showing
            % is handled by ResultsTabGroup, not this function. This just
            % re-shows axes that fileReadMenuSelected hid on reset, and
            % toggles the TF tab's single-signal vs. all-signal sub-view.
            switch mode
                case 1  % single-signal TF
                    app.plot3d.Visible   = 'on';
                    app.plot_pow.Visible = 'on';
                    app.cum_avg.Visible  = 'off';
                case 2  % bands
                    app.amp_axis.Visible   = 'on';
                    app.freq_axis.Visible  = 'on';
                    app.phase_axis.Visible = 'on';
                case 3  % Fourier
                    app.fourier_plot.Visible = 'on';
                case 4  % all-signal average
                    app.plot3d.Visible   = 'off';
                    app.plot_pow.Visible = 'off';
                    app.cum_avg.Visible  = 'on';
            end
        end
    end

    % ------------------------------------------------------------------ %
    %  Callbacks                                                           %
    % ------------------------------------------------------------------ %
    methods (Access = private)

        function fileReadMenuSelected(app, ~)
            [app, A] = MODAreadcheck(app);
            if A ~= 1, return; end

            % Clear all overlapping axes
            allAxes = {app.plot3d, app.plot_pow, app.cum_avg, app.fourier_plot, app.amp_axis, app.phase_axis, app.freq_axis};
            for a = allAxes, cla(a{1},'reset'); a{1}.Visible = 'off'; end

            app.interval_list.Items = {};
            app.freq_1.Value = ''; app.freq_2.Value = '';
            if app.OwnsFigure
                app.ExportViewMenu.Enable = 'off';
                app.OpenViewMenu.Enable   = 'off';
            end

            % Clear data
            fields = {'freqarr','sig','sig_cut','f1_cell','f2_cell','extract_phase','extract_amp',...
                      'time_axis','pow_av','amp_av','pow_WT','amp_WT','bands','wopt',...
                      'time_axis_ds','sig_pp','sampling_freq','peak_value'};
            for f = fields, app.(f{1}) = []; end

            [app, sig] = MODAread(app, 0);
            if isequal(sig,0), return; end

            list = cell(size(sig,1)+1,1);
            for i = 1:size(sig,1), list{i} = sprintf('Signal %d',i); end
            list{end} = 'Average Plot (All)';
            app.signal_list.Items = list;
            app.setListboxByIndex(app.signal_list, 1);

            app.refreshLimitsCallback();
            app.detrendSignalCallback();
            app.status.Value = 'Data loaded. Proceed with transform.';
            app.transform_btn.Enable   = 'on';
        end

        function loadSessionMenuSelected(app, ~)
            app = MODAload(app);
        end

        function resetGUIMenuSelected(app, ~)
            Filtering;
        end

        function refreshLimitsCallback(app)
            x = app.time_series.XLim;
            y = app.time_series.YLim;
            xlim(app.plot_pp, x);
            t = x(2) - x(1);
            xi = x .* app.sampling_freq;
            xi(2) = min(xi(2), size(app.sig,2));
            xi(1) = max(xi(1), 1);
            app.sig_cut        = app.sig(:, xi(1):xi(2));
            app.time_axis_cut  = app.time_axis(xi(1):xi(2));
            app.xl             = [app.time_axis_cut(1) app.time_axis_cut(end)];
            app.xlim_field.Value  = sprintf('%s, %s', num2str(x(1)), num2str(x(2)));
            app.ylim_field.Value  = sprintf('%s, %s', num2str(y(1)), num2str(y(2)));
            app.length_field.Value = num2str(t);
        end

        function refreshLimitsBtnPushed(app, ~)
            app.refreshLimitsCallback();
        end

        function detrendSignalCallback(app)
            ppstat = app.dropdownIndex(app.preprocess);
            if ppstat == 2
                app.plot_pp.Visible = 'on';
                cla(app.plot_pp,'reset');
                sig_select = app.listboxIndex(app.signal_list);
                fmax = str2double(app.max_freq.Value);
                fmin = str2double(app.min_freq.Value);
                L    = size(app.sig_cut,2);
                app.sig_pp = NaN(size(app.sig_cut));
                for j = 1:size(app.sig_cut,1)
                    s  = app.sig_cut(j,:);
                    X  = (1:length(s))'/app.sampling_freq;
                    XM = ones(length(X),4);
                    for pn=1:3, CX=X.^pn; XM(:,pn+1)=(CX-mean(CX))/std(CX); end
                    s  = s(:);
                    w  = warning('off','all');
                    ns = s - XM*(pinv(XM)*s);
                    warning(w);
                    fx = fft(ns,L);
                    Nq = ceil((L+1)/2);
                    ff = [(0:Nq-1),-fliplr(1:L-Nq)]*app.sampling_freq/L;
                    ff = ff(:);
                    fx(abs(ff)<=max([fmin,app.sampling_freq/L]) | abs(ff)>=fmax)=0;
                    app.sig_pp(j,:) = ifft(fx)';
                end
                globalfontsize = 12;
                plot(app.plot_pp, app.time_axis_cut, app.sig_cut(sig_select,:), 'color', app.linecol(1,:));
                hold(app.plot_pp,'on');
                plot(app.plot_pp, app.time_axis_cut, app.sig_pp(sig_select,:), 'color', app.linecol(2,:));
                legend(app.plot_pp,{'Original','Pre-Processed'},'FontSize',globalfontsize,'Location','Best');
                xlim(app.plot_pp,[app.time_axis_cut(1) app.time_axis_cut(end)]);
                xlabel(app.plot_pp,'Time (s)');
                drawnow;
            else
                cla(app.plot_pp,'reset');
                app.plot_pp.Visible = 'off';
            end
        end

        function preprocessDropdownChanged(app, ~)
            sig_select = app.listboxIndex(app.signal_list);
            if sig_select == size(app.sig_cut,1)+1
                app.setListboxByIndex(app.signal_list, 1);
                app.detrendSignalCallback();
                app.displayTypeChanged([]);
            else
                app.detrendSignalCallback();
            end
        end

        function signalListChanged(app, ~)
            if isempty(app.sig), return; end
            sig_select = app.listboxIndex(app.signal_list);
            if sig_select ~= size(app.sig,1)+1 && numel(sig_select)==1
                plot(app.time_series, app.time_axis_cut, app.sig_cut(sig_select,:), 'color', app.linecol(1,:));
                xlim(app.time_series, app.xl);
                xlabel(app.time_series,'Time (s)');
                app.detrendSignalCallback();
                if ~isempty(app.freqarr), app.displayTypeChanged([]); end
            else
                app.displayTypeChanged([]);
            end
        end

        function calcTypeChanged(app, event)
            switch event.NewValue.Tag
                case 'wav'
                    app.calc_type = 1;
                    app.wind_type.Items = {'Lognorm','Morlet','Bump','','',''};
                    app.kaisera.Visible = 'off'; app.kaiseraLabel.Visible = 'off';
                case 'four'
                    app.calc_type = 2;
                    app.wind_type.Items = {'Hann','Gaussian','Blackman','Exp','Rect','Kaiser'};
            end
            drawnow;
        end

        function windTypeChanged(app, ~)
            if strcmp(app.wind_type.Value,'Kaiser')
                app.kaisera.Visible = 'on'; app.kaiseraLabel.Visible = 'on';
            else
                app.kaisera.Visible = 'off'; app.kaiseraLabel.Visible = 'off';
            end
        end

        function plotTypeChanged(app, event)
            switch event.NewValue.Tag
                case 'power', app.plot_type = 1;
                case 'amp',   app.plot_type = 2;
            end
            disp_select = app.resultTabIndex();
            if disp_select > 1, return; end
            app.displayTypeChanged([]);
        end

        function transformBtnPushed(app, ~)
            app.transform_btn.Enable   = 'off';
            app.filter_signal_btn.Enable = 'off';
            app.ridgecalc_btn.Enable   = 'off';

            try
                app.status.Value = 'Calculating Transform...'; drawnow;
                fmax = str2double(app.max_freq.Value);
                fmin = str2double(app.min_freq.Value);
                f0   = str2double(app.central_freq.Value);
                app.fc = f0;

                A = f0 <= 0.4;
                wtype = app.wind_type.Value;
                B = strcmp(wtype,'Bump');
                if (A+0)+(B+0)==2
                    errordlg('Bump wavelet requires f0 > 0.4.','Parameter Error');
                    app.transform_btn.Enable = 'on'; return;
                end
                if fmax > app.sampling_freq/2
                    errordlg(['Max freq cannot exceed Nyquist (' num2str(app.sampling_freq/2) ' Hz).'],'Parameter Error');
                    app.transform_btn.Enable = 'on'; return;
                end
                if app.calc_type==2 && isnan(fmin)
                    errordlg('Minimum frequency must be specified for WFT','Parameter Error');
                    app.transform_btn.Enable = 'on'; return;
                end
                if app.calc_type==1 && fmin <= 1/(length(app.sig_cut)/app.sampling_freq)
                    errordlg('WT minimum frequency too low.','Parameter Error');
                    app.transform_btn.Enable = 'on'; return;
                end

                if strcmp(wtype,'Kaiser')
                    a = str2double(app.kaisera.Value);
                    wtype = ['kaiser-' num2str(a)];
                end

                ppselect  = app.preprocess.Value;
                cutselect = app.cutedges.Value;

                fc_val = f0;
                if app.calc_type==2, fc_val = f0/fmin; end

                N = length(app.time_axis_cut);
                ss = max(get(groot,'Screensize'));
                ds = max(1, floor(N/ss));
                app.time_axis_ds = app.time_axis_cut(1:ds:end);

                app.h_wait = waitbar(0,'Calculating transform...','CreateCancelBtn','setappdata(gcbf,''canceling'',1)');
                setappdata(app.h_wait,'canceling',0);

                for p = 1:size(app.sig_cut,1)
                    if getappdata(app.h_wait,'canceling'), break; end
                    app.status.Value = sprintf('Calculating Transform of Signal %d/%d', p, size(app.sig_cut,1));
                    [WT, app.freqarr, app.wopt] = wtwrapper(app.sig_cut(p,:), app.sampling_freq, fc_val, fmin, fmax, app.calc_type, wtype, cutselect, ppselect);
                    app.amp_WT{p} = abs(WT(:,1:ds:end));
                    app.pow_WT{p} = abs(WT(:,1:ds:end)).^2;
                    app.amp_av{p} = nanmean(app.amp_WT{p},2);
                    app.pow_av{p} = nanmean(app.pow_WT{p},2);
                    waitbar(p/size(app.sig_cut,1), app.h_wait);
                end

                app.selectResultTab(1);
                app.displayTypeChanged([]);
                delete(app.h_wait);

                app.transform_btn.Enable    = 'on';
                app.filter_signal_btn.Enable = 'on';
                app.ridgecalc_btn.Enable     = 'on';
                app.mark_interval_btn.Enable = 'on';
                app.add_interval_btn.Enable  = 'on';
                if app.OwnsFigure
                    app.ExportViewMenu.Enable    = 'on';
                    app.OpenViewMenu.Enable      = 'on';
                    app.FileReadMenu.Enable      = 'off';
                end

            catch e
                errordlg(e.message,'Error');
                app.transform_btn.Enable = 'on';
                try, delete(app.h_wait); catch; end
                rethrow(e);
            end
        end

        function setDropdownByIndex(~, dd, idx)
            if idx >= 1 && idx <= numel(dd.Items)
                dd.Value = dd.Items{idx};
            end
        end

        function displayTypeChanged(app, ~)
            if isempty(app.freqarr) && isempty(app.bands), return; end
            disp_select = app.resultTabIndex();
            if isempty(app.interval_list.Items)
                int_select = 1;
            else
                int_select = app.listboxIndex(app.interval_list);
            end
            sig_select = app.listboxIndex(app.signal_list);

            globalfontsize = 12;

            if disp_select == 1 && sig_select ~= size(app.sig_cut,1)+1
                % Single signal TF display
                app.showPlot(1);
                cla(app.cum_avg,'reset'); cla(app.plot3d,'reset'); cla(app.plot_pow,'reset');
                cla(app.amp_axis,'reset'); cla(app.freq_axis,'reset'); cla(app.phase_axis,'reset');

                if app.plot_type == 1
                    WTpow = app.pow_WT{sig_select};
                    app.peak_value = max(WTpow(:)) + 0.1;
                    pcolor(app.plot3d, app.time_axis_ds, app.freqarr, WTpow);
                    plot(app.plot_pow, app.pow_av{sig_select}, app.freqarr, '-k', 'LineWidth',3, 'color', app.linecol(1,:));
                    xlabel(app.plot_pow,'Average Power');
                else
                    WTamp = app.amp_WT{sig_select};
                    app.peak_value = max(WTamp(:)) + 0.1;
                    pcolor(app.plot3d, app.time_axis_ds, app.freqarr, WTamp);
                    plot(app.plot_pow, app.amp_av{sig_select}, app.freqarr, '-k', 'LineWidth',3, 'color', app.linecol(1,:));
                    xlabel(app.plot_pow,'Average Amplitude');
                end
                colormap(app.plot3d,app.cmap); shading(app.plot3d,'interp');
                if app.calc_type == 1
                    app.plot3d.YScale = 'log'; app.plot_pow.YScale = 'log';
                else
                    app.plot3d.YScale = 'linear'; app.plot_pow.YScale = 'linear';
                end
                ylim(app.plot3d,[min(app.freqarr) max(app.freqarr)]);
                xlim(app.plot3d,[app.time_axis_ds(1) app.time_axis_ds(end)]);
                xlabel(app.plot3d,'Time (s)'); ylabel(app.plot3d,'Frequency (Hz)');
                ylabel(app.plot_pow,'Frequency (Hz)');
                ylim(app.plot_pow,[min(app.freqarr) max(app.freqarr)]);
                app.plot_pow.FontSize = globalfontsize;
                app.plot3d.FontSize   = globalfontsize;
                app.status.Value = 'Done Plotting';

            elseif disp_select == 1 && sig_select == size(app.sig_cut,1)+1 && ~isempty(app.freqarr)
                % All-signal average TF
                app.showPlot(4);
                cla(app.plot3d,'reset'); cla(app.plot_pow,'reset'); cla(app.cum_avg,'reset');
                hold(app.cum_avg,'on');
                if app.plot_type == 1
                    plot(app.cum_avg, app.freqarr, mean(cell2mat(app.pow_av),2),'-','Linewidth',3,'color',app.linecol(1,:));
                    plot(app.cum_avg, app.freqarr, median(cell2mat(app.pow_av),2),'--','Linewidth',3,'color',app.linecol(2,:));
                    ylabel(app.cum_avg,'Average Power');
                else
                    plot(app.cum_avg, app.freqarr, mean(cell2mat(app.amp_av),2),'-','Linewidth',3,'color',app.linecol(1,:));
                    plot(app.cum_avg, app.freqarr, median(cell2mat(app.amp_av),2),'--','Linewidth',3,'color',app.linecol(2,:));
                    ylabel(app.cum_avg,'Average Amplitude');
                end
                xlabel(app.cum_avg,'Frequency (Hz)');
                if app.calc_type==1, app.cum_avg.XScale='log'; else, app.cum_avg.XScale='linear'; end
                app.leg1 = {'Mean','Median'};
                legend(app.cum_avg,app.leg1);
                xlim(app.cum_avg,[app.freqarr(1) app.freqarr(end)]);

            elseif disp_select == 2 && (~isempty(app.bands) || ~isempty(app.recon))
                % Bands display
                if sig_select == size(app.sig_cut,1)+1
                    app.setListboxByIndex(app.signal_list, 1); sig_select = 1;
                end
                app.showPlot(2);
                cla(app.amp_axis,'reset'); cla(app.freq_axis,'reset'); cla(app.phase_axis,'reset');
                hold(app.amp_axis,'on'); hold(app.freq_axis,'on'); hold(app.phase_axis,'on');

                if ~isempty(int_select)
                    for j = 1:numel(int_select)
                        k = int_select(j);
                        if app.etype == 2 && ~isempty(app.bands)
                            plot(app.amp_axis,   app.time_axis_cut, abs(hilbert(app.bands{sig_select,k})), 'color', app.linecol(j,:));
                            plot(app.phase_axis, app.time_axis_cut, angle(hilbert(app.bands{sig_select,k})), 'color', app.linecol(j,:));
                        elseif app.etype == 1 && ~isempty(app.recon)
                            plot(app.amp_axis,   app.time_axis_cut, app.extract_amp{sig_select,k}, 'color', app.linecol(j,:));
                            plot(app.phase_axis, app.time_axis_cut, app.extract_phase{sig_select,k}, 'color', app.linecol(j,:));
                            if isfield(app,'bands_iphi') && ~isempty(app.bands_iphi)
                                plot(app.freq_axis, app.time_axis_cut, diff([0 unwrap(app.bands_iphi{sig_select,k})])*app.sampling_freq/(2*pi), 'color', app.linecol(j,:));
                            end
                        end
                    end
                end
                for ax = {app.amp_axis, app.phase_axis, app.freq_axis}
                    xlim(ax{1}, [app.time_axis_cut(1) app.time_axis_cut(end)]);
                end
                xlabel(app.phase_axis,'Time (s)');
                ylabel(app.amp_axis,'Amplitude'); ylabel(app.phase_axis,'Phase (rad)'); ylabel(app.freq_axis,'Inst. freq (Hz)');
                linkaxes([app.amp_axis app.phase_axis app.freq_axis app.time_series],'x');
                % Only time_series is meant to be dragged/zoomed directly —
                % the other three just mirror its x-range via linkaxes. Their
                % own toolbars/interactivity are disabled so a drag can't
                % accidentally start on one of them and trigger a second,
                % redundant synchronized redraw of the whole linked group
                % (this combination is what caused the "controls lag when
                % interacting with the graph" issue).
                for ax = {app.amp_axis, app.phase_axis, app.freq_axis}
                    disableDefaultInteractivity(ax{1});
                    ax{1}.Toolbar.Visible = 'off';
                end

            elseif disp_select == 3 && (~isempty(app.bands) || ~isempty(app.recon))
                % Fourier display
                app.showPlot(3);
                list = app.interval_list.Items;
                cla(app.fourier_plot,'reset');
                hold(app.fourier_plot,'on');

                ppselect = app.preprocess.Value;
                if strcmp(ppselect,'on') && ~isempty(app.sig_pp)
                    [ftorig, ft_freq] = Fourier(app.sig_pp(sig_select,:), app.sampling_freq);
                else
                    [ftorig, ft_freq] = Fourier(app.sig(sig_select,:), app.sampling_freq);
                end
                plot(app.fourier_plot, ft_freq, ftorig, 'linewidth',2, 'color', app.linecol(1,:));
                app.fourier_plot.Visible = 'on';
                app.fourier_plot.XScale  = 'log';
                app.fourier_plot.YScale  = 'log';
                app.fourier_plot.XLim    = [app.freqarr(1) app.freqarr(end)];
                app.leg2 = {'Original'};

                for j = 1:numel(list)
                    app.f1_cell{j} = list{j}(1:4);
                    app.f2_cell{j} = list{j}(10:min(13,length(list{j})));
                end

                legend(app.fourier_plot, app.leg2, 'FontSize', globalfontsize);

                if ~isempty(app.bands) && app.etype==2
                    for j = 1:numel(int_select)
                        k = int_select(j);
                        [ft, ft_freq] = Fourier(app.bands{sig_select,k}, app.sampling_freq);
                        plot(app.fourier_plot, ft_freq, ft, 'Linewidth',2, 'color', app.linecol(j+1,:));
                        app.leg2{j+1} = [app.f1_cell{k} ' - ' app.f2_cell{k} ' Hz'];
                        legend(app.fourier_plot, app.leg2, 'FontSize', globalfontsize);
                    end
                end

                xlabel(app.fourier_plot,'Frequency (Hz)','FontSize',globalfontsize);
                ylabel(app.fourier_plot,'FT Power','FontSize',globalfontsize);

                x = app.dropdownIndex(app.fourier_scale);
                if x==1, app.fourier_plot.XScale='log'; app.fourier_plot.YScale='log';
                else,     app.fourier_plot.XScale='linear'; app.fourier_plot.YScale='linear'; end
            end
        end

        function ridgecalcBtnPushed(app, ~)
            app.etype = 1;
            app = MODAridge_filter(app.UIFigure, [], app);
            app.displayTypeChanged([]);
        end

        function filterSignalBtnPushed(app, ~)
            app.etype = 2;
            app = MODAridge_filter(app.UIFigure, [], app);
            app.displayTypeChanged([]);
        end

        function markIntervalBtnPushed(app, ~)
            disp_select = app.resultTabIndex();
            sig_select  = app.listboxIndex(app.signal_list);
            if disp_select ~= 1, return; end

            if any(sig_select == size(app.sig,1)+1)
                hold(app.cum_avg,'on');
                [f,~] = ginput(1);
                app.freq_1.Value = num2str(f);
                yl = app.cum_avg.YLim;
                line(app.cum_avg,[f f],yl,'Color','k','LineStyle','--');
                [f,~] = ginput(1);
                app.freq_2.Value = num2str(f);
                yl = app.cum_avg.YLim;
                line(app.cum_avg,[f f],yl,'Color','k','LineStyle','--');
            else
                hold(app.plot3d,'on'); hold(app.plot_pow,'on');
                [f,~] = ginput(1);
                app.freq_1.Value = num2str(f);
                xl = app.plot3d.XLim;
                line(app.plot3d,[xl(1) xl(2)],[f f],[1 1],'Color','k','LineStyle','--');
                yl = app.plot_pow.XLim;
                line(app.plot_pow,[yl(1) yl(2)],[f f],[1 1],'Color','k','LineStyle','--');
                [f,~] = ginput(1);
                app.freq_2.Value = num2str(f);
                xl = app.plot3d.XLim;
                line(app.plot3d,[xl(1) xl(2)],[f f],[1 1],'Color','k','LineStyle','--');
                yl = app.plot_pow.XLim;
                line(app.plot_pow,[yl(1) yl(2)],[f f],[1 1],'Color','k','LineStyle','--');
            end
        end

        function addIntervalBtnPushed(app, ~)
            f1 = str2double(app.freq_1.Value);
            f2 = str2double(app.freq_2.Value);
            if isnan(f1) || isnan(f2)
                errordlg('Please mark or enter a frequency interval first.','Error'); return;
            end
            app.c = app.c + 1;
            fl = sprintf('%.4f - %.4f Hz', f1, f2);
            app.interval_list.Items{end+1} = fl;
            app.setListboxByIndex(app.interval_list, numel(app.interval_list.Items));
        end

        function maxFreqChanged(app, ~)
            app.detrendSignalCallback();
        end

        function minFreqChanged(app, ~)
            app.detrendSignalCallback();
        end

        function fourierScaleChanged(app, ~)
            x = app.dropdownIndex(app.fourier_scale);
            if x==1, app.fourier_plot.XScale='log'; app.fourier_plot.YScale='log';
            else,     app.fourier_plot.XScale='linear'; app.fourier_plot.YScale='linear'; end
        end

        % ---- Export current view (replaces the old per-plot Save-menu list) ----

        function axs = currentViewAxes(app)
            % Returns the axes currently visible in the selected result tab,
            % in display order, for export.
            switch app.resultTabIndex()
                case 1
                    if strcmp(app.cum_avg.Visible,'on')
                        axs = {app.cum_avg};
                    else
                        axs = {app.plot3d, app.plot_pow};
                    end
                case 2
                    axs = {app.amp_axis, app.phase_axis, app.freq_axis};
                otherwise
                    axs = {app.fourier_plot};
            end
        end

        function fig = buildViewFigure(app)
            % Builds a hidden figure containing copies of the axes in the
            % currently selected result tab, laid out left-to-right.
            axs = app.currentViewAxes();
            n = numel(axs);
            fig = figure('Visible','off','Position',[100 100 380*n 420]);
            for i = 1:n
                newAx = copyobj(axs{i}, fig);
                newAx.Units = 'normalized';
                newAx.Position = [(i-1)/n + 0.06/n, 0.12, 0.88/n, 0.8];
            end
        end

        function exportViewMenuSelected(app, ~)
            % Saves the current result view directly to a file the user
            % picks, without ever showing a new figure window.
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
            % Secondary option for users who want to keep tweaking a copy —
            % opens a normal, visible figure instead of saving directly.
            fig = app.buildViewFigure();
            fig.Visible = 'on';
        end

        % ---- Save ---------------------------------------------------

        function saveMatMenuSelected(app, ~)
            try
                [FileName,PathName] = uiputfile('*.mat','Save data as');
                if isequal(FileName,0), return; end
                save_location = fullfile(PathName,FileName);
                filter_data = app.buildFilterData();
                save(save_location,'filter_data');
            catch e, errordlg(e.message,'Error'); rethrow(e); end
        end

        function saveCsvMenuSelected(app, ~)
            try
                filter_data = app.buildFilterData();
                csvsavefolder(filter_data);
            catch e, errordlg(e.message,'Error'); rethrow(e); end
        end

        function saveSessionMenuSelected(app, ~)
            MODAsave(app);
        end

        function fd = buildFilterData(app)
            fd.sig          = app.sig;
            fd.sig_cut      = app.sig_cut;
            fd.sampling_freq = app.sampling_freq;
            fd.time_axis    = app.time_axis;
            fd.freqarr      = app.freqarr;
            if ~isempty(app.bands),         fd.bands         = app.bands;         end
            if ~isempty(app.extract_phase), fd.extract_phase = app.extract_phase; end
            if ~isempty(app.extract_amp),   fd.extract_amp   = app.extract_amp;   end
            if ~isempty(app.amp_WT),        fd.amp_WT        = app.amp_WT;        end
            if ~isempty(app.pow_WT),        fd.pow_WT        = app.pow_WT;        end
        end

        function UIFigureCloseRequest(app, ~)
            MODAclose(app.UIFigure, app);
        end
    end

    % ------------------------------------------------------------------ %
    %  Component creation                                                  %
    % ------------------------------------------------------------------ %
    methods (Access = private)
        function createComponents(app, parentContainer)
            % parentContainer: optional. Omit for a standalone window
            % (legacy behavior); pass a uitab to build onto it instead.
            W = 1600; H = 860;

            if nargin < 2 || isempty(parentContainer)
                app.UIFigure = uifigure('Visible','off','Position',[100 100 W H],'Resize','off','Name','MODA v1.01 Filtering');
                app.UIFigure.CloseRequestFcn = @(s,e) app.UIFigureCloseRequest(e);
                app.OwnsFigure    = true;
                app.RootContainer = app.UIFigure;
            else
                app.RootContainer = parentContainer;
                app.UIFigure      = ancestor(parentContainer, 'figure');
                app.OwnsFigure    = false;
            end

            % Menus (figure-level; only when this module owns the figure)
            if app.OwnsFigure
                app.FileMenu      = uimenu(app.UIFigure,'Text','File');
                app.ResetGUIMenu  = uimenu(app.FileMenu,'Text','Reset GUI','MenuSelectedFcn',@(s,e)app.resetGUIMenuSelected(e));
                app.FileReadMenu  = uimenu(app.FileMenu,'Text','Load time series','MenuSelectedFcn',@(s,e)app.fileReadMenuSelected(e));
                app.LoadSessionMenu = uimenu(app.FileMenu,'Text','Load session','MenuSelectedFcn',@(s,e)app.loadSessionMenuSelected(e));

                % Replaces the old list of ~10 "Save X Plot" menu items (most
                % of which had no MenuSelectedFcn wired at all) with two
                % actions that act on whatever result tab is currently showing.
                app.SavePlotMenu   = uimenu(app.UIFigure,'Text','Save plot');
                app.ExportViewMenu = uimenu(app.SavePlotMenu,'Text','Export current view...', ...
                    'MenuSelectedFcn',@(s,e)app.exportViewMenuSelected(e));
                app.OpenViewMenu   = uimenu(app.SavePlotMenu,'Text','Open current view in new figure', ...
                    'MenuSelectedFcn',@(s,e)app.openViewMenuSelected(e));

                app.SaveMenu        = uimenu(app.UIFigure,'Text','Save data');
                app.SaveCsvMenu     = uimenu(app.SaveMenu,'Text','Save .csv','Enable','off','MenuSelectedFcn',@(s,e)app.saveCsvMenuSelected(e));
                app.SaveMatMenu     = uimenu(app.SaveMenu,'Text','Save .mat','Enable','off','MenuSelectedFcn',@(s,e)app.saveMatMenuSelected(e));
                app.SaveSessionMenu = uimenu(app.SaveMenu,'Text','Save session','Enable','off','MenuSelectedFcn',@(s,e)app.saveSessionMenuSelected(e));
            end

            % Only when this module owns its figure — embedded in MODAApp's
            % tab, the logos would overlap the results panel instead of
            % adding anything (MODAApp already shows its own top-bar banner).
            if app.OwnsFigure
                app.anchorBrandingLogos();
            end

            % ---- Left control sidebar (consistent with the Coherence/
            % Bispectrum/TFA/Bayesian screens: one scrollable panel holding
            % every control, instead of controls scattered across 6
            % separate floating panels while results occupied the left) ----
            ctrlPanel = uipanel(app.RootContainer,'Position',[0 0 330 795],'Title','','Scrollable','on');

            yl = 750;
            uilabel(ctrlPanel,'Position',[5 yl 150 20],'Text','Select Data');
            app.signal_list = uilistbox(ctrlPanel,'Position',[5 yl-90 320 90],'Items',{}, ...
                'ValueChangedFcn',@(s,e)app.signalListChanged(e));

            yl = yl - 120;
            uilabel(ctrlPanel,'Position',[5 yl 320 20],'Text','Status:');
            app.status = uieditfield(ctrlPanel,'text','Position',[5 yl-24 320 22],'Value','Please Import Signal');

            yl = yl - 55;
            app.transform_btn = uibutton(ctrlPanel,'push','Position',[5 yl 320 28],'Text','Calculate Transform','Enable','off', ...
                'ButtonPushedFcn',@(s,e)app.transformBtnPushed(e));
            yl = yl - 34;
            app.ridgecalc_btn = uibutton(ctrlPanel,'push','Position',[5 yl 155 28],'Text','Extract ridge(s)','Enable','off', ...
                'ButtonPushedFcn',@(s,e)app.ridgecalcBtnPushed(e));
            app.filter_signal_btn = uibutton(ctrlPanel,'push','Position',[165 yl 155 28],'Text','Bandpass Filter','Enable','off', ...
                'ButtonPushedFcn',@(s,e)app.filterSignalBtnPushed(e));

            % Frequency params
            yl = yl - 40;
            uilabel(ctrlPanel,'Position',[5 yl 120 20],'Text','Max Freq (Hz):');
            app.max_freq = uieditfield(ctrlPanel,'text','Position',[130 yl 100 22],'ValueChangedFcn',@(s,e)app.maxFreqChanged(e));
            yl = yl - 30;
            uilabel(ctrlPanel,'Position',[5 yl 120 20],'Text','Min Freq (Hz):');
            app.min_freq = uieditfield(ctrlPanel,'text','Position',[130 yl 100 22],'ValueChangedFcn',@(s,e)app.minFreqChanged(e));
            yl = yl - 30;
            uilabel(ctrlPanel,'Position',[5 yl 120 20],'Text','Resolution:');
            app.central_freq = uieditfield(ctrlPanel,'text','Position',[130 yl 100 22]);

            % Window / preprocessing options
            yl = yl - 40;
            uilabel(ctrlPanel,'Position',[5 yl 140 20],'Text','Window Type:');
            app.wind_type = uidropdown(ctrlPanel,'Position',[148 yl 155 22], ...
                'Items',{'Lognorm','Morlet','Bump','','',''},'ValueChangedFcn',@(s,e)app.windTypeChanged(e));
            yl = yl - 30;
            uilabel(ctrlPanel,'Position',[5 yl 140 20],'Text','Preprocess:');
            app.preprocess = uidropdown(ctrlPanel,'Position',[148 yl 155 22],'Items',{'off','on'}, ...
                'ValueChangedFcn',@(s,e)app.preprocessDropdownChanged(e));
            yl = yl - 30;
            uilabel(ctrlPanel,'Position',[5 yl 140 20],'Text','Cut Edges:');
            app.cutedges = uidropdown(ctrlPanel,'Position',[148 yl 155 22],'Items',{'off','on'});
            yl = yl - 30;
            % Kaiser "a" parameter — only relevant when Window Type = Kaiser
            app.kaiseraLabel = uilabel(ctrlPanel,'Position',[5 yl 140 20],'Text','Kaiser a:','Visible','off');
            app.kaisera = uieditfield(ctrlPanel,'text','Value','3','Position',[148 yl 100 22],'Visible','off');

            % Plot type / calc type — panel tall enough that the title bar
            % doesn't overlap the top radio button.
            yl = yl - 88;
            app.plot_type_bg = uibuttongroup(ctrlPanel,'Position',[5 yl 155 78],'Title','Plot Type', ...
                'SelectionChangedFcn',@(bg,ev)app.plotTypeChanged(ev));
            app.power_rb = uiradiobutton(app.plot_type_bg,'Position',[5 30 90 20],'Text','Power','Value',true,'Tag','power');
            app.amp_rb   = uiradiobutton(app.plot_type_bg,'Position',[5 5  90 20],'Text','Amplitude','Tag','amp');

            app.calc_type_bg = uibuttongroup(ctrlPanel,'Position',[165 yl 155 78],'Title','Calc Type', ...
                'SelectionChangedFcn',@(bg,ev)app.calcTypeChanged(ev));
            app.wav_rb  = uiradiobutton(app.calc_type_bg,'Position',[5 30 90 20],'Text','WT','Value',true,'Tag','wav');
            app.four_rb = uiradiobutton(app.calc_type_bg,'Position',[5 5  90 20],'Text','WFT','Tag','four');

            % Limits
            yl = yl - 80;
            uilabel(ctrlPanel,'Position',[5 yl 75 20],'Text','X Limits:');
            app.xlim_field = uieditfield(ctrlPanel,'text','Position',[85 yl 235 22]);
            yl = yl - 30;
            uilabel(ctrlPanel,'Position',[5 yl 75 20],'Text','Y Limits:');
            app.ylim_field = uieditfield(ctrlPanel,'text','Position',[85 yl 235 22]);
            yl = yl - 30;
            uilabel(ctrlPanel,'Position',[5 yl 60 20],'Text','Length:');
            app.length_field = uieditfield(ctrlPanel,'text','Position',[70 yl 100 22]);
            app.refresh_limits_btn = uibutton(ctrlPanel,'push','Position',[180 yl 100 22],'Text','Refresh', ...
                'ButtonPushedFcn',@(s,e)app.refreshLimitsBtnPushed(e));

            % Interval marking (mark/add + freq fields) + interval list
            yl = yl - 40;
            app.mark_interval_btn = uibutton(ctrlPanel,'push','Position',[5 yl 155 26],'Text','Mark region','Enable','off', ...
                'ButtonPushedFcn',@(s,e)app.markIntervalBtnPushed(e));
            app.add_interval_btn  = uibutton(ctrlPanel,'push','Position',[165 yl 155 26],'Text','Add marked region','Enable','off', ...
                'ButtonPushedFcn',@(s,e)app.addIntervalBtnPushed(e));
            yl = yl - 32;
            uilabel(ctrlPanel,'Position',[5 yl 70 20],'Text','Frequency:');
            app.freq_1 = uieditfield(ctrlPanel,'text','Position',[80 yl 110 22]);
            app.freq_2 = uieditfield(ctrlPanel,'text','Position',[200 yl 110 22]);
            yl = yl - 30;
            uilabel(ctrlPanel,'Position',[5 yl 100 18],'Text','Interval List');
            app.interval_list = uilistbox(ctrlPanel,'Position',[5 yl-80 320 80],'Items',{});

            % ---- Time series panel (top right) — main signal preview plus a
            % small pre/post-processing comparison plot alongside it ----
            app.TimeSeriesPanel = uipanel(app.RootContainer,'Position',[330 500 1270 355],'Title','Time Series');
            app.time_series = uiaxes(app.TimeSeriesPanel,'Position',[5 5 900 325]);
            app.plot_pp     = uiaxes(app.TimeSeriesPanel,'Position',[915 5 350 325]);

            % ---- WT pane (bottom right) — results tab group (replaces the
            % old manual overlapping-axes show/hide with real navigable tabs) ----
            % No panel Title here — the ResultsTabGroup below already
            % provides its own header (Time-frequency/Bands/Fourier tabs),
            % so a panel title would collide with it.
            app.WtPane = uipanel(app.RootContainer,'Position',[330 0 1270 500],'BorderType','none');
            PW = 1270; PH = 500;

            app.ResultsTabGroup = uitabgroup(app.WtPane,'Position',[0 0 PW PH], ...
                'SelectionChangedFcn',@(s,e)app.displayTypeChanged(e));
            app.TFTab      = uitab(app.ResultsTabGroup,'Title','Time-frequency');
            app.BandsTab   = uitab(app.ResultsTabGroup,'Title','Bands');
            app.FourierTab = uitab(app.ResultsTabGroup,'Title','Fourier');

            app.plot_pow = uiaxes(app.TFTab,'Position',round([0.7813*PW, 0.1217*PH, 0.2017*PW, 0.8499*PH]));
            app.plot3d   = uiaxes(app.TFTab,'Position',round([0.0703*PW, 0.1228*PH, 0.6317*PW, 0.849*PH]));
            app.cum_avg  = uiaxes(app.TFTab,'Position',round([0.0534*PW, 0.1174*PH, 0.9388*PW, 0.8397*PH]));
            app.cum_avg.Visible = 'off';

            app.amp_axis   = uiaxes(app.BandsTab,'Position',round([0.0557*PW, 0.7336*PH, 0.7653*PW, 0.2393*PH]));
            app.phase_axis = uiaxes(app.BandsTab,'Position',round([0.0557*PW, 0.1422*PH, 0.7653*PW, 0.2393*PH]));
            app.freq_axis  = uiaxes(app.BandsTab,'Position',round([0.0557*PW, 0.4379*PH, 0.7653*PW, 0.2393*PH]));

            app.fourier_plot  = uiaxes(app.FourierTab,'Position',round([0.0534*PW, 0.1151*PH, 0.9388*PW, 0.8397*PH]));
            app.fourier_scale = uidropdown(app.FourierTab,'Items',{'Log','Linear'},'Position',round([0.02*PW, 0.02*PH, 0.13*PW, 0.05*PH]),'ValueChangedFcn',@(s,e)app.fourierScaleChanged(e));

            if app.OwnsFigure
                app.UIFigure.Visible = 'on';
            end
        end
    end

    methods (Access = public)
        function app = Filtering(parentContainer)
            % parentContainer: optional. Omit for a standalone window
            % (legacy behavior); pass a uitab to build onto it instead.
            if nargin < 1
                parentContainer = [];
            end
            createComponents(app, parentContainer);
            registerApp(app, app.UIFigure);
            runStartupFcn(app, @startupFcn);
            if nargout == 0, clear app; end
        end

        function delete(app)
            if app.OwnsFigure && isvalid(app.UIFigure)
                delete(app.UIFigure);
            end
        end
    end

    methods (Access = private)
        function startupFcn(app)
            app.initSettings();
            app.c = 0; app.etype = 2;
            disabledItems = {app.mark_interval_btn, app.add_interval_btn, ...
                             app.filter_signal_btn, app.ridgecalc_btn, app.transform_btn};
            if app.OwnsFigure
                % Menus only exist when this module owns its figure — see createComponents
                disabledItems = [disabledItems, {app.ExportViewMenu, app.OpenViewMenu, ...
                                 app.SaveCsvMenu, app.SaveMatMenu, app.SaveSessionMenu}];
            end
            for item = disabledItems, item{1}.Enable = 'off'; end
        end
    end
end
