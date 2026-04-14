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

        % Menus
        FileMenu, ResetGUIMenu, FileReadMenu, LoadSessionMenu
        SavePlotMenu
        PlotTSMenu, Save3dplotMenu, SaveBothMenu, SaveAvgMenu, SaveMmMenu
        SaveFiltSigPlotMenu, SaveRidgePlotMenu, SavePhasePlotMenu
        AllFiltPlotMenu, SaveFourierMenu
        SaveMenu, SaveCsvMenu, SaveMatMenu, SaveSessionMenu

        % Logos
        logo, nbmplogo

        % Panels
        TimeSeriesPanel, WtPane, FreqParamsPanel
        AdvancedPanel, StatusPanel, LimitsPanel, IntervalPanel

        % Axes (many are overlapping in WtPane)
        time_series, plot_pp
        plot3d, plot_pow, cum_avg
        fourier_plot, amp_axis, phase_axis, freq_axis

        % Controls
        signal_list, interval_list
        status, transform_btn, filter_signal_btn, ridgecalc_btn
        xlim_field, ylim_field, length_field
        max_freq, min_freq, central_freq
        wind_type, preprocess, cutedges, kaisera
        refresh_limits_btn, mark_interval_btn, add_interval_btn
        freq_1, freq_2
        display_type, fourier_scale

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

        function setListboxByIndex(~, lb, idx)
            if idx < 1 || idx > numel(lb.Items), return; end
            lb.Value = lb.Items{idx};
        end

        function initSettings(app)
            load('cmap.mat','cmap');
            app.cmap    = cmap;
            app.linecol = cmap([1,18,40,50,60,64,15],:);
            ss = get(groot,'Screensize');
            sw = ss(3); sh = ss(4);
            if sw < 1600 || sh < 860
                app.UIFigure.Position = [0 0 sw sh];
            else
                app.UIFigure.Position = [round((sw-1600)/2) round((sh-860)/2) 1600 860];
            end
            try, img=imread('physicslogo.png'); image(app.logo,img); axis(app.logo,'off'); axis(app.logo,'image'); catch; end
            try, img=imread('MODAbanner5.png');  image(app.nbmplogo,img); axis(app.nbmplogo,'off'); axis(app.nbmplogo,'image'); catch; end
        end

        function showPlot(app, mode)
            % mode: 1=TF, 2=Bands, 3=Fourier, 4=Average
            axList = {app.plot3d, app.plot_pow, app.cum_avg, app.fourier_plot, app.amp_axis, app.phase_axis, app.freq_axis};
            for a = axList, a{1}.Visible = 'off'; end

            switch mode
                case 1  % single TF
                    app.plot3d.Visible   = 'on';
                    app.plot_pow.Visible = 'on';
                case 2  % bands
                    app.amp_axis.Visible   = 'on';
                    app.freq_axis.Visible  = 'on';
                    app.phase_axis.Visible = 'on';
                case 3  % Fourier
                    app.fourier_plot.Visible = 'on';
                case 4  % average
                    app.cum_avg.Visible = 'on';
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
            app.display_type.Enable = 'off';
            app.freq_1.Value = ''; app.freq_2.Value = '';

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
            app.PlotTSMenu.Enable      = 'on';
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
                    app.kaisera.Enable = 'off';
                case 'four'
                    app.calc_type = 2;
                    app.wind_type.Items = {'Hann','Gaussian','Blackman','Exp','Rect','Kaiser'};
            end
            drawnow;
        end

        function windTypeChanged(app, ~)
            if strcmp(app.wind_type.Value,'Kaiser')
                app.kaisera.Enable = 'on';
            else
                app.kaisera.Enable = 'off';
            end
        end

        function plotTypeChanged(app, event)
            switch event.NewValue.Tag
                case 'power', app.plot_type = 1;
                case 'amp',   app.plot_type = 2;
            end
            disp_select = app.dropdownIndex(app.display_type);
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

                app.display_type.Enable = 'on';
                app.setDropdownByIndex(app.display_type, 1);
                app.displayTypeChanged([]);
                delete(app.h_wait);

                app.transform_btn.Enable    = 'on';
                app.filter_signal_btn.Enable = 'on';
                app.ridgecalc_btn.Enable     = 'on';
                app.Save3dplotMenu.Enable    = 'on';
                app.SaveBothMenu.Enable      = 'on';
                app.SaveAvgMenu.Enable       = 'on';
                app.mark_interval_btn.Enable = 'on';
                app.add_interval_btn.Enable  = 'on';
                app.FileReadMenu.Enable      = 'off';

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
            disp_select = app.dropdownIndex(app.display_type);
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
                app.Save3dplotMenu.Enable = 'on'; app.SaveBothMenu.Enable = 'on';
                app.SaveAvgMenu.Enable = 'on'; app.SaveMmMenu.Enable = 'off';
                app.SaveFiltSigPlotMenu.Enable = 'off'; app.SaveRidgePlotMenu.Enable = 'off';
                app.SavePhasePlotMenu.Enable = 'off'; app.AllFiltPlotMenu.Enable = 'off';
                app.SaveFourierMenu.Enable = 'off'; app.fourier_scale.Visible = 'off';
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
                app.Save3dplotMenu.Enable = 'off'; app.SaveBothMenu.Enable = 'off';
                app.SaveAvgMenu.Enable = 'off'; app.SaveMmMenu.Enable = 'on';
                app.SaveFiltSigPlotMenu.Enable = 'off'; app.SaveFourierMenu.Enable = 'off';
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
                app.SaveFiltSigPlotMenu.Enable = 'on'; app.SaveRidgePlotMenu.Enable = 'on';
                app.SavePhasePlotMenu.Enable   = 'on'; app.AllFiltPlotMenu.Enable = 'on';
                app.Save3dplotMenu.Enable = 'off'; app.SaveBothMenu.Enable = 'off';
                app.SaveFourierMenu.Enable = 'off'; app.fourier_scale.Visible = 'off';
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

            elseif disp_select == 3 && (~isempty(app.bands) || ~isempty(app.recon))
                % Fourier display
                app.showPlot(3);
                app.SaveFourierMenu.Enable = 'on'; app.fourier_scale.Visible = 'on';
                app.Save3dplotMenu.Enable = 'off'; app.SaveBothMenu.Enable = 'off';
                app.SaveFiltSigPlotMenu.Enable = 'off'; app.AllFiltPlotMenu.Enable = 'off';
                app.SaveMmMenu.Enable = 'off';
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
            disp_select = app.dropdownIndex(app.display_type);
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
        function createComponents(app)
            W = 1600; H = 860;
            app.UIFigure = uifigure('Visible','off','Position',[100 100 W H],'Name','MODA v1.01 Filtering');
            app.UIFigure.CloseRequestFcn = @(s,e) app.UIFigureCloseRequest(e);

            % Menus
            app.FileMenu      = uimenu(app.UIFigure,'Text','File');
            app.ResetGUIMenu  = uimenu(app.FileMenu,'Text','Reset GUI','MenuSelectedFcn',@(s,e)app.resetGUIMenuSelected(e));
            app.FileReadMenu  = uimenu(app.FileMenu,'Text','Load time series','MenuSelectedFcn',@(s,e)app.fileReadMenuSelected(e));
            app.LoadSessionMenu = uimenu(app.FileMenu,'Text','Load session','MenuSelectedFcn',@(s,e)app.loadSessionMenuSelected(e));

            app.SavePlotMenu          = uimenu(app.UIFigure,'Text','Save plot');
            app.PlotTSMenu            = uimenu(app.SavePlotMenu,'Text','Plot time series','Enable','off');
            app.Save3dplotMenu        = uimenu(app.SavePlotMenu,'Text','Save TF plot','Enable','off');
            app.SaveBothMenu          = uimenu(app.SavePlotMenu,'Text','Save TF + avg','Enable','off');
            app.SaveAvgMenu           = uimenu(app.SavePlotMenu,'Text','Save avg plot','Enable','off');
            app.SaveMmMenu            = uimenu(app.SavePlotMenu,'Text','Save mean/median','Enable','off');
            app.SaveFiltSigPlotMenu   = uimenu(app.SavePlotMenu,'Text','Save filtered signal','Enable','off');
            app.SaveRidgePlotMenu     = uimenu(app.SavePlotMenu,'Text','Save ridge plot','Enable','off');
            app.SavePhasePlotMenu     = uimenu(app.SavePlotMenu,'Text','Save phase plot','Enable','off');
            app.AllFiltPlotMenu       = uimenu(app.SavePlotMenu,'Text','All filtered signals','Enable','off');
            app.SaveFourierMenu       = uimenu(app.SavePlotMenu,'Text','Save Fourier plot','Enable','off');

            app.SaveMenu        = uimenu(app.UIFigure,'Text','Save data');
            app.SaveCsvMenu     = uimenu(app.SaveMenu,'Text','Save .csv','Enable','off','MenuSelectedFcn',@(s,e)app.saveCsvMenuSelected(e));
            app.SaveMatMenu     = uimenu(app.SaveMenu,'Text','Save .mat','Enable','off','MenuSelectedFcn',@(s,e)app.saveMatMenuSelected(e));
            app.SaveSessionMenu = uimenu(app.SaveMenu,'Text','Save session','Enable','off','MenuSelectedFcn',@(s,e)app.saveSessionMenuSelected(e));

            % Logos
            app.logo     = uiaxes(app.UIFigure,'Position',round([0.0038*W, 0.9188*H, 0.2123*W, 0.0712*H]));
            app.nbmplogo = uiaxes(app.UIFigure,'Position',round([0.2254*W, 0.9217*H, 0.4769*W, 0.0613*H]));
            app.logo.Toolbar.Visible = 'off'; app.nbmplogo.Toolbar.Visible = 'off';

            % Time series panel
            app.TimeSeriesPanel = uipanel(app.UIFigure,'Position',round([0.0085*W, 0.7094*H, 0.6069*W, 0.198*H]),'BorderType','none');
            TPW = round(0.6069*W); TPH = round(0.198*H);
            app.time_series = uiaxes(app.TimeSeriesPanel,'Position',round([0.064*TPW, 0.289*TPH, 0.875*TPW, 0.607*TPH]));

            % WT pane (all overlapping axes)
            app.WtPane = uipanel(app.UIFigure,'Position',round([0.0085*W, 0.0698*H, 0.6946*W, 0.6439*H]),'BorderType','none');
            PW = round(0.6946*W); PH = round(0.6439*H);
            app.plot_pow     = uiaxes(app.WtPane,'Position',round([0.7813*PW, 0.1217*PH, 0.2017*PW, 0.8499*PH]));
            app.plot3d       = uiaxes(app.WtPane,'Position',round([0.0703*PW, 0.1228*PH, 0.6317*PW, 0.849*PH]));
            app.cum_avg      = uiaxes(app.WtPane,'Position',round([0.0534*PW, 0.1174*PH, 0.9388*PW, 0.8397*PH]));
            app.fourier_plot = uiaxes(app.WtPane,'Position',round([0.0534*PW, 0.1151*PH, 0.9388*PW, 0.8397*PH]));
            app.amp_axis     = uiaxes(app.WtPane,'Position',round([0.0557*PW, 0.7336*PH, 0.7653*PW, 0.2393*PH]));
            app.phase_axis   = uiaxes(app.WtPane,'Position',round([0.0557*PW, 0.1422*PH, 0.7653*PW, 0.2393*PH]));
            app.freq_axis    = uiaxes(app.WtPane,'Position',round([0.0557*PW, 0.4379*PH, 0.7653*PW, 0.2393*PH]));

            % display_type and fourier_scale dropdowns (inside WtPane)
            app.display_type  = uidropdown(app.WtPane,'Items',{'Time-frequency','Bands','Fourier'},'Enable','off','Position',round([0.0111*PW, 0.0154*PH, 0.13*PW, 0.0463*PH]),'ValueChangedFcn',@(s,e)app.displayTypeChanged(e));
            app.fourier_scale = uidropdown(app.WtPane,'Items',{'Log','Linear'},'Visible','off','Position',round([0.1446*PW, 0.0158*PH, 0.099*PW, 0.0451*PH]),'ValueChangedFcn',@(s,e)app.fourierScaleChanged(e));

            % Initially hide most axes
            for ax = {app.cum_avg, app.fourier_plot, app.amp_axis, app.phase_axis, app.freq_axis}
                ax{1}.Visible = 'off';
            end

            % Freq params panel
            app.FreqParamsPanel = uipanel(app.UIFigure,'Position',round([0.8223*W, 0.6268*H, 0.17*W, 0.359*H]),'BorderType','none');
            FPW = round(0.17*W); FPH = round(0.359*H);
            uilabel(app.FreqParamsPanel,'Text','Max Freq (Hz)','Position',round([0.048*FPW, 0.833*FPH, 0.367*FPW, 0.082*FPH]));
            app.max_freq = uieditfield(app.FreqParamsPanel,'text','Position',round([0.493*FPW, 0.827*FPH, 0.309*FPW, 0.130*FPH]),'ValueChangedFcn',@(s,e)app.maxFreqChanged(e));
            uilabel(app.FreqParamsPanel,'Text','Min Freq (Hz)','Position',round([0.048*FPW, 0.684*FPH, 0.367*FPW, 0.087*FPH]));
            app.min_freq = uieditfield(app.FreqParamsPanel,'text','Position',round([0.493*FPW, 0.649*FPH, 0.309*FPW, 0.135*FPH]),'ValueChangedFcn',@(s,e)app.minFreqChanged(e));
            uilabel(app.FreqParamsPanel,'Text','Resolution','Position',round([0.063*FPW, 0.462*FPH, 0.382*FPW, 0.091*FPH]));
            app.central_freq = uieditfield(app.FreqParamsPanel,'text','Position',round([0.493*FPW, 0.462*FPH, 0.309*FPW, 0.144*FPH]));

            % Plot type button group
            BG1P = uipanel(app.FreqParamsPanel,'Position',round([0.034*FPW, 0.029*FPH, 0.449*FPW, 0.389*FPH]),'BorderType','none');
            app.plot_type_bg = uibuttongroup(BG1P,'Position',[0 0 round(0.449*FPW) round(0.389*FPH)],'SelectionChangedFcn',@(bg,ev)app.plotTypeChanged(ev));
            BG1PW = round(0.449*FPW); BG1PH = round(0.389*FPH);
            app.power_rb = uiradiobutton(app.plot_type_bg,'Text','Power',    'Tag','power','Position',[round(0.12*BG1PW) round(0.09*BG1PH) round(1.12*BG1PW) round(0.30*BG1PH)]);
            app.amp_rb   = uiradiobutton(app.plot_type_bg,'Text','Amplitude','Tag','amp',  'Position',[round(0.12*BG1PW) round(0.48*BG1PH) round(1.15*BG1PW) round(0.30*BG1PH)]);

            % Calc type button group
            BG2P = uipanel(app.FreqParamsPanel,'Position',round([0.575*FPW, 0.024*FPH, 0.391*FPW, 0.389*FPH]),'BorderType','none');
            app.calc_type_bg = uibuttongroup(BG2P,'Position',[0 0 round(0.391*FPW) round(0.389*FPH)],'SelectionChangedFcn',@(bg,ev)app.calcTypeChanged(ev));
            BG2PW = round(0.391*FPW); BG2PH = round(0.389*FPH);
            app.wav_rb  = uiradiobutton(app.calc_type_bg,'Text','WT', 'Tag','wav', 'Position',[10 round(0.60*BG2PH) round(0.67*BG2PW) round(0.25*BG2PH)]);
            app.four_rb = uiradiobutton(app.calc_type_bg,'Text','WFT','Tag','four','Position',[10 round(0.10*BG2PH) round(0.67*BG2PW) round(0.25*BG2PH)]);

            % Advanced options panel
            app.AdvancedPanel = uipanel(app.UIFigure,'Position',round([0.7246*W, 0.0869*H, 0.26*W, 0.4145*H]),'BorderType','none');
            APW = round(0.26*W); APH = round(0.4145*H);
            app.plot_pp = uiaxes(app.AdvancedPanel,'Position',round([0.0915*APW, 0.1571*APH, 0.8323*APW, 0.3286*APH]));
            uilabel(app.AdvancedPanel,'Text','Window Type','Position',round([0.116*APW, 0.871*APH, 0.265*APW, 0.046*APH]));
            app.wind_type = uidropdown(app.AdvancedPanel,'Items',{'Lognorm','Morlet','Bump','','',''},'Position',round([0.381*APW, 0.853*APH, 0.457*APW, 0.082*APH]),'ValueChangedFcn',@(s,e)app.windTypeChanged(e));
            uilabel(app.AdvancedPanel,'Text','Preprocess','Position',round([0.140*APW, 0.761*APH, 0.217*APW, 0.046*APH]));
            app.preprocess = uidropdown(app.AdvancedPanel,'Items',{'off','on'},'Position',round([0.381*APW, 0.746*APH, 0.457*APW, 0.082*APH]),'ValueChangedFcn',@(s,e)app.preprocessDropdownChanged(e));
            uilabel(app.AdvancedPanel,'Text','Cut Edges','Position',round([0.140*APW, 0.650*APH, 0.217*APW, 0.050*APH]));
            app.cutedges   = uidropdown(app.AdvancedPanel,'Items',{'off','on'},'Position',round([0.381*APW, 0.639*APH, 0.457*APW, 0.082*APH]));
            uilabel(app.AdvancedPanel,'Text','Comparison before and after preprocessing','Position',round([0.021*APW, 0.504*APH, 0.954*APW, 0.061*APH]));
            uilabel(app.AdvancedPanel,'Text','a','Position',round([0.872*APW, 0.871*APH, 0.052*APW, 0.054*APH]));
            app.kaisera    = uieditfield(app.AdvancedPanel,'text','Value','3','Enable','off','Position',round([0.924*APW, 0.857*APH, 0.064*APW, 0.079*APH]));

            % Transform / filter / ridge buttons
            app.transform_btn    = uibutton(app.UIFigure,'Text','Calculate Transform','Enable','off','Position',round([0.7223*W, 0.0157*H, 0.0985*W, 0.0442*H]),'ButtonPushedFcn',@(s,e)app.transformBtnPushed(e));
            app.filter_signal_btn= uibutton(app.UIFigure,'Text','Bandpass Filter','Enable','off','Position',round([0.9108*W, 0.0157*H, 0.0823*W, 0.0442*H]),'ButtonPushedFcn',@(s,e)app.filterSignalBtnPushed(e));
            app.ridgecalc_btn    = uibutton(app.UIFigure,'Text','Extract ridge(s)','Enable','off','Position',round([0.8254*W, 0.0157*H, 0.0815*W, 0.0442*H]),'ButtonPushedFcn',@(s,e)app.ridgecalcBtnPushed(e));

            % Status panel
            app.StatusPanel = uipanel(app.UIFigure,'Position',round([0.0115*W, -0.0014*H, 0.6885*W, 0.0712*H]),'BorderType','none');
            SPW = round(0.6885*W); SPH = round(0.0712*H);
            uilabel(app.StatusPanel,'Text','Status:','Position',round([0.034*SPW, 0.447*SPH, 0.058*SPW, 0.412*SPH]));
            app.status = uieditfield(app.StatusPanel,'text','Value','Please Import Signal','Position',round([0.118*SPW, 0.322*SPH, 0.870*SPW, 0.593*SPH]));

            % Signal + interval lists
            uilabel(app.UIFigure,'Text','Select Data','Position',round([0.6241*W, 0.8957*H, 0.0793*W, 0.0219*H]));
            app.signal_list   = uilistbox(app.UIFigure,'Items',{},'Position',round([0.6233*W, 0.7155*H, 0.08*W, 0.1763*H]),'ValueChangedFcn',@(s,e)app.signalListChanged(e));
            uilabel(app.UIFigure,'Text','Interval List','Position',round([0.7254*W, 0.7883*H, 0.0923*W, 0.0214*H]));
            app.interval_list = uilistbox(app.UIFigure,'Items',{},'Position',round([0.7254*W, 0.6268*H, 0.0931*W, 0.1581*H]));

            % Interval panel (mark/add + freq fields)
            app.IntervalPanel = uipanel(app.UIFigure,'Position',round([0.7246*W, 0.5114*H, 0.2631*W, 0.1026*H]),'BorderType','none');
            IPW = round(0.2631*W); IPH = round(0.1026*H);
            app.mark_interval_btn = uibutton(app.IntervalPanel,'Text','Mark region','Enable','off','Position',round([0.740*IPW, 0.600*IPH, 0.246*IPW, 0.350*IPH]),'ButtonPushedFcn',@(s,e)app.markIntervalBtnPushed(e));
            app.add_interval_btn  = uibutton(app.IntervalPanel,'Text','Add marked region','Enable','off','Position',round([0.743*IPW, 0.183*IPH, 0.240*IPW, 0.350*IPH]),'ButtonPushedFcn',@(s,e)app.addIntervalBtnPushed(e));
            uilabel(app.IntervalPanel,'Text','Frequency','Position',round([0.006*IPW, 0.217*IPH, 0.216*IPW, 0.600*IPH]));
            app.freq_1 = uieditfield(app.IntervalPanel,'text','Position',round([0.222*IPW, 0.233*IPH, 0.145*IPW, 0.600*IPH]));
            uilabel(app.IntervalPanel,'Text','Frequency','Position',round([0.367*IPW, 0.283*IPH, 0.201*IPW, 0.533*IPH]));
            app.freq_2 = uieditfield(app.IntervalPanel,'text','Position',round([0.568*IPW, 0.233*IPH, 0.154*IPW, 0.600*IPH]));

            % Limits panel
            app.LimitsPanel = uipanel(app.UIFigure,'Position',round([0.7231*W, 0.8134*H, 0.0977*W, 0.1709*H]),'BorderType','none');
            LPW = round(0.0977*W); LPH = round(0.1709*H);
            uilabel(app.LimitsPanel,'Text','Xlim','Position',round([0.130*LPW, 0.740*LPH, 0.276*LPW, 0.148*LPH]));
            app.xlim_field   = uieditfield(app.LimitsPanel,'text','Position',round([0.431*LPW, 0.722*LPH, 0.496*LPW, 0.213*LPH]));
            uilabel(app.LimitsPanel,'Text','Ylim','Position',round([0.130*LPW, 0.519*LPH, 0.276*LPW, 0.148*LPH]));
            app.ylim_field   = uieditfield(app.LimitsPanel,'text','Position',round([0.431*LPW, 0.500*LPH, 0.496*LPW, 0.204*LPH]));
            uilabel(app.LimitsPanel,'Text','Length','Position',round([0.073*LPW, 0.301*LPH, 0.325*LPW, 0.159*LPH]));
            app.length_field = uieditfield(app.LimitsPanel,'text','Position',round([0.431*LPW, 0.269*LPH, 0.496*LPW, 0.204*LPH]));
            app.refresh_limits_btn = uibutton(app.LimitsPanel,'Text','Refresh','Position',round([0.260*LPW, 0.028*LPH, 0.504*LPW, 0.213*LPH]),'ButtonPushedFcn',@(s,e)app.refreshLimitsBtnPushed(e));

            app.UIFigure.Visible = 'on';
        end
    end

    methods (Access = public)
        function app = Filtering()
            createComponents(app);
            registerApp(app, app.UIFigure);
            runStartupFcn(app, @startupFcn);
            if nargout == 0, clear app; end
        end

        function delete(app)
            delete(app.UIFigure);
        end
    end

    methods (Access = private)
        function startupFcn(app)
            app.initSettings();
            app.c = 0; app.etype = 2;
            disabledItems = {app.PlotTSMenu, app.Save3dplotMenu, app.SaveBothMenu, app.SaveAvgMenu, app.SaveMmMenu, ...
                             app.SaveFiltSigPlotMenu, app.SaveRidgePlotMenu, app.SavePhasePlotMenu, ...
                             app.AllFiltPlotMenu, app.SaveFourierMenu, app.SaveCsvMenu, app.SaveMatMenu, ...
                             app.SaveSessionMenu, app.mark_interval_btn, app.add_interval_btn, ...
                             app.filter_signal_btn, app.ridgecalc_btn, app.transform_btn};
            for item = disabledItems, item{1}.Enable = 'off'; end
        end
    end
end
