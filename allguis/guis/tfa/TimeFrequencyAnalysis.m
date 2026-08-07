%Version 1.02
%********************************************************************************
%*************************** Time-Frequency Analysis GUI ************************
%********************************************************************************
%---------------------------Credits---------------------------------------
% Wavelet and windowed Fourier transform: Dmytro Iatsenko
%
%----------------------------Documentation--------------------------------
% Reads a single or matrix of signals in any format readable by MATLAB.
% User can select the part of the signal they want to use, and calculate the
% wavelet transform or windowed Fourier transform of that part.
% Displays the amplitude/power surface plot and the time-averaged plot.
% Save options for graphs and data from wavelet/WFT transforms.
%
% Migrated from GUIDE to App Designer (classdef).
% Compatible with MATLAB R2023a through R2026a.

classdef TimeFrequencyAnalysis < matlab.apps.AppBase

    % ------------------------------------------------------------------ %
    %  UI component properties                                             %
    % ------------------------------------------------------------------ %
    properties (Access = public)
        UIFigure
        RootContainer   % parent for built components: UIFigure (standalone) or a uitab (embedded)
        OwnsFigure = true   % false when embedded into a shell app's uitab

        % Menus
        FileMenu
        ResetGUIMenu
        FileReadMenu
        LoadSessionMenu
        SaveFigureMenu
        ExportViewMenu
        OpenViewMenu
        SaveMenu
        MatSaveMenu
        CsvSaveMenu
        SaveWTCoeffMenu
        SaveSessionMenu

        % Panels
        TimeSeriesPanel
        WtPane

        % Logo axes
        logo
        nbmplogo

        % Plot axes
        time_series
        plot_pp
        plot3d
        plot_pow
        cum_avg

        % Controls
        signal_list
        status
        signal_length
        wavlet_transform   % "Transform All" button
        wt_single          % "Transform Single" button
        xlim_field
        ylim_field
        length_field
        intervals_field
        max_freq
        min_freq
        central_freq
        wavelet_type
        preprocess
        cutedges
        kaisera
        kaiseraLabel
        refresh_limits_btn

        % Button groups
        plot_type_bg
        power_rb
        amp_rb
        calc_type_bg
        wav_rb
        four_rb

        % Statistics controls — collapsed behind statsToggle by default
        statsToggle
        statsControls   % cell array of handles toggled together
        group1
        group2
        testtype
        alpha
        replot_btn
        avgtype
        testinput
    end

    % ------------------------------------------------------------------ %
    %  Data properties (replace handles struct)                           %
    % ------------------------------------------------------------------ %
    properties (Access = public)
        cmap
        linecol
        line2width = 2

        calc_type = 1   % 1 = WT, 2 = WFT
        plot_type = 2   % 1 = power, 2 = amplitude

        sig
        sig_cut
        sig_pp
        sampling_freq
        freqarr
        wopt
        WT
        amp_WT
        pow_WT
        amp_arr
        pow_arr
        time_axis
        time_axis_us
        peak_value
        fc
        currsig = []
        failed = false
        it = 0

        % Statistics
        p
        g1
        g2
        gr1
        gr2
        ttype
        ttypeS
        testin
        legstat
        leg1

        h_wait   % waitbar handle
    end

    % ------------------------------------------------------------------ %
    %  Helper methods                                                      %
    % ------------------------------------------------------------------ %
    methods (Access = private)

        function idx = listboxIndex(~, lb)
            % Return numeric index of selected listbox item(s).
            items = lb.Items;
            sel   = lb.Value;
            if ischar(sel), sel = {sel}; end
            idx = find(ismember(items, sel));
            if isempty(idx), idx = 1; end
        end

        function idx = dropdownIndex(~, dd)
            % Return numeric index of selected dropdown item.
            idx = find(strcmp(dd.Items, dd.Value), 1);
            if isempty(idx), idx = 1; end
        end

        function setListboxByIndex(~, lb, idx)
            % Set listbox selection by numeric index.
            if idx < 1 || idx > numel(lb.Items), return; end
            lb.Value = lb.Items{idx};
        end

        function showPlotMode(app, mode)
            % mode 1 = single-signal TF (plot3d + plot_pow)
            % mode 2 = all-signal average (cum_avg)
            switch mode
                case 1
                    app.plot3d.Visible    = 'on';
                    app.plot_pow.Visible  = 'on';
                    app.cum_avg.Visible   = 'off';
                case 2
                    app.plot3d.Visible    = 'off';
                    app.plot_pow.Visible  = 'off';
                    app.cum_avg.Visible   = 'on';
            end
        end

        function initSettings(app)
            % Equivalent to MODAsettings() for App Designer.
            load('cmap.mat', 'cmap');
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
            % Consistent branding placement across every module screen:
            % MODA's own logo anchored top-right, university logo anchored
            % bottom-right, both with a background matching the window
            % instead of a separate white box, using uiimage (not
            % uiaxes+image()) so they never stretch/warp as the window or
            % embedding tab is resized. Same absolute Position/size in every
            % module file — that's what makes the anchoring consistent
            % ("not jumping") as the user switches between screens.
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

    % ------------------------------------------------------------------ %
    %  Callbacks                                                           %
    % ------------------------------------------------------------------ %
    methods (Access = private)

        % ---- File menu -----------------------------------------------

        function fileReadMenuSelected(app, ~)
            [app, A] = MODAreadcheck(app);
            if A ~= 1, return; end

            cla(app.cum_avg,  'reset'); cla(app.plot3d, 'reset');
            cla(app.plot_pow, 'reset');
            app.showPlotMode(1);
            cla(app.time_series, 'reset'); cla(app.plot_pp, 'reset');
            app.signal_list.Items = {};

            fields = {'freqarr','sig','sig_cut','sig_pp','time_axis','time_axis_us',...
                      'pow_arr','amp_arr','pow_WT','amp_WT','WT','wopt',...
                      'sampling_freq','peak_value'};
            for f = fields
                app.(f{1}) = [];
            end

            [app, sig, E] = MODAread(app, 0);
            if isequal(sig,0) || E == 0, return; end

            list = cell(size(sig,1)+1,1);
            for i = 1:size(sig,1)
                list{i} = sprintf('Signal %d', i);
            end
            list{end} = 'Average Plot (All)';
            app.signal_list.Items = list;
            app.setListboxByIndex(app.signal_list, 1);

            fs = app.sampling_freq;
            globalfontsize = 12;
            plot(app.time_series, app.time_axis, sig(1,:), 'color', app.linecol(1,:));
            app.time_series.FontSize = globalfontsize;
            xlim(app.time_series, [0, size(sig,2)/fs]);
            xlabel(app.time_series, 'Time (s)');
            app.refreshLimitsCallback();
            cla(app.plot_pp, 'reset');
            app.preprocessCallback();
            xlabel(app.time_series, 'Time (s)');
            app.status.Value = 'Select data and continue with transform';
            app.signal_length.Value = sprintf('%s minutes', num2str(size(sig,2)/fs/60));
        end

        function loadSessionMenuSelected(app, ~)
            app = MODAload(app);
        end

        function resetGUIMenuSelected(app, ~)
            TimeFrequencyAnalysis;
        end

        % ---- Preprocessing -------------------------------------------

        function preprocessCallback(app)
            xl  = csv_to_mvar(app.xlim_field.Value);
            times = app.time_axis;
            if isempty(times), return; end
            indices = times >= xl(1) & times <= xl(2);

            cla(app.plot_pp, 'reset');
            L  = size(app.sig, 2);
            signal_selected = app.listboxIndex(app.signal_list);
            fs   = app.sampling_freq;
            fmax = str2double(app.max_freq.Value);
            fmin = str2double(app.min_freq.Value);

            sigBackup   = app.sig;
            app.sig     = app.sig(:, indices);
            app.sig_pp  = cell(size(app.sig,1), 1);

            for i = 1:size(app.sig,1)
                s = app.sig(i,:);
                X = (1:length(s))'/fs;
                XM = ones(length(X),4);
                for pn = 1:3
                    CX = X.^pn;
                    XM(:,pn+1) = (CX - mean(CX)) / std(CX);
                end
                s = s(:);
                w = warning('off','all');
                ns = s - XM*(pinv(XM)*s);
                warning(w);
                fx = fft(ns, L);
                Nq = ceil((L+1)/2);
                ff = [(0:Nq-1), -fliplr(1:L-Nq)] * fs/L;
                ff = ff(:);
                fx(abs(ff) <= max([fmin, fs/L]) | abs(ff) >= fmax) = 0;
                app.sig_pp{i,1} = ifft(fx)';
            end
            app.sig = sigBackup;

            globalfontsize = 12;
            plot(app.plot_pp, app.time_axis, app.sig(signal_selected,:), 'color', app.linecol(1,:));
            hold(app.plot_pp, 'on');
            pp_times = times(indices);
            pp_sig   = app.sig_pp{signal_selected,1};
            pp_sig   = pp_sig(1:sum(indices));
            plot(app.plot_pp, pp_times, pp_sig, 'color', app.linecol(2,:));
            legend(app.plot_pp, {'Original','Pre-Processed'}, 'FontSize', globalfontsize-2, ...
                   'Location','Best','units','normalized');
            xlim(app.plot_pp, [0, size(app.sig,2)/fs]);
            app.plot_pp.FontSize = globalfontsize;
            xlabel(app.plot_pp, 'Time (s)', 'FontSize', globalfontsize-2);
            xl2 = csv_to_mvar(app.xlim_field.Value) .* fs;
            xl2(2) = min(xl2(2), size(app.sig,2));
            xl2(1) = max(xl2(1), 1);
            xlim(app.plot_pp, xl2/fs);
            drawnow;
        end

        % ---- Signal list --------------------------------------------

        function signalListChanged(app, ~)
            signal_selected = app.listboxIndex(app.signal_list);

            if any(signal_selected == size(app.sig,1)+1)
                if app.OwnsFigure, app.SaveWTCoeffMenu.Enable = 'off'; end
            else
                if numel(signal_selected) == 1
                    % single signal selected
                else
                    app.setListboxByIndex(app.signal_list, 1);
                    drawnow;
                    app.xyplotCallback();
                end
            end

            if any(signal_selected ~= size(app.sig,1)+1) && numel(signal_selected)==1
                if app.OwnsFigure, app.SaveWTCoeffMenu.Enable = 'on'; end
                globalfontsize = 12;
                plot(app.time_series, app.time_axis, app.sig(signal_selected,:), 'color', app.linecol(1,:));
                app.time_series.FontSize = globalfontsize;
                xl = csv_to_mvar(app.xlim_field.Value);
                xlim(app.time_series, xl);
                xlabel(app.time_series, 'Time (s)');
                app.refreshLimitsCallback();
                cla(app.plot_pp, 'reset');
                app.preprocessCallback();
                xlabel(app.time_series, 'Time (s)');
                app.status.Value = 'Select data and continue with transform';
                if ~isempty(app.amp_WT)
                    app.xyplotCallback();
                end
                app.intervalsCallback();
            elseif any(signal_selected == size(app.sig,1)+1)
                app.xyplotCallback();
                app.intervalsCallback();
            end
        end

        % ---- Transform buttons -------------------------------------

        function wavletTransformButtonPushed(app, ~)
            app.currsig = [];
            app = MODATFAcalc(app.UIFigure, [], app, 1);
            if ~app.failed
                app.intervalsCallback();
                app.xyplotCallback();
            end
        end

        function wtSingleButtonPushed(app, ~)
            app.currsig = app.listboxIndex(app.signal_list);
            app = MODATFAcalc(app.UIFigure, [], app, 2);
            if ~app.failed
                app.intervalsCallback();
                app.xyplotCallback();
            end
        end

        % ---- Plotting -----------------------------------------------

        function xyplotCallback(app)
            signal_selected = app.listboxIndex(app.signal_list);
            globalfontsize  = 12;

            if any(signal_selected == size(app.sig,1)+1) && ~isempty(app.freqarr)
                % --- All-signal average mode ---
                app.showPlotMode(2);
                if app.OwnsFigure
                    app.ExportViewMenu.Enable = 'on';
                    app.OpenViewMenu.Enable   = 'on';
                end

                cla(app.cum_avg, 'reset');
                cla(app.time_series, 'reset');
                hold(app.cum_avg, 'on');

                if app.plot_type == 1
                    if size(app.sig,1) == 1
                        plot(app.cum_avg, app.freqarr, cell2mat(app.pow_arr), '-', 'Linewidth',3, 'color', app.linecol(1,:));
                        plot(app.cum_avg, app.freqarr, cell2mat(app.pow_arr), '--','Linewidth',3, 'color', app.linecol(2,:));
                    else
                        plot(app.cum_avg, app.freqarr, mean(cell2mat(app.pow_arr)), '-', 'Linewidth',3, 'color', app.linecol(1,:));
                        plot(app.cum_avg, app.freqarr, median(cell2mat(app.pow_arr)), '--','Linewidth',3, 'color', app.linecol(2,:));
                    end
                    ylabel(app.cum_avg, 'Average Power',     'FontSize', globalfontsize);
                    xlabel(app.cum_avg, 'Frequency (Hz)',    'FontSize', globalfontsize);
                else
                    if size(app.sig,1) == 1
                        plot(app.cum_avg, app.freqarr, cell2mat(app.amp_arr), '-', 'Linewidth',3, 'color', app.linecol(1,:));
                        plot(app.cum_avg, app.freqarr, cell2mat(app.amp_arr), '--','Linewidth',3, 'color', app.linecol(2,:));
                    else
                        plot(app.cum_avg, app.freqarr, mean(cell2mat(app.amp_arr)), '-', 'Linewidth',3, 'color', app.linecol(1,:));
                        plot(app.cum_avg, app.freqarr, median(cell2mat(app.amp_arr)), '--','Linewidth',3, 'color', app.linecol(2,:));
                    end
                    ylabel(app.cum_avg, 'Average Amplitude', 'FontSize', globalfontsize);
                    xlabel(app.cum_avg, 'Frequency (Hz)',    'FontSize', globalfontsize);
                end

                app.leg1 = {'Mean','Median'};
                legend(app.cum_avg, app.leg1, 'FontSize', globalfontsize);

                ind = 2; ls = 1; sty = '-';
                for i = 1:numel(signal_selected)
                    ind = ind + 1;
                    if ind > 7
                        ind = 1; ls = ls * -1;
                        sty = 'on'; if ls < 0, sty = '-.'; else sty = '-'; end
                    end
                    si = signal_selected(i);
                    if app.plot_type == 1 && si <= size(app.sig,1)
                        plot(app.cum_avg, app.freqarr, app.pow_arr{si,1}, sty, 'color', app.linecol(ind,:), 'LineWidth', app.line2width);
                        ylabel(app.cum_avg, 'Average Power', 'FontSize', globalfontsize);
                        xlabel(app.cum_avg, 'Frequency (Hz)', 'FontSize', globalfontsize);
                        app.leg1{i+2} = sprintf('Signal %d', si);
                        legend(app.cum_avg, app.leg1);
                    elseif si <= size(app.sig,1)
                        plot(app.cum_avg, app.freqarr, app.amp_arr{si,1}, sty, 'color', app.linecol(ind,:), 'LineWidth', app.line2width);
                        ylabel(app.cum_avg, 'Average Amplitude', 'FontSize', globalfontsize);
                        xlabel(app.cum_avg, 'Frequency (Hz)', 'FontSize', globalfontsize);
                        app.leg1{i+2} = sprintf('Signal %d', si);
                        legend(app.cum_avg, app.leg1);
                    end
                end

                grid(app.cum_avg, 'off'); box(app.cum_avg, 'on');
                title(app.cum_avg, 'Transform average for all signals');
                if app.calc_type == 1
                    app.cum_avg.XScale = 'log';
                else
                    app.cum_avg.XScale = 'linear';
                end
                xlim(app.cum_avg, [min(app.freqarr) max(app.freqarr)]);

            elseif ~isempty(app.freqarr)
                % --- Single-signal TF mode ---
                app.showPlotMode(1);
                if app.OwnsFigure
                    app.ExportViewMenu.Enable = 'on';
                    app.OpenViewMenu.Enable   = 'on';
                end

                cla(app.cum_avg, 'reset');
                cla(app.plot3d,  'reset');
                cla(app.plot_pow,'reset');

                if app.plot_type == 1
                    WTpow = app.pow_WT{signal_selected,1};
                    app.peak_value = max(WTpow(:)) + 0.1;
                    pcolor(app.plot3d, app.time_axis_us, app.freqarr, WTpow);
                    colorbar(app.plot3d);
                    plot(app.plot_pow, app.pow_arr{signal_selected,1}, app.freqarr, '-k', 'LineWidth', 3);
                    xlabel(app.plot_pow, 'Average Power');
                else
                    WTamp = app.amp_WT{signal_selected,1};
                    app.peak_value = max(WTamp(:)) + 0.1;
                    pcolor(app.plot3d, app.time_axis_us, app.freqarr, WTamp);
                    colorbar(app.plot3d);
                    plot(app.plot_pow, app.amp_arr{signal_selected,1}, app.freqarr, '-k', 'LineWidth', 3);
                    xlabel(app.plot_pow, 'Average Amplitude');
                end

                colormap(app.plot3d, app.cmap);
                shading(app.plot3d, 'interp');

                if app.calc_type == 1
                    app.plot3d.YScale   = 'log';
                    app.plot_pow.YScale = 'log';
                end
                ylim(app.plot3d,  [min(app.freqarr) max(app.freqarr)]);
                xlim(app.plot3d,  [app.time_axis_us(1) app.time_axis_us(end)]);
                xlabel(app.plot3d,  'Time (s)');
                ylabel(app.plot3d,  'Frequency (Hz)');
                ylabel(app.plot_pow,'Frequency (Hz)');
                ylim(app.plot_pow, [min(app.freqarr) max(app.freqarr)]);
                app.status.Value = 'Done Plotting';
            end

            grid(app.plot3d,  'on');
            grid(app.plot_pow,'off');
            grid(app.cum_avg, 'off');
            app.plot3d.FontSize   = globalfontsize;
            app.plot_pow.FontSize = globalfontsize;
            app.cum_avg.FontSize  = globalfontsize;
        end

        % ---- Interval lines -----------------------------------------

        function intervalsCallback(app)
            if isempty(app.sig), return; end
            intervals = csv_to_mvar(app.intervals_field.Value);
            intervals = sort(intervals);

            allAxes = {app.plot3d, app.plot_pow, app.cum_avg};
            for k = 1:numel(allAxes)
                ax = allAxes{k};
                ch = allchild(ax);
                for j = 1:numel(ch)
                    if strcmpi(get(ch(j),'Type'),'Line')
                        if strcmp(get(ch(j),'linestyle'),'--') && get(ch(j),'linewidth') <= 1
                            delete(ch(j));
                        end
                    end
                end
                ax.YTickMode = 'auto'; ax.XTickMode = 'auto';
            end

            signal_selected   = app.listboxIndex(app.signal_list);
            hold(app.cum_avg, 'on');

            if numel(signal_selected) > 1
                xl = app.cum_avg.YLim;
                for j = 1:numel(intervals)
                    x = xl; z = ones(1,2); y = intervals(j)*ones(1,2);
                    plot3(app.cum_avg, y, x, z, '--k');
                    xticks = app.cum_avg.XTick;
                    app.cum_avg.XTick = unique(sort([xticks intervals]));
                end
            elseif any(signal_selected == size(app.sig,1)+1)
                xl = app.cum_avg.YLim;
                for j = 1:numel(intervals)
                    x = xl; z = ones(1,2); y = intervals(j)*ones(1,2);
                    plot3(app.cum_avg, y, x, z, '--k', 'HandleVisibility','off');
                    xticks = app.cum_avg.XTick;
                    app.cum_avg.XTick = unique(sort([xticks intervals]));
                end
            else
                visAxes = allAxes(cellfun(@(a) strcmp(a.Visible,'on'), allAxes));
                for k = 1:numel(visAxes)
                    ax = visAxes{k};
                    hold(ax, 'on');
                    warning('off');
                    xl = ax.XLim;
                    for j = 1:numel(intervals)
                        x = xl; z = ones(1,2); y = intervals(j)*ones(1,2);
                        plot3(ax, x, y, z, '--k');
                    end
                    yticks = ax.YTick;
                    ax.YTick = unique(sort([yticks intervals]));
                    warning('on');
                    hold(ax, 'off');
                    try, ax.XLim = xl; catch; end
                end
            end
            app.plot_pow.YTickLabel = {};
        end

        % ---- Axis limits -------------------------------------------

        function xlimFieldValueChanged(app, ~)
            xl = csv_to_mvar(app.xlim_field.Value);
            xlim(app.time_series, xl);
            xlim(app.plot_pp, xl);
            app.length_field.Value = num2str(xl(2)-xl(1));
        end

        function ylimFieldValueChanged(app, ~)
            yl = csv_to_mvar(app.ylim_field.Value);
            ylim(app.time_series, yl);
        end

        function refreshLimitsCallback(app)
            x = app.time_series.XLim;
            xlim(app.plot_pp, x);
            t = x(2) - x(1);
            app.xlim_field.Value  = sprintf('%s, %s', num2str(x(1)), num2str(x(2)));
            y = app.time_series.YLim;
            app.ylim_field.Value  = sprintf('%s, %s', num2str(y(1)), num2str(y(2)));
            app.length_field.Value = num2str(t);
            app.preprocessCallback();
        end

        function refreshLimitsBtnPushed(app, ~)
            app.refreshLimitsCallback();
        end

        % ---- Plot type / calc type radio groups --------------------

        function plotTypeChanged(app, event)
            switch event.NewValue.Tag
                case 'power', app.plot_type = 1;
                case 'amp',   app.plot_type = 2;
            end
            if isempty(app.p)
                app.xyplotCallback();
            else
                app.replotButtonPushed();
            end
            app.intervalsCallback();
        end

        function calcTypeChanged(app, event)
            switch event.NewValue.Tag
                case 'wav'
                    app.calc_type = 1;
                    app.wavelet_type.Items = {'Lognorm','Morlet','Bump','','',''};
                    app.kaisera.Visible = 'off'; app.kaiseraLabel.Visible = 'off';
                case 'four'
                    app.calc_type = 2;
                    app.wavelet_type.Items = {'Gaussian','Hann','Blackman','Exp','Rect','Kaiser'};
            end
            drawnow;
        end

        function waveletTypeChanged(app, ~)
            if strcmp(app.wavelet_type.Value, 'Kaiser')
                app.kaisera.Visible = 'on'; app.kaiseraLabel.Visible = 'on';
            else
                app.kaisera.Visible = 'off'; app.kaiseraLabel.Visible = 'off';
            end
        end

        function statsToggleChanged(app, ~)
            vis = 'off';
            if app.statsToggle.Value, vis = 'on'; end
            for c = app.statsControls, c{1}.Visible = vis; end
        end

        % ---- Save / plot export ------------------------------------

        % ---- Export current view (replaces the old 5-item Save-figure list) ----

        function fig = buildViewFigure(app)
            % Builds a hidden figure reproducing whichever result view is
            % currently showing: the single-signal TF+average pair, or the
            % all-signal mean/median plot.
            si = app.listboxIndex(app.signal_list);
            isAverage = any(si == size(app.sig,1)+1) && ~isempty(app.freqarr);

            if ~isAverage
                fig = figure('Visible','off');
                ax1 = axes(fig,'Position',[0.07 0.2 .55 .7]);
                ax2 = axes(fig,'Position',[0.75 0.2 .2  .7]);
                if app.plot_type == 1
                    pcolor(ax1, app.time_axis_us, app.freqarr, app.pow_WT{si,1});
                    cb = colorbar(ax1); ylabel(cb,'Wavelet power');
                    plot(ax2, app.pow_arr{si,1}, app.freqarr,'-k','LineWidth',3);
                    xlabel(ax2,'Average Power');
                else
                    pcolor(ax1, app.time_axis_us, app.freqarr, app.amp_WT{si,1});
                    cb = colorbar(ax1); ylabel(cb,'Wavelet amplitude');
                    plot(ax2, app.amp_arr{si,1}, app.freqarr,'-k','LineWidth',3);
                    xlabel(ax2,'Average Amplitude');
                end
                colormap(fig, app.cmap); shading(ax1,'interp');
                if app.calc_type == 1, ax1.YScale = 'log'; ax2.YScale = 'log'; end
                fig.Position = fig.Position .* [1 1 0.6 0.5];
            else
                fig = figure('Visible','off'); ax = axes(fig);
                if app.plot_type == 1
                    plot(ax, app.freqarr, mean(cell2mat(app.pow_arr)),'-','LineWidth',3,'color',app.linecol(1,:));
                    hold(ax,'on');
                    plot(ax, app.freqarr, median(cell2mat(app.pow_arr)),'--','LineWidth',3,'color',app.linecol(2,:));
                    ylabel(ax,'Average Power');
                else
                    plot(ax, app.freqarr, mean(cell2mat(app.amp_arr)),'-','LineWidth',3,'color',app.linecol(1,:));
                    hold(ax,'on');
                    plot(ax, app.freqarr, median(cell2mat(app.amp_arr)),'--','LineWidth',3,'color',app.linecol(2,:));
                    ylabel(ax,'Average Amplitude');
                end
                if ~isempty(app.leg1)
                    legend(ax, app.leg1);
                end
                xlabel(ax,'Frequency (Hz)');
                if app.calc_type == 1, ax.XScale = 'log'; end
                ax.Position = [0.1 0.2 .85 .7];
                fig.Position = fig.Position .* [1 1 0.5 0.5];
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

        % ---- Data save -------------------------------------------

        function matSaveMenuSelected(app, ~)
            try
                [FileName,PathName] = uiputfile('*.mat','Save data as');
                if isequal(FileName,0), return; end
                save_location = fullfile(PathName, FileName);
                xl = csv_to_mvar(app.xlim_field.Value);
                L  = xl(2)*app.wopt.fs - xl(1)*app.wopt.fs;
                TFR_data = app.buildTFRstruct(xl, L);
                save(save_location, 'TFR_data');
            catch e
                errordlg(e.message,'Error'); rethrow(e);
            end
        end

        function csvSaveMenuSelected(app, ~)
            try
                [FileName,PathName] = uiputfile('*.*');
                if isequal(FileName,0), return; end
                save_location = fullfile(PathName, FileName(1:end-3));
                xl = csv_to_mvar(app.xlim_field.Value);
                L  = xl(2)*app.wopt.fs - xl(1)*app.wopt.fs;
                TFR_data = app.buildTFRstruct(xl, L);
                data = app.csvsaving(TFR_data, app.plot_type, app.calc_type);
                cell2csv([save_location '_TFdata.csv'], data, ',');
            catch e
                errordlg(e.message,'Error'); rethrow(e);
            end
        end

        function saveWTCoeffMenuSelected(app, ~)
            try
                [FileName,PathName] = uiputfile('*.mat','Save data as');
                if isequal(FileName,0), return; end
                save_location = fullfile(PathName, FileName);
                signal_selected = app.listboxIndex(app.signal_list);
                xl = csv_to_mvar(app.xlim_field.Value);
                L  = xl(2)*app.wopt.fs - xl(1)*app.wopt.fs;
                TFR_data = struct();
                if strcmp(app.wopt.Preprocess,'on')
                    TFR_data.Preprocessed_data = app.sig_pp{signal_selected,1};
                end
                if app.calc_type == 1
                    TFR_data.Analysis_type  = 'Wavelet';
                    TFR_data.Wavelet_type   = app.wopt.Wavelet;
                else
                    TFR_data.Analysis_type  = 'Windowed Fourier';
                    TFR_data.Window_type    = app.wopt.Window;
                end
                TFR_data.TFcoefficients      = app.WT;
                TFR_data.Frequency           = app.freqarr;
                TFR_data.Time                = linspace(xl(1),xl(2),L);
                TFR_data.Sampling_frequency  = app.wopt.fs;
                TFR_data.fmax                = app.wopt.fmax;
                TFR_data.fmin                = app.wopt.fmin;
                TFR_data.fr                  = str2double(app.central_freq.Value);
                TFR_data.Preprocessing       = app.wopt.Preprocess;
                TFR_data.Cut_Edges           = app.wopt.CutEdges;
                save(save_location, 'TFR_data');
            catch e
                errordlg(e.message,'Error'); rethrow(e);
            end
        end

        function saveSessionMenuSelected(app, ~)
            MODAsave(app);
        end

        % ---- Statistics replot -----------------------------------

        function replotButtonPushed(app, ~)
            if isempty(app.amp_arr) && isempty(app.pow_arr), return; end
            app.status.Value = 'Plotting...';
            if app.OwnsFigure, app.SaveWTCoeffMenu.Enable = 'off'; end
            cla(app.cum_avg, 'reset');
            cla(app.plot3d,  'reset');
            cla(app.plot_pow,'reset');
            app.cum_avg.Visible  = 'on';
            app.plot3d.Visible   = 'off';
            app.plot_pow.Visible = 'off';
            hold(app.cum_avg,'on');

            avgt     = app.dropdownIndex(app.avgtype);
            app.ttype  = app.dropdownIndex(app.testtype);
            tlist    = app.testtype.Items;
            app.ttypeS = tlist{app.ttype};
            a        = str2double(app.alpha.Value);
            app.testin = app.dropdownIndex(app.testinput);

            if app.testin == 2
                avg_wt = cell2mat(app.pow_arr);
            else
                avg_wt = cell2mat(app.amp_arr);
            end
            app.g1 = str2num(app.group1.Value); %#ok<ST2NM>
            app.g2 = str2num(app.group2.Value); %#ok<ST2NM>

            if isempty(app.g1) || isempty(app.g2)
                errordlg('Please enter signal numbers in group field','Error'); return;
            end
            if numel(app.g1)<2 || numel(app.g2)<2
                errordlg('Group size must be larger than one for statistical analysis','Error'); return;
            end
            app.gr1 = avg_wt(app.g1,:);
            app.gr2 = avg_wt(app.g2,:);
            if app.ttype == 2 && size(app.gr1,1) ~= size(app.gr2,1)
                errordlg('Groups must be the same size for a paired test','Error'); return;
            end

            app.h_wait = waitbar(0,'Calculating statistics, please wait...');
            x = ones(1,numel(app.freqarr)).*a;
            app.p = [];
            for j = 1:numel(app.freqarr)
                if app.ttype == 2
                    app.p(j) = signrank(app.gr1(:,j), app.gr2(:,j));
                else
                    app.p(j) = ranksum(app.gr1(:,j), app.gr2(:,j));
                end
            end
            col = [0.9 0.9 0.9];

            if avgt == 1
                plot(app.cum_avg, app.freqarr, median(app.gr1), 'Linewidth',2,'color',app.linecol(1,:));
                plot(app.cum_avg, app.freqarr, median(app.gr2), 'Linewidth',2,'color',app.linecol(2,:));
                val = fillsig(app.freqarr, median(app.gr1), median(app.gr2), app.p, x,'less',col);
                if val==1, app.legstat={'Group 1 median','Group 2 median',['p < ' num2str(a)]};
                else,      app.legstat={'Group 1 median','Group 2 median'}; end
                legend(app.cum_avg, app.legstat);
                if app.testin==2, ylabel(app.cum_avg,'Median power');
                else,             ylabel(app.cum_avg,'Median amplitude'); end
            else
                plot(app.cum_avg, app.freqarr, mean(app.gr1), 'Linewidth',2,'color',app.linecol(1,:));
                plot(app.cum_avg, app.freqarr, mean(app.gr2), 'Linewidth',2,'color',app.linecol(2,:));
                val = fillsig(app.freqarr, mean(app.gr1), mean(app.gr2), app.p, x,'less',col);
                if val==1, app.legstat={'Group 1 mean','Group 2 mean',['p < ' num2str(a)]};
                else,      app.legstat={'Group 1 mean','Group 2 mean'}; end
                legend(app.cum_avg, app.legstat);
                if app.testin==2, ylabel(app.cum_avg,'Mean power');
                else,             ylabel(app.cum_avg,'Mean amplitude'); end
            end
            if app.calc_type==1, app.cum_avg.XScale = 'log'; end
            app.cum_avg.XLim = [app.freqarr(1) app.freqarr(end)];
            box(app.cum_avg,'on');
            title(app.cum_avg,'Statistical comparison');
            xlabel(app.cum_avg,'Frequency (Hz)');
            app.intervalsCallback();
            delete(app.h_wait);
            app.status.Value = 'Done Plotting';
        end

        function maxFreqValueChanged(app, ~)
            app.preprocessCallback();
        end

        function minFreqValueChanged(app, ~)
            app.preprocessCallback();
        end

        % ---- Close --------------------------------------------------

        function UIFigureCloseRequest(app, ~)
            MODAclose(app.UIFigure, app);
        end
    end

    % ------------------------------------------------------------------ %
    %  Private helper: build TFR data struct for saving                   %
    % ------------------------------------------------------------------ %
    methods (Access = private)
        function TFR_data = buildTFRstruct(app, xl, L)
            TFR_data = struct();
            if strcmp(app.wopt.Preprocess,'on')
                TFR_data.Preprocessed_data = app.sig_pp;
            end
            if app.calc_type == 1
                TFR_data.Analysis_type = 'Wavelet';
                TFR_data.Wavelet_type  = app.wopt.Wavelet;
            else
                TFR_data.Analysis_type = 'Windowed Fourier';
                TFR_data.Window_type   = app.wopt.Window;
            end
            if app.plot_type == 1
                TFR_data.Power    = cell2mat(app.pow_arr)';
            else
                TFR_data.Amplitude = cell2mat(app.amp_arr)';
            end
            TFR_data.Frequency          = app.freqarr;
            TFR_data.Time               = linspace(xl(1), xl(2), L);
            TFR_data.Sampling_frequency = app.wopt.fs;
            TFR_data.fmax               = app.wopt.fmax;
            TFR_data.fmin               = app.wopt.fmin;
            TFR_data.fr                 = str2double(app.central_freq.Value);
            TFR_data.Preprocessing      = app.wopt.Preprocess;
            TFR_data.Cut_Edges          = app.wopt.CutEdges;
            if ~isempty(app.p)
                TFR_data.group1_index = app.g1;
                TFR_data.group2_index = app.g2;
                TFR_data.group1       = app.gr1;
                TFR_data.group2       = app.gr2;
                TFR_data.p_value      = app.p;
                TFR_data.testtype     = app.ttypeS;
                TFR_data.testinput    = app.testinput.Items{app.testin};
            end
            if ~isempty(app.currsig)
                TFR_data.Selected_sig = app.currsig;
            end
        end

        function data = csvsaving(app, D, hp, hc)
            L = numel(D.Frequency);
            if hp==1, N = size(D.Power,2); else, N = size(D.Amplitude,2); end
            data = cell(L+20, (N*2)+1);
            data{1,1} = 'MODA v1.0 - Time-Frequency Analysis';
            data{2,1} = datestr(now);
            data{4,1} = 'PARAMETERS';
            if hc==1, data{5,1}='Analysis type'; data{5,2}='Wavelet';
                      data{6,1}='Wavelet type';  data{6,2}=D.Wavelet_type;
            else,     data{5,1}='Analysis type'; data{5,2}='Windowed Fourier';
                      data{6,1}='Window type';   data{6,2}=D.Window_type;
            end
            data{7,1}='Sampling frequency (Hz)'; data{7,2}=D.Sampling_frequency;
            data{8,1}='Maximum frequency (Hz)';  data{8,2}=D.fmax;
            data{9,1}='Minimum frequency (Hz)';  data{9,2}=D.fmin;
            data{10,1}='Frequency resolution';   data{10,2}=D.fr;
            data{11,1}='Preprocessing';           data{11,2}=D.Preprocessing;
            data{12,1}='Cut Edges';               data{12,2}=D.Cut_Edges;
            data{13,1}='Time start (s)';          data{13,2}=min(D.Time);
            data{14,1}='Time end (s)';            data{14,2}=max(D.Time);
            dstart = 20;
            data{dstart,1} = 'Frequency';
            for l = 1:L, data{l+dstart,1} = D.Frequency(l); end
            if hp==1
                for j=1:N
                    if isempty(app.currsig), lbl=['Power ' num2str(j)];
                    else, lbl=['Power ' num2str(app.currsig)]; end
                    data{dstart,j+1} = lbl;
                    for k=1:L, data{k+dstart,j+1}=D.Power(k,j); end
                end
            else
                for j=1:N
                    if isempty(app.currsig), lbl=['Amplitude ' num2str(j)];
                    else, lbl=['Amplitude ' num2str(app.currsig)]; end
                    data{dstart,j+1} = lbl;
                    for k=1:L, data{k+dstart,j+1}=D.Amplitude(k,j); end
                end
            end
        end
    end

    % ------------------------------------------------------------------ %
    %  Component creation                                                  %
    % ------------------------------------------------------------------ %
    methods (Access = private)
        function createComponents(app, parentContainer)
            % parentContainer: optional. When omitted, this module creates and
            % owns its own standalone uifigure (unchanged legacy behavior).
            % When supplied (e.g. a uitab from a shell app), components are
            % built onto it instead, and this module does not own/manage a
            % figure-level menu bar or close/visibility lifecycle.
            if nargin < 2 || isempty(parentContainer)
                app.UIFigure = uifigure('Visible','off');
                app.UIFigure.Position = [100 100 1600 860];
                app.UIFigure.Resize   = 'off';
                app.UIFigure.Name     = 'MODA v1.01 Time-Frequency Analysis';
                app.UIFigure.CloseRequestFcn = @(src,ev) app.UIFigureCloseRequest(ev);
                app.OwnsFigure   = true;
                app.RootContainer = app.UIFigure;
            else
                app.RootContainer = parentContainer;
                app.UIFigure      = ancestor(parentContainer, 'figure');
                app.OwnsFigure    = false;
            end

            % ---- Menus (figure-level; only when this module owns the figure) ---
            if app.OwnsFigure
                app.FileMenu      = uimenu(app.UIFigure,'Text','File');
                app.ResetGUIMenu  = uimenu(app.FileMenu,'Text','Reset GUI', 'MenuSelectedFcn',@(s,e)app.resetGUIMenuSelected(e));
                app.FileReadMenu  = uimenu(app.FileMenu,'Text','Load time series','MenuSelectedFcn',@(s,e)app.fileReadMenuSelected(e));
                app.LoadSessionMenu = uimenu(app.FileMenu,'Text','Load session','MenuSelectedFcn',@(s,e)app.loadSessionMenuSelected(e));

                % Replaces the old 5-item "Save figure" list with two actions
                % that act on whichever view (single-signal or all-signal
                % average) is currently showing.
                app.SaveFigureMenu = uimenu(app.UIFigure,'Text','Save figure');
                app.ExportViewMenu = uimenu(app.SaveFigureMenu,'Text','Export current view...','Enable','off','MenuSelectedFcn',@(s,e)app.exportViewMenuSelected(e));
                app.OpenViewMenu   = uimenu(app.SaveFigureMenu,'Text','Open current view in new figure','Enable','off','MenuSelectedFcn',@(s,e)app.openViewMenuSelected(e));

                app.SaveMenu        = uimenu(app.UIFigure,'Text','Save data');
                app.MatSaveMenu     = uimenu(app.SaveMenu,'Text','Save .mat','Enable','off','MenuSelectedFcn',@(s,e)app.matSaveMenuSelected(e));
                app.CsvSaveMenu     = uimenu(app.SaveMenu,'Text','Save .csv','Enable','off','MenuSelectedFcn',@(s,e)app.csvSaveMenuSelected(e));
                app.SaveWTCoeffMenu = uimenu(app.SaveMenu,'Text','Save WT coefficients','Enable','off','MenuSelectedFcn',@(s,e)app.saveWTCoeffMenuSelected(e));
                app.SaveSessionMenu = uimenu(app.SaveMenu,'Text','Save session','Enable','off','MenuSelectedFcn',@(s,e)app.saveSessionMenuSelected(e));
            end

            % ---- Branding: only when this module owns its figure. The
            % logos sit at the edges of the results panel, so when embedded
            % in MODAApp's tab (which already shows its own top-bar banner)
            % they'd overlap plotted content instead of adding anything.
            if app.OwnsFigure
                app.anchorBrandingLogos();
            end

            % ---- Left control sidebar (consistent with the Coherence/
            % Bispectrum screens: one scrollable panel holding every
            % control, instead of 8 separate floating panels) ----
            ctrlPanel = uipanel(app.RootContainer,'Position',[0 0 330 795],'Title','','Scrollable','on');

            yl = 750;
            uilabel(ctrlPanel,'Position',[5 yl 150 20],'Text','Select Data');
            app.signal_list = uilistbox(ctrlPanel,'Position',[5 yl-100 320 100],'Items',{}, ...
                'ValueChangedFcn',@(s,e)app.signalListChanged(e));

            yl = yl - 130;
            uilabel(ctrlPanel,'Position',[5 yl 320 20],'Text','Status:');
            app.status = uieditfield(ctrlPanel,'text','Position',[5 yl-24 320 22],'Value','Please Import Signal');

            yl = yl - 55;
            app.wavlet_transform = uibutton(ctrlPanel,'push','Position',[5 yl 155 28],'Text','Transform All', ...
                'ButtonPushedFcn',@(s,e)app.wavletTransformButtonPushed(e));
            app.wt_single        = uibutton(ctrlPanel,'push','Position',[165 yl 155 28],'Text','Transform Single', ...
                'ButtonPushedFcn',@(s,e)app.wtSingleButtonPushed(e));

            yl = yl - 40;
            uilabel(ctrlPanel,'Position',[5 yl 160 20],'Text','Data Length:');
            app.signal_length = uieditfield(ctrlPanel,'text','Position',[170 yl 155 22]);

            % Frequency params
            yl = yl - 40;
            uilabel(ctrlPanel,'Position',[5 yl 120 20],'Text','Max Freq (Hz):');
            app.max_freq = uieditfield(ctrlPanel,'text','Position',[130 yl 100 22],'ValueChangedFcn',@(s,e)app.maxFreqValueChanged(e));
            yl = yl - 30;
            uilabel(ctrlPanel,'Position',[5 yl 120 20],'Text','Min Freq (Hz):');
            app.min_freq = uieditfield(ctrlPanel,'text','Position',[130 yl 100 22],'ValueChangedFcn',@(s,e)app.minFreqValueChanged(e));
            yl = yl - 30;
            uilabel(ctrlPanel,'Position',[5 yl 120 20],'Text','Resolution:');
            app.central_freq = uieditfield(ctrlPanel,'text','Position',[130 yl 100 22]);

            % Wavelet / windowed-Fourier options
            yl = yl - 40;
            uilabel(ctrlPanel,'Position',[5 yl 140 20],'Text','WT / WFT Type:');
            app.wavelet_type = uidropdown(ctrlPanel,'Position',[148 yl 155 22], ...
                'Items',{'Lognorm','Morlet','Bump','','',''},'ValueChangedFcn',@(s,e)app.waveletTypeChanged(e));
            yl = yl - 30;
            uilabel(ctrlPanel,'Position',[5 yl 140 20],'Text','Preprocess:');
            app.preprocess = uidropdown(ctrlPanel,'Position',[148 yl 155 22],'Items',{'off','on'});
            yl = yl - 30;
            uilabel(ctrlPanel,'Position',[5 yl 140 20],'Text','Cut Edges:');
            app.cutedges = uidropdown(ctrlPanel,'Position',[148 yl 155 22],'Items',{'off','on'});
            yl = yl - 30;
            % Kaiser "a" parameter — only relevant when WT/WFT Type = Kaiser
            app.kaiseraLabel = uilabel(ctrlPanel,'Position',[5 yl 140 20],'Text','Kaiser a:','Visible','off');
            app.kaisera = uieditfield(ctrlPanel,'text','Value','3','Position',[148 yl 100 22],'Visible','off');

            % Plot type / calc type — panel tall enough that the title bar
            % doesn't overlap the top radio button (a uibuttongroup title
            % reserves space at the top of its own Height, not extra space
            % outside it).
            yl = yl - 88;
            app.plot_type_bg = uibuttongroup(ctrlPanel,'Position',[5 yl 155 78],'Title','Plot Type', ...
                'SelectionChangedFcn',@(bg,ev)app.plotTypeChanged(ev));
            app.power_rb = uiradiobutton(app.plot_type_bg,'Position',[5 30 90 20],'Text','Power','Tag','power');
            app.amp_rb   = uiradiobutton(app.plot_type_bg,'Position',[5 5  90 20],'Text','Amplitude','Value',true,'Tag','amp');

            app.calc_type_bg = uibuttongroup(ctrlPanel,'Position',[165 yl 155 78],'Title','Calc Type', ...
                'SelectionChangedFcn',@(bg,ev)app.calcTypeChanged(ev));
            app.wav_rb  = uiradiobutton(app.calc_type_bg,'Position',[5 30 90 20],'Text','WT','Value',true,'Tag','wav');
            app.four_rb = uiradiobutton(app.calc_type_bg,'Position',[5 5  90 20],'Text','WFT','Tag','four');

            % Limits
            yl = yl - 80;
            uilabel(ctrlPanel,'Position',[5 yl 75 20],'Text','X Limits:');
            app.xlim_field = uieditfield(ctrlPanel,'text','Position',[85 yl 235 22],'ValueChangedFcn',@(s,e)app.xlimFieldValueChanged(e));
            yl = yl - 30;
            uilabel(ctrlPanel,'Position',[5 yl 75 20],'Text','Y Limits:');
            app.ylim_field = uieditfield(ctrlPanel,'text','Position',[85 yl 235 22],'ValueChangedFcn',@(s,e)app.ylimFieldValueChanged(e));
            yl = yl - 30;
            uilabel(ctrlPanel,'Position',[5 yl 60 20],'Text','Length:');
            app.length_field = uieditfield(ctrlPanel,'text','Position',[70 yl 100 22]);
            app.refresh_limits_btn = uibutton(ctrlPanel,'push','Position',[180 yl 100 22],'Text','Refresh', ...
                'ButtonPushedFcn',@(s,e)app.refreshLimitsBtnPushed(e));

            % Intervals
            yl = yl - 35;
            uilabel(ctrlPanel,'Position',[5 yl 90 20],'Text','Intervals (Hz):');
            app.intervals_field = uieditfield(ctrlPanel,'text','Position',[100 yl 220 22]);

            % Statistics (group comparison) — a distinct, less-frequently-used
            % feature, so it's collapsed behind a toggle like Kaiser-a above.
            yl = yl - 40;
            app.statsToggle = uicheckbox(ctrlPanel,'Position',[5 yl 250 22], ...
                'Text','Show group statistics','ValueChangedFcn',@(s,e)app.statsToggleChanged(e));

            yl = yl - 30;
            l1 = uilabel(ctrlPanel,'Position',[5 yl 60 20],'Text','Group 1','Visible','off');
            app.group1 = uieditfield(ctrlPanel,'text','Position',[70 yl 100 22],'Visible','off');
            l2 = uilabel(ctrlPanel,'Position',[180 yl 60 20],'Text','Group 2','Visible','off');
            app.group2 = uieditfield(ctrlPanel,'text','Position',[240 yl 85 22],'Visible','off');
            yl = yl - 30;
            l3 = uilabel(ctrlPanel,'Position',[5 yl 40 20],'Text','Test','Visible','off');
            app.testtype = uidropdown(ctrlPanel,'Items',{'Unpaired','Paired'},'Position',[50 yl 120 22],'Visible','off');
            l4 = uilabel(ctrlPanel,'Position',[180 yl 45 20],'Text','alpha','Visible','off');
            app.alpha = uieditfield(ctrlPanel,'text','Value','0.05','Position',[225 yl 60 22],'Visible','off');
            yl = yl - 30;
            l5 = uilabel(ctrlPanel,'Position',[5 yl 40 20],'Text','Avg','Visible','off');
            app.avgtype = uidropdown(ctrlPanel,'Items',{'Median','Mean'},'Position',[50 yl 120 22],'Visible','off');
            yl = yl - 30;
            l6 = uilabel(ctrlPanel,'Position',[5 yl 80 20],'Text','Test Input','Visible','off');
            app.testinput = uidropdown(ctrlPanel,'Items',{'Amplitude','Power'},'Position',[90 yl 120 22],'Visible','off');
            yl = yl - 34;
            app.replot_btn = uibutton(ctrlPanel,'push','Position',[5 yl 150 26],'Text','Calculate','Visible','off', ...
                'ButtonPushedFcn',@(s,e)app.replotButtonPushed(e));
            app.statsControls = {l1,l2,l3,l4,l5,l6,app.group1,app.group2,app.testtype,app.alpha,app.avgtype,app.testinput,app.replot_btn};

            % ---- Time series panel (top right) — main signal preview plus a
            % small pre/post-processing comparison plot alongside it ----
            app.TimeSeriesPanel = uipanel(app.RootContainer,'Position',[330 500 1270 355],'Title','Time Series');
            app.time_series = uiaxes(app.TimeSeriesPanel,'Position',[5 5 900 345]);
            app.plot_pp     = uiaxes(app.TimeSeriesPanel,'Position',[915 5 350 345]);

            % ---- WT pane (overlapping axes, bottom right) ----
            app.WtPane = uipanel(app.RootContainer,'Position',[330 0 1270 500],'Title','Time-Frequency Analysis');
            PW = 1270; PH = 500;
            app.plot_pow = uiaxes(app.WtPane,'Position', round([0.8006*PW, 0.1232*PH, 0.1772*PW, 0.8485*PH]));
            app.plot3d   = uiaxes(app.WtPane,'Position', round([0.0696*PW, 0.1232*PH, 0.6551*PW, 0.8485*PH]));
            app.cum_avg  = uiaxes(app.WtPane,'Position', round([0.0527*PW, 0.1172*PH, 0.9346*PW, 0.8404*PH]));
            app.cum_avg.Visible  = 'off';

            % Final visibility (only this module's own standalone figure)
            if app.OwnsFigure
                app.UIFigure.Visible = 'on';
            end
        end
    end

    % ------------------------------------------------------------------ %
    %  App lifecycle                                                        %
    % ------------------------------------------------------------------ %
    methods (Access = public)

        function app = TimeFrequencyAnalysis(parentContainer)
            % parentContainer: optional. Omit to launch as a standalone
            % window (unchanged legacy behavior); pass a uitab (or other
            % container) to build this module's UI onto it instead.
            if nargin < 1
                parentContainer = [];
            end
            createComponents(app, parentContainer);
            % NOTE: when embedded (parentContainer supplied), registering
            % against a figure shared by other embedded modules is revisited
            % in the shell-embedding phase; harmless no-op risk today since
            % no caller yet passes parentContainer.
            registerApp(app, app.UIFigure);
            runStartupFcn(app, @startupFcn);
            if nargout == 0
                clear app;
            end
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
            % Disable controls that need data first (menus only exist when
            % this module owns its figure — see createComponents)
            if app.OwnsFigure
                app.MatSaveMenu.Enable    = 'off';
                app.CsvSaveMenu.Enable    = 'off';
                app.SaveWTCoeffMenu.Enable = 'off';
                app.SaveSessionMenu.Enable = 'off';
            end
        end
    end
end
