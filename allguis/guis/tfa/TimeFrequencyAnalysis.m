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

        % Menus
        FileMenu
        ResetGUIMenu
        FileReadMenu
        LoadSessionMenu
        SaveFigureMenu
        PlotTSMenu
        Save3dplotMenu
        SaveBothMenu
        SaveAvgMenu
        SaveMmMenu
        SaveMenu
        MatSaveMenu
        CsvSaveMenu
        SaveWTCoeffMenu
        SaveSessionMenu

        % Panels
        TimeSeriesPanel
        WtPane
        FreqParamsPanel
        AdvancedPanel
        IntervalsPanel
        LimitsPanel
        DataLenPanel
        StatsPanel
        CalcTypePanel   % container for calc_type button group
        PlotTypePanel   % container for plot_type button group

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
        refresh_limits_btn

        % Button groups
        plot_type_bg
        power_rb
        amp_rb
        calc_type_bg
        wav_rb
        four_rb

        % Statistics panel controls
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

            ss = get(groot,'Screensize');
            sw = ss(3); sh = ss(4);
            if sw < 1600 || sh < 860
                app.UIFigure.Position = [0 0 sw sh];
            else
                app.UIFigure.Position = [round((sw-1600)/2) round((sh-860)/2) 1600 860];
            end

            try
                img = imread('physicslogo.png');
                image(app.logo, img);
                axis(app.logo, 'off'); axis(app.logo, 'image');
            catch; end
            try
                img = imread('MODAbanner5.png');
                image(app.nbmplogo, img);
                axis(app.nbmplogo, 'off'); axis(app.nbmplogo, 'image');
            catch; end
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
                app.SaveWTCoeffMenu.Enable = 'off';
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
                app.SaveWTCoeffMenu.Enable = 'on';
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
                app.Save3dplotMenu.Enable  = 'off';
                app.SaveBothMenu.Enable    = 'off';
                app.SaveAvgMenu.Enable     = 'off';
                app.SaveMmMenu.Enable      = 'on';

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
                app.Save3dplotMenu.Enable = 'on';
                app.SaveBothMenu.Enable   = 'on';
                app.SaveAvgMenu.Enable    = 'on';
                app.SaveMmMenu.Enable     = 'off';

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
                    app.kaisera.Enable = 'off';
                case 'four'
                    app.calc_type = 2;
                    app.wavelet_type.Items = {'Gaussian','Hann','Blackman','Exp','Rect','Kaiser'};
            end
            drawnow;
        end

        function waveletTypeChanged(app, ~)
            if strcmp(app.wavelet_type.Value, 'Kaiser')
                app.kaisera.Enable = 'on';
            else
                app.kaisera.Enable = 'off';
            end
        end

        % ---- Save / plot export ------------------------------------

        function plotTSMenuSelected(app, ~)
            Fig = figure;
            ax  = copyobj(app.time_series.InnerPosition, Fig);
            ax2 = axes(Fig);
            globalfontsize = 12;
            plot(ax2, app.time_axis, app.sig(app.listboxIndex(app.signal_list),:));
            set(ax2,'Units','normalized','Position',[0.1 0.25 .85 .6],'FontSize',globalfontsize);
            Fig.Position = Fig.Position .* [1 1 0.5 0.3];
            xlabel(ax2, 'Time (s)');
        end

        function save3dplotMenuSelected(app, ~)
            si = app.listboxIndex(app.signal_list);
            Fig = figure; ax = axes(Fig);
            if app.plot_type == 1
                pcolor(ax, app.time_axis_us, app.freqarr, app.pow_WT{si,1});
                cb = colorbar(ax); ylabel(cb,'Wavelet power');
            else
                pcolor(ax, app.time_axis_us, app.freqarr, app.amp_WT{si,1});
                cb = colorbar(ax); ylabel(cb,'Wavelet amplitude');
            end
            colormap(ax, app.cmap); shading(ax,'interp');
            if app.calc_type == 1, ax.YScale = 'log'; end
            ax.Position = [0.1 0.2 .85 .7];
            Fig.Position = Fig.Position .* [1 1 0.5 0.5];
            xlabel(ax,'Time (s)'); ylabel(ax,'Frequency (Hz)');
        end

        function saveAvgMenuSelected(app, ~)
            si = app.listboxIndex(app.signal_list);
            Fig = figure; ax = axes(Fig);
            if app.plot_type == 1
                plot(ax, app.pow_arr{si,1}, app.freqarr, '-k','LineWidth',3);
                xlabel(ax,'Average Power');
            else
                plot(ax, app.amp_arr{si,1}, app.freqarr, '-k','LineWidth',3);
                xlabel(ax,'Average Amplitude');
            end
            if app.calc_type == 1, ax.YScale = 'log'; end
            ylabel(ax,'Frequency (Hz)');
            ax.Position = [0.1 0.2 .85 .7];
            Fig.Position = Fig.Position .* [1 1 0.5 0.5];
        end

        function saveBothMenuSelected(app, ~)
            si = app.listboxIndex(app.signal_list);
            Fig = figure;
            ax1 = axes(Fig,'Position',[0.07 0.2 .55 .7]);
            ax2 = axes(Fig,'Position',[0.75 0.2 .2  .7]);
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
            colormap(Fig, app.cmap); shading(ax1,'interp');
            if app.calc_type == 1, ax1.YScale = 'log'; ax2.YScale = 'log'; end
            Fig.Position = Fig.Position .* [1 1 0.6 0.5];
        end

        function saveMmMenuSelected(app, ~)
            Fig = figure; ax = axes(Fig);
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
            Fig.Position = Fig.Position .* [1 1 0.5 0.5];
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
            app.SaveWTCoeffMenu.Enable = 'off';
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
            app.Save3dplotMenu.Enable = 'off';
            app.SaveBothMenu.Enable   = 'off';
            app.SaveAvgMenu.Enable    = 'off';
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
        function createComponents(app)
            % Figure
            app.UIFigure = uifigure('Visible','off');
            app.UIFigure.Position = [100 100 1600 860];
            app.UIFigure.Name     = 'MODA v1.01 Time-Frequency Analysis';
            app.UIFigure.CloseRequestFcn = @(src,ev) app.UIFigureCloseRequest(ev);

            % ---- Menus ---
            app.FileMenu      = uimenu(app.UIFigure,'Text','File');
            app.ResetGUIMenu  = uimenu(app.FileMenu,'Text','Reset GUI', 'MenuSelectedFcn',@(s,e)app.resetGUIMenuSelected(e));
            app.FileReadMenu  = uimenu(app.FileMenu,'Text','Load time series','MenuSelectedFcn',@(s,e)app.fileReadMenuSelected(e));
            app.LoadSessionMenu = uimenu(app.FileMenu,'Text','Load session','MenuSelectedFcn',@(s,e)app.loadSessionMenuSelected(e));

            app.SaveFigureMenu = uimenu(app.UIFigure,'Text','Save figure');
            app.PlotTSMenu     = uimenu(app.SaveFigureMenu,'Text','Plot time series','Enable','off','MenuSelectedFcn',@(s,e)app.plotTSMenuSelected(e));
            app.Save3dplotMenu = uimenu(app.SaveFigureMenu,'Text','Save TF plot','Enable','off','MenuSelectedFcn',@(s,e)app.save3dplotMenuSelected(e));
            app.SaveBothMenu   = uimenu(app.SaveFigureMenu,'Text','Save TF + average','Enable','off','MenuSelectedFcn',@(s,e)app.saveBothMenuSelected(e));
            app.SaveAvgMenu    = uimenu(app.SaveFigureMenu,'Text','Save average plot','Enable','off','MenuSelectedFcn',@(s,e)app.saveAvgMenuSelected(e));
            app.SaveMmMenu     = uimenu(app.SaveFigureMenu,'Text','Save mean/median plot','Enable','off','MenuSelectedFcn',@(s,e)app.saveMmMenuSelected(e));

            app.SaveMenu        = uimenu(app.UIFigure,'Text','Save data');
            app.MatSaveMenu     = uimenu(app.SaveMenu,'Text','Save .mat','Enable','off','MenuSelectedFcn',@(s,e)app.matSaveMenuSelected(e));
            app.CsvSaveMenu     = uimenu(app.SaveMenu,'Text','Save .csv','Enable','off','MenuSelectedFcn',@(s,e)app.csvSaveMenuSelected(e));
            app.SaveWTCoeffMenu = uimenu(app.SaveMenu,'Text','Save WT coefficients','Enable','off','MenuSelectedFcn',@(s,e)app.saveWTCoeffMenuSelected(e));
            app.SaveSessionMenu = uimenu(app.SaveMenu,'Text','Save session','Enable','off','MenuSelectedFcn',@(s,e)app.saveSessionMenuSelected(e));

            W = 1600; H = 860;

            % ---- Logo axes (normalised positions from .fig) ---
            app.logo    = uiaxes(app.UIFigure,'Position', round([0.0142*W, 0.9243*H, 0.1799*W, 0.0623*H]));
            app.nbmplogo = uiaxes(app.UIFigure,'Position', round([0.194*W,  0.9231*H, 0.5366*W, 0.0635*H]));
            app.logo.Toolbar.Visible    = 'off';
            app.nbmplogo.Toolbar.Visible = 'off';

            % ---- Time series panel ---
            app.TimeSeriesPanel = uipanel(app.UIFigure,'Position', round([0.0149*W, 0.6911*H, 0.5104*W, 0.2063*H]),'BorderType','none');
            app.time_series = uiaxes(app.TimeSeriesPanel,'Position', round([0.0993*0.5104*W, 0.2946*0.2063*H, 0.8754*0.5104*W, 0.5504*0.2063*H]));

            % ---- WT pane (overlapping axes) ---
            app.WtPane = uipanel(app.UIFigure,'Position', round([0.0134*W, 0.0659*H, 0.7097*W, 0.6178*H]),'BorderType','none');
            PW = round(0.7097*W); PH = round(0.6178*H);
            app.plot_pow = uiaxes(app.WtPane,'Position', round([0.8006*PW, 0.1232*PH, 0.1772*PW, 0.8485*PH]));
            app.plot3d   = uiaxes(app.WtPane,'Position', round([0.0696*PW, 0.1232*PH, 0.6551*PW, 0.8485*PH]));
            app.cum_avg  = uiaxes(app.WtPane,'Position', round([0.0527*PW, 0.1172*PH, 0.9346*PW, 0.8404*PH]));
            app.cum_avg.Visible  = 'off';

            % ---- Frequency params panel ---
            app.FreqParamsPanel = uipanel(app.UIFigure,'Position', round([0.8388*W, 0.6935*H, 0.153*W, 0.188*H]),'BorderType','none');
            FPW = round(0.153*W); FPH = round(0.188*H);
            uilabel(app.FreqParamsPanel,'Text','Max Freq (Hz)','Position',round([0.07*FPW, 0.72*FPH, 0.5*FPW, 0.14*FPH]));
            app.max_freq = uieditfield(app.FreqParamsPanel,'text','Position',round([0.61*FPW, 0.7*FPH, 0.34*FPW, 0.24*FPH]),'ValueChangedFcn',@(s,e)app.maxFreqValueChanged(e));
            uilabel(app.FreqParamsPanel,'Text','Min Freq (Hz)','Position',round([0.07*FPW, 0.38*FPH, 0.5*FPW, 0.14*FPH]));
            app.min_freq = uieditfield(app.FreqParamsPanel,'text','Position',round([0.61*FPW, 0.37*FPH, 0.34*FPW, 0.24*FPH]),'ValueChangedFcn',@(s,e)app.minFreqValueChanged(e));
            uilabel(app.FreqParamsPanel,'Text','Resolution','Position',round([0.05*FPW, 0.05*FPH, 0.53*FPW, 0.16*FPH]));
            app.central_freq = uieditfield(app.FreqParamsPanel,'text','Position',round([0.61*FPW, 0.05*FPH, 0.34*FPW, 0.24*FPH]));

            % ---- Transform buttons ---
            app.wavlet_transform = uibutton(app.UIFigure,'Text','Transform All','Position',round([0.5418*W, 0.0085*H, 0.0873*W, 0.0464*H]),'ButtonPushedFcn',@(s,e)app.wavletTransformButtonPushed(e));
            app.wt_single        = uibutton(app.UIFigure,'Text','Transform Single','Position',round([0.6336*W, 0.0085*H, 0.0858*W, 0.0464*H]),'ButtonPushedFcn',@(s,e)app.wtSingleButtonPushed(e));

            % ---- Advanced options panel ---
            app.AdvancedPanel = uipanel(app.UIFigure,'Position', round([0.7366*W, 0.0085*H, 0.256*W, 0.4286*H]),'BorderType','none');
            APW = round(0.256*W); APH = round(0.4286*H);
            app.plot_pp = uiaxes(app.AdvancedPanel,'Position', round([0.0914*APW, 0.1687*APH, 0.8319*APW, 0.3912*APH]));
            uilabel(app.AdvancedPanel,'Text','WT / WFT Type','Position',round([0.029*APW, 0.859*APH, 0.287*APW, 0.066*APH]));
            app.wavelet_type = uidropdown(app.AdvancedPanel,'Items',{'Lognorm','Morlet','Bump','','',''},'Position',round([0.331*APW, 0.859*APH, 0.457*APW, 0.081*APH]),'ValueChangedFcn',@(s,e)app.waveletTypeChanged(e));
            uilabel(app.AdvancedPanel,'Text','Pre-process','Position',round([0.014*APW, 0.774*APH, 0.313*APW, 0.054*APH]));
            app.preprocess   = uidropdown(app.AdvancedPanel,'Items',{'off','on'},'Position',round([0.332*APW, 0.760*APH, 0.458*APW, 0.082*APH]));
            uilabel(app.AdvancedPanel,'Text','Cut Edges','Position',round([0.029*APW, 0.677*APH, 0.215*APW, 0.050*APH]));
            app.cutedges     = uidropdown(app.AdvancedPanel,'Items',{'off','on'},'Position',round([0.332*APW, 0.661*APH, 0.456*APW, 0.081*APH]));
            uilabel(app.AdvancedPanel,'Text','Comparison before and after preprocessing','Position',round([0.016*APW, 0.524*APH, 0.968*APW, 0.056*APH]));
            uilabel(app.AdvancedPanel,'Text','a','Position',round([0.838*APW, 0.885*APH, 0.041*APW, 0.050*APH]));
            app.kaisera = uieditfield(app.AdvancedPanel,'text','Value','3','Enable','off','Position',round([0.887*APW, 0.878*APH, 0.058*APW, 0.069*APH]));

            % ---- Intervals + status panel ---
            app.IntervalsPanel = uipanel(app.UIFigure,'Position', round([0.0149*W, 0.0012*H, 0.5187*W, 0.0635*H]),'BorderType','none');
            IPW = round(0.5187*W); IPH = round(0.0635*H);
            uilabel(app.IntervalsPanel,'Text','Intervals (Hz):','Position',round([0.007*IPW, 0.39*IPH, 0.127*IPW, 0.31*IPH]));
            app.intervals_field = uieditfield(app.IntervalsPanel,'text','Position',round([0.152*IPW, 0.286*IPH, 0.271*IPW, 0.518*IPH]));
            uilabel(app.IntervalsPanel,'Text','Status:','Position',round([0.445*IPW, 0.37*IPH, 0.091*IPW, 0.35*IPH]));
            app.status = uieditfield(app.IntervalsPanel,'text','Value','Please Import Signal','Position',round([0.54*IPW, 0.286*IPH, 0.444*IPW, 0.518*IPH]));

            % ---- Signal list ---
            app.signal_list = uilistbox(app.UIFigure,'Items',{},'Position', round([0.6396*W, 0.6984*H, 0.0813*W, 0.1673*H]),'ValueChangedFcn',@(s,e)app.signalListChanged(e));
            uilabel(app.UIFigure,'Text','Select Data','Position', round([0.6403*W, 0.8694*H, 0.0806*W, 0.0208*H]));

            % ---- Limits panel ---
            app.LimitsPanel = uipanel(app.UIFigure,'Position', round([0.5328*W, 0.6935*H, 0.0993*W, 0.2051*H]),'BorderType','none');
            LPW = round(0.0993*W); LPH = round(0.2051*H);
            uilabel(app.LimitsPanel,'Text','Xlim','Position',round([0.048*LPW, 0.831*LPH, 0.280*LPW, 0.095*LPH]));
            app.xlim_field   = uieditfield(app.LimitsPanel,'text','Position',round([0.436*LPW, 0.803*LPH, 0.503*LPW, 0.146*LPH]),'ValueChangedFcn',@(s,e)app.xlimFieldValueChanged(e));
            uilabel(app.LimitsPanel,'Text','Ylim','Position',round([0.048*LPW, 0.613*LPH, 0.279*LPW, 0.097*LPH]));
            app.ylim_field   = uieditfield(app.LimitsPanel,'text','Position',round([0.443*LPW, 0.590*LPH, 0.483*LPW, 0.146*LPH]),'ValueChangedFcn',@(s,e)app.ylimFieldValueChanged(e));
            uilabel(app.LimitsPanel,'Text','Length','Position',round([0.047*LPW, 0.337*LPH, 0.349*LPW, 0.157*LPH]));
            app.length_field = uieditfield(app.LimitsPanel,'text','Position',round([0.443*LPW, 0.376*LPH, 0.483*LPW, 0.146*LPH]));
            app.refresh_limits_btn = uibutton(app.LimitsPanel,'Text','Refresh','Position',round([0.223*LPW, 0.113*LPH, 0.587*LPW, 0.188*LPH]),'ButtonPushedFcn',@(s,e)app.refreshLimitsBtnPushed(e));

            % ---- Data length panel ---
            app.DataLenPanel = uipanel(app.UIFigure,'Position', round([0.7388*W, 0.9194*H, 0.2604*W, 0.0672*H]),'BorderType','none');
            DPW = round(0.2604*W); DPH = round(0.0672*H);
            uilabel(app.DataLenPanel,'Text','Data Length','Position',round([0.029*DPW, 0.400*DPH, 0.240*DPW, 0.356*DPH]));
            app.signal_length = uieditfield(app.DataLenPanel,'text','Position',round([0.278*DPW, 0.156*DPH, 0.552*DPW, 0.711*DPH]));

            % ---- Plot type button group ---
            app.PlotTypePanel = uipanel(app.UIFigure,'Position', round([0.7358*W, 0.7912*H, 0.097*W, 0.0891*H]),'BorderType','none');
            app.plot_type_bg  = uibuttongroup(app.PlotTypePanel,'Position',[0 0 round(0.097*W) round(0.0891*H)],'SelectionChangedFcn',@(bg,ev)app.plotTypeChanged(ev));
            PPH = round(0.0891*H);
            app.amp_rb   = uiradiobutton(app.plot_type_bg,'Text','Amplitude','Tag','amp',  'Position',[round(0.084*round(0.097*W)) round(0.594*PPH) round(0.776*round(0.097*W)) round(0.333*PPH)]);
            app.power_rb = uiradiobutton(app.plot_type_bg,'Text','Power',    'Tag','power','Position',[round(0.084*round(0.097*W)) round(0.145*PPH) round(0.645*round(0.097*W)) round(0.377*PPH)]);

            % ---- Calc type button group ---
            app.CalcTypePanel = uipanel(app.UIFigure,'Position', round([0.7366*W, 0.6935*H, 0.097*W, 0.0916*H]),'BorderType','none');
            app.calc_type_bg  = uibuttongroup(app.CalcTypePanel,'Position',[0 0 round(0.097*W) round(0.0916*H)],'SelectionChangedFcn',@(bg,ev)app.calcTypeChanged(ev));
            CPH = round(0.0916*H);
            app.wav_rb  = uiradiobutton(app.calc_type_bg,'Text','WT', 'Tag','wav', 'Position',[round(0.056*round(0.097*W)) round(0.592*CPH) round(0.778*round(0.097*W)) round(0.329*CPH)]);
            app.four_rb = uiradiobutton(app.calc_type_bg,'Text','WFT','Tag','four','Position',[round(0.056*round(0.097*W)) round(0.145*CPH) round(0.921*round(0.097*W)) round(0.382*CPH)]);

            % ---- Statistics panel ---
            app.StatsPanel = uipanel(app.UIFigure,'Position', round([0.7373*W, 0.4481*H, 0.2545*W, 0.2332*H]),'BorderType','none');
            SPW = round(0.2545*W); SPH = round(0.2332*H);
            uilabel(app.StatsPanel,'Text','Group 1', 'Position',round([0.009*SPW, 0.637*SPH, 0.181*SPW, 0.266*SPH]));
            app.group1 = uieditfield(app.StatsPanel,'text','Position',round([0.199*SPW, 0.758*SPH, 0.27*SPW, 0.195*SPH]));
            uilabel(app.StatsPanel,'Text','Group 2', 'Position',round([0.015*SPW, 0.443*SPH, 0.166*SPW, 0.208*SPH]));
            app.group2 = uieditfield(app.StatsPanel,'text','Position',round([0.199*SPW, 0.503*SPH, 0.27*SPW, 0.195*SPH]));
            uilabel(app.StatsPanel,'Text','Test',    'Position',round([0.496*SPW, 0.497*SPH, 0.116*SPW, 0.154*SPH]));
            app.testtype = uidropdown(app.StatsPanel,'Items',{'Unpaired','Paired'},'Position',round([0.617*SPW, 0.387*SPH, 0.27*SPW, 0.266*SPH]));
            uilabel(app.StatsPanel,'Text','alpha',   'Position',round([0.033*SPW, 0.161*SPH, 0.122*SPW, 0.270*SPH]));
            app.alpha    = uieditfield(app.StatsPanel,'text','Value','0.05','Position',round([0.294*SPW, 0.255*SPH, 0.092*SPW, 0.215*SPH]));
            app.replot_btn = uibutton(app.StatsPanel,'Text','Calculate','Position',round([0.629*SPW, 0.114*SPH, 0.217*SPW, 0.282*SPH]),'ButtonPushedFcn',@(s,e)app.replotButtonPushed(e));
            uilabel(app.StatsPanel,'Text','Avg',     'Position',round([0.496*SPW, 0.674*SPH, 0.122*SPW, 0.236*SPH]));
            app.avgtype  = uidropdown(app.StatsPanel,'Items',{'Median','Mean'},'Position',round([0.614*SPW, 0.689*SPH, 0.27*SPW, 0.232*SPH]));
            uilabel(app.StatsPanel,'Text','Test Input','Position',round([-0.009*SPW, 0.074*SPH, 0.223*SPW, 0.134*SPH]));
            app.testinput = uidropdown(app.StatsPanel,'Items',{'Amplitude','Power'},'Position',round([0.184*SPW, 0.034*SPH, 0.362*SPW, 0.181*SPH]));

            % Final visibility
            app.UIFigure.Visible = 'on';
        end
    end

    % ------------------------------------------------------------------ %
    %  App lifecycle                                                        %
    % ------------------------------------------------------------------ %
    methods (Access = public)

        function app = TimeFrequencyAnalysis()
            createComponents(app);
            registerApp(app, app.UIFigure);
            runStartupFcn(app, @startupFcn);
            if nargout == 0
                clear app;
            end
        end

        function delete(app)
            delete(app.UIFigure);
        end
    end

    methods (Access = private)
        function startupFcn(app)
            app.initSettings();
            % Disable controls that need data first
            app.PlotTSMenu.Enable     = 'off';
            app.Save3dplotMenu.Enable = 'off';
            app.SaveBothMenu.Enable   = 'off';
            app.SaveAvgMenu.Enable    = 'off';
            app.SaveMmMenu.Enable     = 'off';
            app.kaisera.Enable        = 'off';
            app.MatSaveMenu.Enable    = 'off';
            app.CsvSaveMenu.Enable    = 'off';
            app.SaveWTCoeffMenu.Enable = 'off';
            app.SaveSessionMenu.Enable = 'off';
        end
    end
end
