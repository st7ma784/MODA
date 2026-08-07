%Version 1.02
%********************************************************************************
%***************************** Bayesian Inference GUI ***************************
%********************************************************************************
%
% Dynamical Bayesian inference for detecting coupling between oscillators.
% Migrated from GUIDE to App Designer (classdef).
% Compatible with MATLAB R2023a through R2026a.

classdef Bayesian < matlab.apps.AppBase

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
        LoadFiltMenu
        LoadFilt2Menu
        LoadSessionMenu
        PlotMenu
        ExportViewMenu
        OpenViewMenu
        SaveMenu
        SaveCsvMenu
        SaveMatMenu
        CfVidMenu
        SaveSessionMenu

        % Logos
        logo
        nbmplogo

        % Panels
        TimePairPanel
        PlotsPane

        % Axes
        time_series_1
        time_series_2
        phi1_axes
        phi2_axes
        coupling_strength_axis
        CF1
        CF2

        % Controls
        signal_list
        interval_list_1
        interval_list_2
        status
        calculate_btn
        add_interval_btn
        delete_set_btn
        xlim_field
        length_field
        refresh_limits_btn
        window_size
        overlap
        order
        surrnum
        prop_const
        alphasig
        freq_1
        freq_2
        display_type
        time_slider
        scaleon
        curr_time
    end

    % ------------------------------------------------------------------ %
    %  Data properties                                                     %
    % ------------------------------------------------------------------ %
    properties (Access = public)
        cmap
        linecol
        line2width = 2

        sig
        sig_cut
        time_axis
        time_axis_cut
        sampling_freq

        % Parameter sets
        c = 0
        int1
        int2
        winds
        pr
        ovr
        forder
        ns
        confidence_level
        pinput   % loaded phases input

        % Calculation results
        p1
        p2
        bands
        tm
        cc
        e_noise
        cpl1
        cpl2
        cf1
        cf2
        mcf1
        mcf2
        surr_cpl1
        surr_cpl2
        cfmax
        cfmin
        tp1
        tp2
        leg1

        h_wait
        it = 0
    end

    % ------------------------------------------------------------------ %
    %  Helpers                                                             %
    % ------------------------------------------------------------------ %
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

        function buildBayesData(app, Bayes_data_out)
            % populate a struct for saving
        end
    end

    % ------------------------------------------------------------------ %
    %  Callbacks                                                           %
    % ------------------------------------------------------------------ %
    methods (Access = private)

        function fileReadMenuSelected(app, ~)
            app = MODAread(app, 1, "even");
            if isfield(app,'sampling_freq') || ~isempty(app.sampling_freq)
                app.refreshLimitsCallback();
            end
        end

        function loadFiltMenuSelected(app, ~)
            app = MODAbayes_loadfilt(app.UIFigure, [], app, 1);
            if ~isempty(app.sampling_freq)
                app.refreshLimitsCallback();
            end
        end

        function loadFilt2MenuSelected(app, ~)
            app = MODAbayes_loadfilt(app.UIFigure, [], app, 2);
            if ~isempty(app.sampling_freq)
                app.refreshLimitsCallback();
            end
        end

        function loadSessionMenuSelected(app, ~)
            app = MODAload(app);
        end

        function resetGUIMenuSelected(app, ~)
            Bayesian;
        end

        function refreshLimitsCallback(app)
            x  = app.time_series_2.XLim;
            t  = x(2) - x(1);
            xl = x .* app.sampling_freq;
            xl(2) = min(xl(2), size(app.sig,2));
            xl(1) = max(xl(1), 1);
            app.sig_cut       = app.sig(:, xl(1):xl(2));
            app.time_axis_cut = app.time_axis(:, xl(1):xl(2));
            app.xlim_field.Value   = sprintf('%s, %s', num2str(x(1)), num2str(x(2)));
            app.length_field.Value = num2str(t);
            app.signalListChanged([]);
        end

        function refreshLimitsBtnPushed(app, ~)
            app.refreshLimitsCallback();
        end

        function signalListChanged(app, ~)
            if isempty(app.sig), return; end
            sig_select = app.listboxIndex(app.signal_list);
            plot(app.time_series_1, app.time_axis_cut, app.sig_cut(sig_select,:),   'color', app.linecol(1,:));
            plot(app.time_series_2, app.time_axis_cut, app.sig_cut(sig_select + size(app.sig,1)/2, :), 'color', app.linecol(1,:));
            xl = csv_to_mvar(app.xlim_field.Value);
            xlim(app.time_series_1, xl); xlim(app.time_series_2, xl);
            xlabel(app.time_series_2, 'Time (s)');
            ylabel(app.time_series_1, 'Sig 1'); ylabel(app.time_series_2, 'Sig 2');
            app.time_series_1.XTickLabel = {};
            app.status.Value = 'Plot complete';
            if ~isempty(app.p1)
                app.intervalList1Changed([]);
            else
                drawnow;
            end
        end

        function addIntervalBtnPushed(app, ~)
            app.c = app.c + 1;
            if app.OwnsFigure
                app.FileReadMenu.Enable  = 'off';
                app.LoadFiltMenu.Enable  = 'off';
                app.LoadFilt2Menu.Enable = 'off';
            end

            if ~isempty(app.pinput)
                % phases already loaded
            else
                f1r = csv_to_mvar(app.freq_1.Value);
                f2r = csv_to_mvar(app.freq_2.Value);
                if numel(f1r)~=2
                    errordlg('Incorrect format in Freq range 1. Example: 0.6,2','Error');
                    app.c = app.c - 1; return;
                end
                if numel(f2r)~=2
                    errordlg('Incorrect format in Freq range 2. Example: 0.6,2','Error');
                    app.c = app.c - 1; return;
                end
                app.int1(app.c,:) = f1r;
                app.int2(app.c,:) = f2r;
            end

            ws = str2double(app.window_size.Value);
            if isnan(ws)
                ws = 10/min([app.int1(app.c,:) app.int2(app.c,:)]);
            end
            app.winds(app.c,:)            = ws;
            app.pr(app.c,:)               = str2double(app.prop_const.Value);
            app.ovr(app.c,:)              = str2double(app.overlap.Value);
            app.forder(app.c,:)           = str2double(app.order.Value);
            app.ns(app.c,:)               = str2double(app.surrnum.Value);
            app.confidence_level(app.c,:) = str2double(app.alphasig.Value);

            if ~isempty(app.pinput)
                fl = sprintf('%.3f,%.3f | %.2f | %.2f | %.2f | %d | %d | %d', min(str2num(app.int1)), max(str2num(app.int1)), ws, app.ovr(app.c,:), app.pr(app.c,:), app.forder(app.c,:), app.ns(app.c,:), app.confidence_level(app.c,:)); %#ok<ST2NM>
                f2 = sprintf('%.3f,%.3f | %.2f | %.2f | %.2f | %d | %d | %d', min(str2num(app.int2)), max(str2num(app.int2)), ws, app.ovr(app.c,:), app.pr(app.c,:), app.forder(app.c,:), app.ns(app.c,:), app.confidence_level(app.c,:)); %#ok<ST2NM>
            else
                fl = sprintf('%.3f,%.3f | %.2f | %.2f | %.2f | %d | %d | %d', min(app.int1(app.c,:)), max(app.int1(app.c,:)), ws, app.ovr(app.c,:), app.pr(app.c,:), app.forder(app.c,:), app.ns(app.c,:), app.confidence_level(app.c,:));
                f2 = sprintf('%.3f,%.3f | %.2f | %.2f | %.2f | %d | %d | %d', min(app.int2(app.c,:)), max(app.int2(app.c,:)), ws, app.ovr(app.c,:), app.pr(app.c,:), app.forder(app.c,:), app.ns(app.c,:), app.confidence_level(app.c,:));
            end

            app.interval_list_1.Items{end+1} = fl;
            app.interval_list_2.Items{end+1} = f2;
        end

        function calculateBtnPushed(app, ~)
            x  = app.time_series_2.XLim;
            xl = x .* app.sampling_freq;
            xl(2) = min(xl(2), size(app.sig,2));
            xl(1) = max(xl(1), 1);
            app.sig_cut       = app.sig(:, xl(1):xl(2));
            app.time_axis_cut = app.time_axis(:, xl(1):xl(2));

            if app.OwnsFigure
                app.FileReadMenu.Enable  = 'off';
                app.LoadFiltMenu.Enable  = 'off';
                app.LoadFilt2Menu.Enable = 'off';
            end
            app.calculate_btn.Enable = 'off';

            try
                app.status.Value = 'Calculating...'; drawnow;
                app.h_wait = waitbar(0,'Calculating DBI...','CreateCancelBtn','setappdata(gcbf,''canceling'',1)');
                setappdata(app.h_wait,'canceling',0);
                N = size(app.sig,1)/2;

                for k = 1:app.c
                    win  = app.winds(k);
                    pr   = app.pr(k);
                    ovr  = app.ovr(k);
                    bn   = app.forder(k);
                    ns   = app.ns(k);
                    signif = app.confidence_level(k);

                    for j = 1:N
                        if getappdata(app.h_wait,'canceling'), break; end

                        if ~isempty(app.pinput)
                            phi1 = app.sig_cut(j,:);
                            phi2 = app.sig_cut(j+N,:);
                        else
                            [app.bands{j,k},~]   = loop_butter(app.sig_cut(j,:),   app.int1(k,:), app.sampling_freq);
                            phi1 = angle(hilbert(app.bands{j,k}));
                            [app.bands{j+N,k},~] = loop_butter(app.sig_cut(j+N,:), app.int2(k,:), app.sampling_freq);
                            phi2 = angle(hilbert(app.bands{j+N,k}));
                        end
                        app.p1{j,k} = phi1;
                        app.p2{j,k} = phi2;

                        [app.tm{j,k}, app.cc{j,k}, app.e_noise{j,k}] = bayes_main(phi1, phi2, win, 1/app.sampling_freq, ovr, pr, 0, bn);
                        cpl1_ = zeros(1, size(app.cc{j,k},1));
                        cpl2_ = zeros(1, size(app.cc{j,k},1));
                        for m = 1:size(app.cc{j,k},1)
                            [cpl1_(m), cpl2_(m)] = dirc(app.cc{j,k}(m,:), bn);
                            [~,~,q21(:,:,m), q12(:,:,m)] = CFprint(app.cc{j,k}(m,:), bn);
                        end
                        app.cpl1{j,k} = cpl1_;
                        app.cpl2{j,k} = cpl2_;
                        app.cf1{j,k}  = q21;
                        app.cf2{j,k}  = q12;
                        app.mcf1{j,k} = squeeze(mean(q21,3));
                        app.mcf2{j,k} = squeeze(mean(q12,3));

                        surr1 = surrcalc(phi1, ns, 'CPP', 0, app.sampling_freq);
                        surr2 = surrcalc(phi2, ns, 'CPP', 0, app.sampling_freq);
                        scpl1 = zeros(ns, size(app.cc{j,k},1));
                        scpl2 = zeros(ns, size(app.cc{j,k},1));
                        for n = 1:ns
                            [~, cc_surr_n] = bayes_main(surr1(n,:), surr2(n,:), win, 1/app.sampling_freq, ovr, pr, 1, bn);
                            for idx2 = 1:size(cc_surr_n,1)
                                [scpl1(n,idx2), scpl2(n,idx2)] = dirc(cc_surr_n(idx2,:), bn);
                            end
                        end

                        alph = 1 - (signif/100);
                        K    = floor((ns+1)*alph);
                        if K == 0
                            app.surr_cpl1{j,k} = max(scpl1);
                            app.surr_cpl2{j,k} = max(scpl2);
                        else
                            s1 = sort(scpl1,'descend');
                            s2 = sort(scpl2,'descend');
                            app.surr_cpl1{j,k} = s1(K,:);
                            app.surr_cpl2{j,k} = s2(K,:);
                        end

                        waitbar((j + N*(k-1))/(N*app.c), app.h_wait, sprintf('Calculating DBI: pair %d, set %d', j, k));
                    end
                end

                delete(app.h_wait);
                app.status.Value = 'Calculation complete';
                app.calculate_btn.Enable = 'on';
                if app.OwnsFigure
                    app.SaveCsvMenu.Enable  = 'on';
                    app.SaveMatMenu.Enable  = 'on';
                    app.SaveSessionMenu.Enable = 'on';
                end
                app.intervalList1Changed([]);

            catch e
                errordlg(e.message,'Error');
                app.calculate_btn.Enable = 'on';
                try, delete(app.h_wait); catch; end
                rethrow(e);
            end
        end

        function intervalList1Changed(app, ~)
            if isempty(app.interval_list_1.Items), return; end
            int_select = app.listboxIndex(app.interval_list_1);
            app.setListboxByIndex(app.interval_list_2, int_select);
            app.displayTypeChanged([]);
        end

        function deleteSetBtnPushed(app, ~)
            app = MODAbayes_intdelete(app.UIFigure, [], app);
            app.intervalList1Changed([]);
        end

        function displayTypeChanged(app, ~)
            disp_select = app.dropdownIndex(app.display_type);
            sig_select  = app.listboxIndex(app.signal_list);
            int_select  = app.listboxIndex(app.interval_list_1);
            app.setListboxByIndex(app.interval_list_2, int_select);

            % Remove dashed lines from time series
            app.removeDashedLines(app.time_series_1);
            app.removeDashedLines(app.time_series_2);

            if disp_select == 1
                if app.OwnsFigure
                    app.ExportViewMenu.Enable = 'on';
                    app.OpenViewMenu.Enable   = 'on';
                    app.CfVidMenu.Enable      = 'off';
                end
                app.curr_time.Visible     = 'off';
                xlim_backup = app.time_series_1.XLim;

                % Clear plots pane axes
                cla(app.phi1_axes,'reset'); cla(app.phi2_axes,'reset');
                cla(app.coupling_strength_axis,'reset');
                cla(app.CF1,'reset'); cla(app.CF2,'reset');
                app.time_series_1.XLim = xlim_backup;

                app.scaleon.Visible    = 'off';
                app.time_slider.Visible = 'off';
                app.curr_time.Visible  = 'off';
                app.CF1.Visible        = 'off';
                app.CF2.Visible        = 'off';
                app.phi1_axes.Visible  = 'on';
                app.phi2_axes.Visible  = 'on';
                app.coupling_strength_axis.Visible = 'on';

                app.removeDashedLines(app.time_series_1);
                app.removeDashedLines(app.time_series_2);

                if ~isempty(app.p1) && size(app.p1,2) >= int_select
                    plot(app.phi1_axes, app.time_axis_cut, app.p1{sig_select,int_select}, 'color', app.linecol(1,:));
                    plot(app.phi2_axes, app.time_axis_cut, app.p2{sig_select,int_select}, 'color', app.linecol(1,:));
                    xlim(app.phi1_axes, [app.time_axis_cut(1) app.time_axis_cut(end)]);
                    xlim(app.phi2_axes, [app.time_axis_cut(1) app.time_axis_cut(end)]);
                    ylabel(app.phi1_axes,'phi1'); ylabel(app.phi2_axes,'phi2');
                    app.phi1_axes.XTickLabel = {}; app.phi2_axes.XTickLabel = {};

                    hold(app.coupling_strength_axis,'on');
                    tm_offset = app.tm{sig_select,int_select} + app.time_axis_cut(1);
                    plot(app.coupling_strength_axis, tm_offset, app.cpl1{sig_select,int_select}, 'color', app.linecol(1,:), 'linewidth', app.line2width);
                    plot(app.coupling_strength_axis, tm_offset, app.cpl2{sig_select,int_select}, 'color', app.linecol(2,:), 'linewidth', app.line2width);
                    plot(app.coupling_strength_axis, tm_offset, app.surr_cpl1{sig_select,int_select}, 'color', app.linecol(1,:), 'linewidth', app.line2width, 'LineStyle','--');
                    plot(app.coupling_strength_axis, tm_offset, app.surr_cpl2{sig_select,int_select}, 'color', app.linecol(2,:), 'linewidth', app.line2width, 'LineStyle','--');
                    app.leg1 = {'Data 2->1','Data 1->2','Surr 2->1','Surr 1->2'};
                    legend(app.coupling_strength_axis, app.leg1, 'orientation','horizontal');
                    xlabel(app.coupling_strength_axis,'Time (s)');
                    ylabel(app.coupling_strength_axis,'Coupling Strength');
                    xlim(app.coupling_strength_axis,[app.time_axis_cut(1) app.time_axis_cut(end)]);
                    linkaxes([app.time_series_1 app.time_series_2 app.phi1_axes app.phi2_axes app.coupling_strength_axis],'x');
                    % Only the two raw-signal previews are meant to be
                    % dragged/zoomed directly — phi1/phi2/coupling-strength
                    % just mirror the x-range via linkaxes. Disabling their
                    % own interactivity avoids a second, redundant
                    % synchronized redraw of this 5-axis linked group if a
                    % drag accidentally starts on one of them (the likely
                    % cause of "controls lag when interacting with the graph").
                    for ax = {app.phi1_axes, app.phi2_axes, app.coupling_strength_axis}
                        disableDefaultInteractivity(ax{1});
                        ax{1}.Toolbar.Visible = 'off';
                    end
                end

            elseif disp_select == 2
                if app.OwnsFigure
                    app.ExportViewMenu.Enable = 'on';
                    app.OpenViewMenu.Enable   = 'on';
                    app.CfVidMenu.Enable      = 'on';
                end

                cla(app.phi1_axes,'reset');             app.phi1_axes.Visible  = 'off';
                cla(app.phi2_axes,'reset');             app.phi2_axes.Visible  = 'off';
                cla(app.coupling_strength_axis,'reset'); app.coupling_strength_axis.Visible = 'off';
                cla(app.CF1);  app.CF1.Visible = 'on';
                cla(app.CF2);  app.CF2.Visible = 'on';
                app.scaleon.Visible  = 'on';
                app.time_slider.Visible = 'on';
                app.curr_time.Visible = 'off';
                app.time_slider.Value = 0;

                % Re-plot signal pair
                xl = csv_to_mvar(app.xlim_field.Value);
                plot(app.time_series_1, app.time_axis_cut, app.sig_cut(sig_select,:), 'color', app.linecol(1,:));
                xlim(app.time_series_1, xl);
                plot(app.time_series_2, app.time_axis_cut, app.sig_cut(sig_select+size(app.sig,1)/2,:), 'color', app.linecol(1,:));
                xlim(app.time_series_2, xl);
                ylabel(app.time_series_1,'Sig 1'); ylabel(app.time_series_2,'Sig 2');
                app.time_series_1.XTickLabel = {};

                if ~isempty(app.tm) && size(app.tm,1) >= sig_select && size(app.tm,2) >= int_select
                    maxval = size(app.tm{sig_select,int_select},2);
                    app.time_slider.Limits    = [0 maxval];
                    app.time_slider.MajorTicks = [];

                    t1=0:0.13:2*pi; t2=0:0.13:2*pi;
                    cf1m = squeeze(mean(app.cf1{sig_select,int_select},3));
                    cf2m = squeeze(mean(app.cf2{sig_select,int_select},3));
                    app.cfmax = max(max([max(max(cf1m)) max(max(cf2m))]));
                    app.cfmin = min(min([min(min(cf1m)) min(min(cf2m))]));

                    app.curr_time.Visible = 'off';
                    surf(app.CF1,t1,t2,app.mcf1{sig_select,int_select},'FaceColor','interp');
                    surf(app.CF2,t1,t2,app.mcf2{sig_select,int_select},'FaceColor','interp');
                    app.labelCFAxes();
                    colormap(app.CF1,app.cmap); colormap(app.CF2,app.cmap);
                    xlim(app.CF1,[0 2*pi]); ylim(app.CF1,[0 2*pi]);
                    xlim(app.CF2,[0 2*pi]); ylim(app.CF2,[0 2*pi]);
                    if app.scaleon.Value
                        app.CF1.ZLim = [app.cfmin app.cfmax];
                        app.CF2.ZLim = [app.cfmin app.cfmax];
                    end
                end
            end
        end

        function labelCFAxes(app)
            xlabel(app.CF1,'\phi_1'); ylabel(app.CF1,'\phi_2'); zlabel(app.CF1,'q_1(\phi_1,\phi_2)');
            xlabel(app.CF2,'\phi_1'); ylabel(app.CF2,'\phi_2'); zlabel(app.CF2,'q_2(\phi_1,\phi_2)');
            view(app.CF1,[-40 50]); view(app.CF2,[-40 50]);
        end

        function timeSliderValueChanged(app, ~)
            if app.dropdownIndex(app.display_type) ~= 2, return; end
            sig_select = app.listboxIndex(app.signal_list);
            int_select = app.listboxIndex(app.interval_list_1);
            win        = app.winds(int_select);
            slider_val = round(app.time_slider.Value);

            app.removeDashedLines(app.time_series_1);
            app.removeDashedLines(app.time_series_2);

            if slider_val == 0
                app.intervalList1Changed([]);
                return;
            end

            t1=0:0.13:2*pi; t2=0:0.13:2*pi;
            app.tp1 = (app.tm{sig_select,int_select}(slider_val) - win/2) + app.time_axis_cut(1);
            app.tp2 = (app.tm{sig_select,int_select}(slider_val) + win/2) + app.time_axis_cut(1);

            for ax = {app.time_series_1, app.time_series_2}
                yl = ax{1}.YLim;
                hold(ax{1},'on');
                plot(ax{1},[app.tp1 app.tp1],yl,'--','color',[0.85 0.325 0.098],'linewidth',1);
                plot(ax{1},[app.tp2 app.tp2],yl,'--','color',[0.85 0.325 0.098],'linewidth',1);
                hold(ax{1},'off');
            end

            app.cfmax = max([max(max(app.cf1{sig_select,int_select}(:,:,slider_val))) max(max(app.cf2{sig_select,int_select}(:,:,slider_val)))]);
            app.cfmin = min([min(min(app.cf1{sig_select,int_select}(:,:,slider_val))) min(min(app.cf2{sig_select,int_select}(:,:,slider_val)))]);

            surf(app.CF1,t1,t2,app.cf1{sig_select,int_select}(:,:,slider_val),'FaceColor','interp');
            surf(app.CF2,t1,t2,app.cf2{sig_select,int_select}(:,:,slider_val),'FaceColor','interp');
            app.labelCFAxes();
            xlim(app.CF1,[0 2*pi]); ylim(app.CF1,[0 2*pi]);
            xlim(app.CF2,[0 2*pi]); ylim(app.CF2,[0 2*pi]);
            if app.scaleon.Value
                app.CF1.ZLim = [app.cfmin app.cfmax];
                app.CF2.ZLim = [app.cfmin app.cfmax];
            end
            app.scaleon.Visible = 'on';
            app.curr_time.Text  = sprintf('%s - %s s', num2str(app.tp1), num2str(app.tp2));
            app.curr_time.Visible = 'on';
            view(app.CF1,[-40 50]); view(app.CF2,[-40 50]);
        end

        function scaleonValueChanged(app, ~)
            app.timeSliderValueChanged([]);
        end

        function removeDashedLines(~, ax)
            ch = allchild(ax);
            for j = 1:numel(ch)
                if strcmpi(get(ch(j),'Type'),'Line') && strcmp(get(ch(j),'linestyle'),'--')
                    delete(ch(j));
                end
            end
        end

        function setListboxByIndex(~, lb, idx)
            if idx < 1 || idx > numel(lb.Items), return; end
            lb.Value = lb.Items{idx};
        end

        % ---- Save ---------------------------------------------------

        function saveCsvMenuSelected(app, ~)
            try
                Bayes_data = app.buildBayesDataStruct();
                csvsavefolder(Bayes_data);
            catch e
                errordlg(e.message,'Error'); rethrow(e);
            end
        end

        function saveMatMenuSelected(app, ~)
            try
                [FileName,PathName] = uiputfile('*.mat','Save as');
                if isequal(FileName,0), return; end
                save_location = fullfile(PathName, FileName);
                Bayes_data = app.buildBayesDataStruct();
                save(save_location,'Bayes_data');
            catch e
                errordlg(e.message,'Error'); rethrow(e);
            end
        end

        % ---- Export current view (replaces the old dead 7-item Plot menu) ----
        function axs = currentViewAxes(app)
            candidates = {app.phi1_axes, app.phi2_axes, app.coupling_strength_axis, app.CF1, app.CF2};
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
            cols = max(1, min(n,2));
            rows = max(1, ceil(n/cols));
            fig = figure('Visible','off','Position',[100 100 480*cols 380*rows]);
            for i = 1:n
                newAx = copyobj(axs{i}, fig);
                subplot(rows, cols, i, newAx);
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

        function cfVidMenuSelected(app, ~)
            int_select = app.listboxIndex(app.interval_list_1);
            sig_select = app.listboxIndex(app.signal_list);
            c1  = app.cf1{sig_select,int_select};
            c2  = app.cf2{sig_select,int_select};
            win = app.winds(int_select);
            t   = app.tm{sig_select,int_select};
            [FileName,PathName] = uiputfile('*.avi','Save video as');
            if isequal(FileName,0), return; end
            save_location = fullfile(PathName, FileName);
            v = VideoWriter(save_location); v.FrameRate=2; open(v);
            hf = figure('position',[100 100 1200 500]);
            t1=0:0.13:2*pi; t2=0:0.13:2*pi;
            for j=1:size(c1,3)
                cfm = max([max(max(c1(:,:,j))) max(max(c2(:,:,j)))]);
                cfn = min([min(min(c1(:,:,j))) min(min(c2(:,:,j)))]);
                ax1=subplot(1,2,1,'parent',hf); ax2=subplot(1,2,2,'parent',hf);
                surf(ax1,t1,t2,c1(:,:,j),'FaceColor','interp');
                surf(ax2,t1,t2,c2(:,:,j),'FaceColor','interp');
                xlabel(ax1,'\phi_1'); ylabel(ax1,'\phi_2'); zlabel(ax1,'q_1');
                xlabel(ax2,'\phi_1'); ylabel(ax2,'\phi_2'); zlabel(ax2,'q_2');
                tp = t(j) + app.time_axis_cut(1);
                title(ax1,sprintf('t=%.2f s, win=%.2f s',tp,win));
                title(ax2,sprintf('CF 1-->2'));
                xlim(ax1,[0 2*pi]); ylim(ax1,[0 2*pi]); ax1.ZLim=[cfn cfm];
                xlim(ax2,[0 2*pi]); ylim(ax2,[0 2*pi]); ax2.ZLim=[cfn cfm];
                drawnow; writeVideo(v, getframe(hf));
            end
            close(v); close(hf);
        end

        function saveSessionMenuSelected(app, ~)
            MODAsave(app);
        end

        function Bd = buildBayesDataStruct(app)
            Bd.phase1                          = app.p1;
            Bd.phase2                          = app.p2;
            Bd.sampling_freq                   = app.sampling_freq;
            Bd.time                            = app.time_axis_cut;
            Bd.interval1                       = app.int1;
            Bd.interval2                       = app.int2;
            Bd.surrnum                         = app.ns;
            Bd.confidence_level                = app.confidence_level;
            Bd.Bayes_win                       = app.winds;
            Bd.overlap                         = app.ovr;
            Bd.propagation_const               = app.pr;
            Bd.Fourier_base                    = app.forder;
            Bd.Bayestime                       = app.tm;
            Bd.coupling_strength_2to1          = app.cpl1;
            Bd.coupling_strength_1to2          = app.cpl2;
            Bd.coupling_function_2to1          = app.cf1;
            Bd.coupling_function_1to2          = app.cf2;
            Bd.mean_cfunc_2to1                 = app.mcf1;
            Bd.mean_cfunc_1to2                 = app.mcf2;
            Bd.surrogate_coupling_strength_2to1 = app.surr_cpl1;
            Bd.surrogate_coupling_strength_1to2 = app.surr_cpl2;
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
                app.UIFigure = uifigure('Visible','off','Position',[100 100 W H],'Resize','off','Name','MODA v1.01 Bayesian Inference');
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
                app.FileMenu       = uimenu(app.UIFigure,'Text','File');
                app.ResetGUIMenu   = uimenu(app.FileMenu,'Text','Reset GUI','MenuSelectedFcn',@(s,e)app.resetGUIMenuSelected(e));
                app.FileReadMenu   = uimenu(app.FileMenu,'Text','Load time series','MenuSelectedFcn',@(s,e)app.fileReadMenuSelected(e));
                app.LoadFiltMenu   = uimenu(app.FileMenu,'Text','Load from filtering (single)','MenuSelectedFcn',@(s,e)app.loadFiltMenuSelected(e));
                app.LoadFilt2Menu  = uimenu(app.FileMenu,'Text','Load from filtering (two files)','MenuSelectedFcn',@(s,e)app.loadFilt2MenuSelected(e));
                app.LoadSessionMenu= uimenu(app.FileMenu,'Text','Load session','MenuSelectedFcn',@(s,e)app.loadSessionMenuSelected(e));

                % Replaces the old 7-item Plot menu (none of which had a
                % MenuSelectedFcn wired — dead menu items) with two working
                % actions that act on whichever axes are currently visible.
                app.PlotMenu       = uimenu(app.UIFigure,'Text','Plot');
                app.ExportViewMenu = uimenu(app.PlotMenu,'Text','Export current view...','Enable','off', ...
                    'MenuSelectedFcn',@(s,e)app.exportViewMenuSelected(e));
                app.OpenViewMenu   = uimenu(app.PlotMenu,'Text','Open current view in new figure','Enable','off', ...
                    'MenuSelectedFcn',@(s,e)app.openViewMenuSelected(e));

                app.SaveMenu       = uimenu(app.UIFigure,'Text','Save');
                app.SaveCsvMenu    = uimenu(app.SaveMenu,'Text','Save .csv','Enable','off','MenuSelectedFcn',@(s,e)app.saveCsvMenuSelected(e));
                app.SaveMatMenu    = uimenu(app.SaveMenu,'Text','Save .mat','Enable','off','MenuSelectedFcn',@(s,e)app.saveMatMenuSelected(e));
                app.CfVidMenu      = uimenu(app.SaveMenu,'Text','Save CF video','Enable','off','MenuSelectedFcn',@(s,e)app.cfVidMenuSelected(e));
                app.SaveSessionMenu= uimenu(app.SaveMenu,'Text','Save session','Enable','off','MenuSelectedFcn',@(s,e)app.saveSessionMenuSelected(e));
            end

            % Only when this module owns its figure — embedded in MODAApp's
            % tab, the logos would overlap the results panel instead of
            % adding anything (MODAApp already shows its own top-bar banner).
            if app.OwnsFigure
                app.anchorBrandingLogos();
            end

            % ---- Left control sidebar (consistent with the Coherence/
            % Bispectrum/TFA screens: one scrollable panel holding every
            % control, instead of controls scattered on the right while
            % results sat on the left) ----
            ctrlPanel = uipanel(app.RootContainer,'Position',[0 0 330 795],'Title','','Scrollable','on');

            yl = 750;
            uilabel(ctrlPanel,'Position',[5 yl 150 20],'Text','Select Signal Pair');
            app.signal_list = uilistbox(ctrlPanel,'Position',[5 yl-90 320 90],'Items',{}, ...
                'ValueChangedFcn',@(s,e)app.signalListChanged(e));

            yl = yl - 120;
            uilabel(ctrlPanel,'Position',[5 yl 320 20],'Text','Status:');
            app.status = uieditfield(ctrlPanel,'text','Position',[5 yl-24 320 22],'Value','Please Import Signal');

            yl = yl - 55;
            app.calculate_btn = uibutton(ctrlPanel,'push','Position',[5 yl 320 28],'Text','Calculate', ...
                'ButtonPushedFcn',@(s,e)app.calculateBtnPushed(e));

            yl = yl - 40;
            uilabel(ctrlPanel,'Position',[5 yl 90 20],'Text','Freq range 1:');
            app.freq_1 = uieditfield(ctrlPanel,'text','Position',[100 yl 220 22]);
            yl = yl - 30;
            uilabel(ctrlPanel,'Position',[5 yl 90 20],'Text','Freq range 2:');
            app.freq_2 = uieditfield(ctrlPanel,'text','Position',[100 yl 220 22]);

            % Bayesian inference parameters
            yl = yl - 40;
            uilabel(ctrlPanel,'Position',[5 yl 140 20],'Text','Window Size (s):');
            app.window_size = uieditfield(ctrlPanel,'text','Position',[148 yl 100 22]);
            yl = yl - 30;
            uilabel(ctrlPanel,'Position',[5 yl 140 20],'Text','Overlap:');
            app.overlap = uieditfield(ctrlPanel,'text','Value','1','Position',[148 yl 100 22]);
            yl = yl - 30;
            uilabel(ctrlPanel,'Position',[5 yl 140 20],'Text','Order (FO):');
            app.order = uieditfield(ctrlPanel,'text','Value','2','Position',[148 yl 100 22]);
            yl = yl - 30;
            uilabel(ctrlPanel,'Position',[5 yl 140 20],'Text','Confidence:');
            app.alphasig = uieditfield(ctrlPanel,'text','Value','95','Position',[148 yl 100 22]);
            yl = yl - 30;
            uilabel(ctrlPanel,'Position',[5 yl 140 20],'Text','Propagation:');
            app.prop_const = uieditfield(ctrlPanel,'text','Value','.2','Position',[148 yl 100 22]);
            yl = yl - 30;
            uilabel(ctrlPanel,'Position',[5 yl 140 20],'Text','Num. surrogates:');
            app.surrnum = uieditfield(ctrlPanel,'text','Value','19','Position',[148 yl 100 22]);

            yl = yl - 40;
            app.add_interval_btn = uibutton(ctrlPanel,'push','Position',[5 yl 155 28],'Text','Add parameter set', ...
                'ButtonPushedFcn',@(s,e)app.addIntervalBtnPushed(e));
            app.delete_set_btn = uibutton(ctrlPanel,'push','Position',[165 yl 155 28],'Text','Delete parameter set', ...
                'ButtonPushedFcn',@(s,e)app.deleteSetBtnPushed(e));

            yl = yl - 40;
            uilabel(ctrlPanel,'Position',[5 yl 150 18],'Text','Freq band 1');
            app.interval_list_1 = uilistbox(ctrlPanel,'Position',[5 yl-75 155 75],'Items',{}, ...
                'ValueChangedFcn',@(s,e)app.intervalList1Changed(e));
            uilabel(ctrlPanel,'Position',[165 yl 150 18],'Text','Freq band 2');
            app.interval_list_2 = uilistbox(ctrlPanel,'Position',[165 yl-75 155 75],'Items',{});

            % Limits
            yl = yl - 105;
            uilabel(ctrlPanel,'Position',[5 yl 75 20],'Text','X Limits:');
            app.xlim_field = uieditfield(ctrlPanel,'text','Position',[85 yl 235 22]);
            yl = yl - 30;
            uilabel(ctrlPanel,'Position',[5 yl 60 20],'Text','Length:');
            app.length_field = uieditfield(ctrlPanel,'text','Position',[70 yl 100 22]);
            app.refresh_limits_btn = uibutton(ctrlPanel,'push','Position',[180 yl 100 22],'Text','Refresh', ...
                'ButtonPushedFcn',@(s,e)app.refreshLimitsBtnPushed(e));

            % Display mode + Coupling-Function-only controls
            yl = yl - 40;
            uilabel(ctrlPanel,'Position',[5 yl 90 20],'Text','Display:');
            app.display_type = uidropdown(ctrlPanel,'Position',[100 yl 220 22], ...
                'Items',{'Phase + Coupling Strength','Coupling Functions'},'ValueChangedFcn',@(s,e)app.displayTypeChanged(e));
            yl = yl - 30;
            app.scaleon = uicheckbox(ctrlPanel,'Text','Match Z-scale','Value',false,'Visible','off','Position',[5 yl 200 22], ...
                'ValueChangedFcn',@(s,e)app.scaleonValueChanged(e));
            yl = yl - 30;
            uilabel(ctrlPanel,'Position',[5 yl 320 18],'Text','Start:Time Evolution:End; Go to start for mean','Visible','off');
            yl = yl - 22;
            app.time_slider = uislider(ctrlPanel,'Value',0,'Limits',[0 1],'Visible','off','Position',[10 yl 300 3], ...
                'ValueChangedFcn',@(s,e)app.timeSliderValueChanged(e));
            app.curr_time = uilabel(ctrlPanel,'Text','','Visible','off','Position',[5 yl-25 200 20]);

            % ---- Time series pair panel (top right) ----
            app.TimePairPanel = uipanel(app.RootContainer,'Position',[330 500 1270 355],'Title','Time Series');
            TPW = 1270; TPH = 355;
            app.time_series_1 = uiaxes(app.TimePairPanel,'Position',round([0.0823*TPW, 0.6165*TPH, 0.891*TPW, 0.3137*TPH]));
            app.time_series_2 = uiaxes(app.TimePairPanel,'Position',round([0.0823*TPW, 0.2045*TPH, 0.891*TPW, 0.3137*TPH]));

            % ---- Plots pane (bottom right) ----
            app.PlotsPane = uipanel(app.RootContainer,'Position',[330 0 1270 500],'Title','Results');
            PPW = 1270; PPH = 500;
            app.phi1_axes              = uiaxes(app.PlotsPane,'Position',round([0.0779*PPW, 0.7221*PPH, 0.8921*PPW, 0.262*PPH]));
            app.phi2_axes              = uiaxes(app.PlotsPane,'Position',round([0.0779*PPW, 0.426*PPH,  0.8921*PPW, 0.262*PPH]));
            app.coupling_strength_axis = uiaxes(app.PlotsPane,'Position',round([0.0781*PPW, 0.1304*PPH, 0.8919*PPW, 0.2609*PPH]));
            app.CF2 = uiaxes(app.PlotsPane,'Position',round([0.5673*PPW, 0.1913*PPH, 0.3893*PPW, 0.7403*PPH]));
            app.CF1 = uiaxes(app.PlotsPane,'Position',round([0.0901*PPW, 0.1906*PPH, 0.3914*PPW, 0.7423*PPH]));

            % CF1/CF2 initially off
            app.CF1.Visible = 'off'; app.CF2.Visible = 'off';

            if app.OwnsFigure
                app.UIFigure.Visible = 'on';
            end
        end
    end

    methods (Access = public)
        function app = Bayesian(parentContainer)
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
            if app.OwnsFigure
                app.ExportViewMenu.Enable = 'off';
                app.OpenViewMenu.Enable   = 'off';
                app.SaveCsvMenu.Enable   = 'off';
                app.SaveMatMenu.Enable   = 'off';
                app.CfVidMenu.Enable     = 'off';
                app.SaveSessionMenu.Enable = 'off';
            end
        end
    end
end
