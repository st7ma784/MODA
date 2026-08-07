classdef MODAApp < matlab.apps.AppBase
    % MODAApp - Main Application Window for MODA v2.0
    %
    % This is the modernized version of MODA created with App Designer.
    % Compatible with MATLAB R2023a and later (tested up to R2026a)
    %
    % Original GUIDE version created: October 3, 2017
    % Refactored to App Designer: March 2026
    %
    % Core Team:
    % - Wavelet Transform: Dmytro Iatsenko
    % - Bayesian Inference: Tomislav Stankovski
    % - Original GUI: MODA Development Team
    % - App Designer Modernization: 2026 Development Team

    properties (Access = public)
        UIFigure                        matlab.ui.Figure
    end

    properties (Access = private)
        % UI Components - Main Window
        LogoAxes                        matlab.ui.control.UIAxes
        LogoImage                       matlab.ui.control.Image
        MainGridLayout                  matlab.ui.container.GridLayout
        ExitButton                      matlab.ui.control.Button

        % Tabs — one per module, built lazily on first visit. No landing
        % page: the app opens directly into the first module.
        MainTabGroup                     matlab.ui.container.TabGroup
        TimeFrequencyTab                 matlab.ui.container.Tab
        TimeFrequencyApp   % embedded TimeFrequencyAnalysis instance, built lazily
        CoherenceTab                     matlab.ui.container.Tab
        CoherenceApp   % embedded CoherenceMulti instance, built lazily
        FilteringTab                     matlab.ui.container.Tab
        FilteringApp   % embedded Filtering instance, built lazily
        BispectralTab                    matlab.ui.container.Tab
        BispectralApp   % embedded Bispectrum instance, built lazily
        BayesianTab                      matlab.ui.container.Tab
        BayesianApp   % embedded Bayesian instance, built lazily

        % Status bar
        StatusBar                        matlab.ui.control.Label

        % Application state
        AppVersion = "2.0"
        MATLABMinVersion = "R2023a"
        LastUpdated = "April 14, 2026"
    end

    methods (Access = public)
        function app = MODAApp(varargin)
            % Constructor - automatically called when app launches

            % Ensure all module GUIs and shared helper code are on the path
            % (modules below live under allguis/, plus shared helpers like
            % MODAsettings/MODAread under allguis/codes/Universal and
            % cmap.mat/images under allguis/codes/cmap, allguis/images)
            repoRoot = fileparts(mfilename('fullpath'));
            addpath(genpath(fullfile(repoRoot, 'allguis')));

            createComponents(app);

            % Check MATLAB version
            app.checkMATLABVersion();

            % Set app title and metadata
            app.UIFigure.Name = sprintf('MODA v%s - Multiscale Oscillatory Dynamics Analysis', app.AppVersion);

            % Execute startup function
            runStartupFcn(app, @startupFcn);
        end

        function delete(app)
            % Destructor - cleanup operations when app closes
            embeddedApps = {app.TimeFrequencyApp, app.CoherenceApp, app.FilteringApp, ...
                             app.BispectralApp, app.BayesianApp};
            for a = embeddedApps
                if ~isempty(a{1}) && isvalid(a{1})
                    delete(a{1});
                end
            end
            if isvalid(app.UIFigure)
                delete(app.UIFigure);
            end
        end
    end

    methods (Access = private)

        function createComponents(app)
            % Create main figure. Sized to comfortably host an embedded
            % module tab (modules were designed for a 1600x860 canvas).
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [50 50 1650 1000];
            app.UIFigure.CloseRequestFcn = createCallbackFcn(app, @UIFigureCloseRequest, true);

            % Create main grid layout
            app.MainGridLayout = uigridlayout(app.UIFigure);
            app.MainGridLayout.ColumnWidth = {'1x', '1x', '1x'};
            app.MainGridLayout.RowHeight = {80, '1x', 50};
            app.MainGridLayout.Padding = [10 10 10 10];
            app.MainGridLayout.RowSpacing = 10;
            app.MainGridLayout.ColumnSpacing = 10;

            % ===== TOP BAR: LOGO (left) + EXIT (far right) =====
            logoPanel = uipanel(app.MainGridLayout);
            logoPanel.BackgroundColor = [1 1 1];
            logoPanel.BorderType = 'none';
            logoPanel.Layout.Row = 1;
            logoPanel.Layout.Column = [1 2];

            % uiimage is the purpose-built component for a static logo/banner
            % — unlike uiaxes+image(), its ScaleMethod ('fit') preserves
            % aspect ratio automatically instead of stretching/warping.
            % uiimage does NOT auto-resize with a plain uipanel parent
            % though (no layout manager there), so it's nested in a
            % single-cell uigridlayout, which DOES actively resize its
            % child on every window resize — that's what keeps the banner
            % correctly scaled live as the window changes size.
            imgPath = which('frontbanner.png');
            if ~isempty(imgPath)
                logoGrid = uigridlayout(logoPanel, [1 1]);
                logoGrid.Padding = [0 0 0 0];
                logoGrid.BackgroundColor = [1 1 1];
                app.LogoImage = uiimage(logoGrid);
                app.LogoImage.ImageSource = imgPath;
                app.LogoImage.ScaleMethod = 'fit';
            else
                app.LogoAxes = uiaxes(logoPanel);
                app.LogoAxes.Units = 'normalized';
                app.LogoAxes.Position = [0 0 1 1];
                app.LogoAxes.XLim = [0 1];
                app.LogoAxes.YLim = [0 1];
                axis(app.LogoAxes, 'off');
                text(app.LogoAxes, 0.5, 0.5, 'MODA v2.0', ...
                    'HorizontalAlignment', 'center', ...
                    'VerticalAlignment', 'middle', ...
                    'FontSize', 28, ...
                    'FontWeight', 'bold', ...
                    'Color', [0.2 0.4 0.7]);
            end

            app.ExitButton = uibutton(app.MainGridLayout, 'push');
            app.ExitButton.Layout.Row = 1;
            app.ExitButton.Layout.Column = 3;
            app.ExitButton.Text = 'Exit';
            app.ExitButton.FontSize = 14;
            app.ExitButton.FontWeight = 'bold';
            app.ExitButton.BackgroundColor = [0.8 0.3 0.3];
            app.ExitButton.FontColor = [1 1 1];
            app.ExitButton.ButtonPushedFcn = createCallbackFcn(app, @ExitPushed, true);

            % ===== MIDDLE SECTION: MODULE TABS (no landing page) =====
            app.MainTabGroup = uitabgroup(app.MainGridLayout);
            app.MainTabGroup.Layout.Row = 2;
            app.MainTabGroup.Layout.Column = [1 3];
            app.MainTabGroup.SelectionChangedFcn = createCallbackFcn(app, @MainTabGroupSelectionChanged, true);

            app.TimeFrequencyTab = uitab(app.MainTabGroup, 'Title', 'Time-Frequency Analysis');
            app.CoherenceTab     = uitab(app.MainTabGroup, 'Title', 'Coherence');
            app.FilteringTab     = uitab(app.MainTabGroup, 'Title', 'Filtering');
            app.BispectralTab    = uitab(app.MainTabGroup, 'Title', 'Bispectral');
            app.BayesianTab      = uitab(app.MainTabGroup, 'Title', 'Bayesian');

            % ===== BOTTOM SECTION: STATUS BAR =====
            app.StatusBar = uilabel(app.MainGridLayout);
            app.StatusBar.Layout.Row = 3;
            app.StatusBar.Layout.Column = [1 3];
            app.StatusBar.Text = sprintf('MODA v%s | MATLAB %s+ | Last Updated: %s', ...
                app.AppVersion, app.MATLABMinVersion, app.LastUpdated);
            app.StatusBar.HorizontalAlignment = 'left';
            app.StatusBar.BackgroundColor = [0.9 0.9 0.9];
            app.StatusBar.FontSize = 10;
        end

        function startupFcn(app)
            % Startup function - runs when app first launches

            % Make figure visible
            movegui(app.UIFigure, 'center');
            app.UIFigure.Visible = 'on';

            % Check toolbox availability
            app.checkToolboxes();

            % Initialize app state
            app.initializeAppState();

            % No landing page — load and show the first module immediately.
            app.ensureModuleLoaded('TimeFrequency');
        end

        function checkMATLABVersion(app)
            % Verify MATLAB version is compatible
            v = version('-release');
            releaseYear = str2double(v(1:4));

            if releaseYear < 2023
                uialert(app.UIFigure, ...
                    sprintf('MODA v%s requires MATLAB R2023a or later.\nYou are using: %s\n\nSome features may not work correctly.', ...
                    app.AppVersion, v), ...
                    'MATLAB Version Warning');
            end
        end

        function checkToolboxes(app)
            % Verify required toolboxes are installed
            requiredToolboxes = {
                'Signal Processing Toolbox'
                'Wavelet Toolbox'
                'Statistics and Machine Learning Toolbox'
            };

            installedToolboxes = ver;
            installedNames = {installedToolboxes.Name};

            missingToolboxes = {};
            for i = 1:length(requiredToolboxes)
                if ~any(strcmpi(installedNames, requiredToolboxes{i}))
                    missingToolboxes{end+1} = requiredToolboxes{i};
                end
            end

            if ~isempty(missingToolboxes)
                msg = sprintf('The following toolboxes are required but not installed:\n\n');
                for i = 1:length(missingToolboxes)
                    msg = [msg, sprintf('  - %s\n', missingToolboxes{i})];
                end
                msg = [msg, sprintf('\nPlease install these toolboxes to use all features.')];

                uialert(app.UIFigure, msg, 'Missing Toolboxes', 'icon', 'warning');
            end
        end

        function initializeAppState(app)
            % Initialize application state variables
            % (Can be extended for persistent state management)
        end

        % ===== Tab navigation =====

        function MainTabGroupSelectionChanged(app, event)
            % Lazily build whichever module tab the user just selected.
            % Only fires on user-driven tab clicks, not programmatic
            % selection (see startupFcn, which loads the first tab directly).
            tab = event.NewValue;
            if tab == app.TimeFrequencyTab
                app.ensureModuleLoaded('TimeFrequency');
            elseif tab == app.CoherenceTab
                app.ensureModuleLoaded('Coherence');
            elseif tab == app.FilteringTab
                app.ensureModuleLoaded('Filtering');
            elseif tab == app.BispectralTab
                app.ensureModuleLoaded('Bispectral');
            elseif tab == app.BayesianTab
                app.ensureModuleLoaded('Bayesian');
            end
        end

        function ensureModuleLoaded(app, moduleName)
            % Builds the named module into its tab on first visit, and
            % reuses it thereafter so switching away and back preserves
            % state. Shows a watch cursor + status text (with an explicit
            % drawnow) while building, since construction can take a
            % noticeable moment and otherwise the UI looks unresponsive.
            [tabProp, appProp, ctor, label] = app.moduleInfo(moduleName);
            tab = app.(tabProp);

            if isempty(app.(appProp)) || ~isvalid(app.(appProp))
                app.updateStatus(['Loading ' label '...']);
                app.UIFigure.Pointer = 'watch';
                drawnow;
                try
                    app.(appProp) = ctor(tab);
                    app.updateStatus([label ' ready']);
                catch ME
                    app.UIFigure.Pointer = 'arrow';
                    uialert(app.UIFigure, ...
                        sprintf('Error launching %s:\n\n%s', label, ME.message), ...
                        'Error', 'icon', 'error');
                    app.updateStatus('Error - see message above');
                    app.UIFigure.Pointer = 'arrow';
                    return;
                end
                app.UIFigure.Pointer = 'arrow';
            end
            app.MainTabGroup.SelectedTab = tab;
        end

        function [tabProp, appProp, ctor, label] = moduleInfo(~, moduleName)
            switch moduleName
                case 'TimeFrequency'
                    tabProp = 'TimeFrequencyTab'; appProp = 'TimeFrequencyApp';
                    ctor = @TimeFrequencyAnalysis; label = 'Time-Frequency Analysis';
                case 'Coherence'
                    tabProp = 'CoherenceTab'; appProp = 'CoherenceApp';
                    ctor = @CoherenceMulti; label = 'Wavelet Phase Coherence';
                case 'Filtering'
                    tabProp = 'FilteringTab'; appProp = 'FilteringApp';
                    ctor = @Filtering; label = 'Signal Filtering';
                case 'Bispectral'
                    tabProp = 'BispectralTab'; appProp = 'BispectralApp';
                    ctor = @Bispectrum; label = 'Bispectral Analysis';
                case 'Bayesian'
                    tabProp = 'BayesianTab'; appProp = 'BayesianApp';
                    ctor = @Bayesian; label = 'Bayesian Inference';
            end
        end

        function ExitPushed(app, event)
            % Exit button callback
            delete(app);
        end

        function UIFigureCloseRequest(app, event)
            % Handle window close button
            delete(app);
        end

        function updateStatus(app, statusText)
            % Update status bar with message
            app.StatusBar.Text = sprintf('[%s] %s', datetime('now', 'Format', 'HH:mm:ss'), statusText);
        end
    end

    methods (Access = public)
        function setAppVersion(app, versionStr)
            % Allows external updates to app version
            app.AppVersion = versionStr;
            app.UIFigure.Name = sprintf('MODA v%s - Multiscale Oscillatory Dynamics Analysis', app.AppVersion);
        end
    end
end
