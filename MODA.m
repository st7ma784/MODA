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
        MainGridLayout                  matlab.ui.container.GridLayout
        
        % Buttons for main analysis modules
        TimeFrequencyButton              matlab.ui.control.Button
        CoherenceButton                  matlab.ui.control.Button
        FilteringButton                  matlab.ui.control.Button
        BispectralButton                 matlab.ui.control.Button
        BayesianButton                   matlab.ui.control.Button
        ExitButton                       matlab.ui.control.Button
        
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
            createComponents(app);
            registerAppComponents(app);
            
            % Check MATLAB version
            app.checkMATLABVersion();
            
            % Set app title and metadata
            app.UIFigure.Name = sprintf('MODA v%s - Multiscale Oscillatory Dynamics Analysis', app.AppVersion);
            
            % Execute startup function
            runStartupFcn(app, @startupFcn);
        end
        
        function delete(app)
            % Destructor - cleanup operations when app closes
            % Save app state if needed
            % Clear any persistent data
        end
    end
    
    methods (Access = private)
        
        function createComponents(app)
            % Create main figure
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [100 100 800 700];
            app.UIFigure.CloseRequestFcn = createCallbackFcn(app, @UIFigureCloseRequest, true);
            
            % Create main grid layout
            app.MainGridLayout = uigridlayout(app.UIFigure);
            app.MainGridLayout.ColumnWidth = {'1x', '1x', '1x'};
            app.MainGridLayout.RowHeight = {'80px', '1x', '50px'};
            app.MainGridLayout.Padding = [10 10 10 10];
            app.MainGridLayout.RowSpacing = 10;
            app.MainGridLayout.ColumnSpacing = 10;
            
            % ===== TOP SECTION: LOGO =====
            logoPanel = uipanel(app.UIFigure, 'Parent', app.MainGridLayout);
            logoPanel.BackgroundColor = [1 1 1];
            logoPanel.BorderType = 'none';
            logoPanel.Layout.Row = 1;
            logoPanel.Layout.Column = [1 3];
            
            app.LogoAxes = uiaxes(logoPanel);
            app.LogoAxes.Position = [0 0 logoPanel.Position(3) logoPanel.Position(4)];
            app.LogoAxes.XLim = [0 1];
            app.LogoAxes.YLim = [0 1];
            axis(app.LogoAxes, 'off');
            
            % Try to load banner image
            try
                imgPath = which('frontbanner.png');
                if ~isempty(imgPath)
                    img = imread(imgPath);
                    image(app.LogoAxes, img);
                else
                    % Fallback: display text
                    text(app.LogoAxes, 0.5, 0.5, 'MODA v2.0', ...
                        'HorizontalAlignment', 'center', ...
                        'VerticalAlignment', 'middle', ...
                        'FontSize', 28, ...
                        'FontWeight', 'bold', ...
                        'Color', [0.2 0.4 0.7]);
                end
            catch
                text(app.LogoAxes, 0.5, 0.5, 'MODA v2.0', ...
                    'HorizontalAlignment', 'center', ...
                    'VerticalAlignment', 'middle', ...
                    'FontSize', 28, ...
                    'FontWeight', 'bold', ...
                    'Color', [0.2 0.4 0.7]);
            end
            
            % ===== MIDDLE SECTION: ANALYSIS BUTTONS =====
            buttonPanel = uipanel(app.UIFigure, 'Parent', app.MainGridLayout);
            buttonPanel.Title = 'Select Analysis Module';
            buttonPanel.Layout.Row = 2;
            buttonPanel.Layout.Column = [1 3];
            
            buttonLayout = uigridlayout(buttonPanel);
            buttonLayout.ColumnWidth = {'1x', '1x'};
            buttonLayout.RowHeight = repmat({'1x'}, 1, 3);
            buttonLayout.Padding = [10 10 10 10];
            buttonLayout.RowSpacing = 8;
            buttonLayout.ColumnSpacing = 8;
            
            % Time-Frequency Analysis Button
            app.TimeFrequencyButton = uibutton(buttonPanel, 'push');
            app.TimeFrequencyButton.Layout.Row = 1;
            app.TimeFrequencyButton.Layout.Column = 1;
            app.TimeFrequencyButton.Text = 'Time-Frequency Analysis';
            app.TimeFrequencyButton.FontSize = 14;
            app.TimeFrequencyButton.FontWeight = 'bold';
            app.TimeFrequencyButton.BackgroundColor = [0.2 0.6 0.9];
            app.TimeFrequencyButton.FontColor = [1 1 1];
            app.TimeFrequencyButton.ButtonPushedFcn = createCallbackFcn(app, @TimeFrequencyPushed, true);
            
            % Coherence Analysis Button
            app.CoherenceButton = uibutton(buttonPanel, 'push');
            app.CoherenceButton.Layout.Row = 1;
            app.CoherenceButton.Layout.Column = 2;
            app.CoherenceButton.Text = 'Wavelet Phase Coherence';
            app.CoherenceButton.FontSize = 14;
            app.CoherenceButton.FontWeight = 'bold';
            app.CoherenceButton.BackgroundColor = [0.2 0.6 0.9];
            app.CoherenceButton.FontColor = [1 1 1];
            app.CoherenceButton.ButtonPushedFcn = createCallbackFcn(app, @CoherencePushed, true);
            
            % Filtering Button
            app.FilteringButton = uibutton(buttonPanel, 'push');
            app.FilteringButton.Layout.Row = 2;
            app.FilteringButton.Layout.Column = 1;
            app.FilteringButton.Text = 'Signal Filtering';
            app.FilteringButton.FontSize = 14;
            app.FilteringButton.FontWeight = 'bold';
            app.FilteringButton.BackgroundColor = [0.2 0.6 0.9];
            app.FilteringButton.FontColor = [1 1 1];
            app.FilteringButton.ButtonPushedFcn = createCallbackFcn(app, @FilteringPushed, true);
            
            % Bispectral Analysis Button
            app.BispectralButton = uibutton(buttonPanel, 'push');
            app.BispectralButton.Layout.Row = 2;
            app.BispectralButton.Layout.Column = 2;
            app.BispectralButton.Text = 'Bispectral Analysis';
            app.BispectralButton.FontSize = 14;
            app.BispectralButton.FontWeight = 'bold';
            app.BispectralButton.BackgroundColor = [0.2 0.6 0.9];
            app.BispectralButton.FontColor = [1 1 1];
            app.BispectralButton.ButtonPushedFcn = createCallbackFcn(app, @BispectralPushed, true);
            
            % Bayesian Inference Button
            app.BayesianButton = uibutton(buttonPanel, 'push');
            app.BayesianButton.Layout.Row = 3;
            app.BayesianButton.Layout.Column = 1;
            app.BayesianButton.Text = 'Bayesian Inference';
            app.BayesianButton.FontSize = 14;
            app.BayesianButton.FontWeight = 'bold';
            app.BayesianButton.BackgroundColor = [0.2 0.6 0.9];
            app.BayesianButton.FontColor = [1 1 1];
            app.BayesianButton.ButtonPushedFcn = createCallbackFcn(app, @BayesianPushed, true);
            
            % Exit Button
            app.ExitButton = uibutton(buttonPanel, 'push');
            app.ExitButton.Layout.Row = 3;
            app.ExitButton.Layout.Column = 2;
            app.ExitButton.Text = 'Exit';
            app.ExitButton.FontSize = 14;
            app.ExitButton.FontWeight = 'bold';
            app.ExitButton.BackgroundColor = [0.8 0.3 0.3];
            app.ExitButton.FontColor = [1 1 1];
            app.ExitButton.ButtonPushedFcn = createCallbackFcn(app, @ExitPushed, true);
            
            % ===== BOTTOM SECTION: STATUS BAR =====
            app.StatusBar = uilabel(app.UIFigure, 'Parent', app.MainGridLayout);
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
                if ~any(strfind(char(installedNames), requiredToolboxes{i}))
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
        
        % ===== BUTTON CALLBACKS =====
        
        function TimeFrequencyPushed(app, event)
            % Time-Frequency Analysis button callback
            try
                app.updateStatus('Launching Time-Frequency Analysis...');
                TimeFrequencyAnalysis();
                app.updateStatus('Time-Frequency Analysis closed');
            catch ME
                uialert(app.UIFigure, ...
                    sprintf('Error launching Time-Frequency Analysis:\n\n%s', ME.message), ...
                    'Error', 'icon', 'error');
                app.updateStatus('Error - see message above');
            end
        end
        
        function CoherencePushed(app, event)
            % Coherence Analysis button callback
            try
                app.updateStatus('Launching Wavelet Phase Coherence...');
                CoherenceMulti();
                app.updateStatus('Wavelet Phase Coherence closed');
            catch ME
                uialert(app.UIFigure, ...
                    sprintf('Error launching Coherence Analysis:\n\n%s', ME.message), ...
                    'Error', 'icon', 'error');
                app.updateStatus('Error - see message above');
            end
        end
        
        function FilteringPushed(app, event)
            % Signal Filtering button callback
            try
                app.updateStatus('Launching Signal Filtering...');
                Filtering();
                app.updateStatus('Signal Filtering closed');
            catch ME
                uialert(app.UIFigure, ...
                    sprintf('Error launching Signal Filtering:\n\n%s', ME.message), ...
                    'Error', 'icon', 'error');
                app.updateStatus('Error - see message above');
            end
        end
        
        function BispectralPushed(app, event)
            % Bispectral Analysis button callback
            try
                app.updateStatus('Launching Bispectral Analysis...');
                Bispectrum();
                app.updateStatus('Bispectral Analysis closed');
            catch ME
                uialert(app.UIFigure, ...
                    sprintf('Error launching Bispectral Analysis:\n\n%s', ME.message), ...
                    'Error', 'icon', 'error');
                app.updateStatus('Error - see message above');
            end
        end
        
        function BayesianPushed(app, event)
            % Bayesian Inference button callback
            try
                app.updateStatus('Launching Bayesian Inference...');
                Bayesian();
                app.updateStatus('Bayesian Inference closed');
            catch ME
                uialert(app.UIFigure, ...
                    sprintf('Error launching Bayesian Inference:\n\n%s', ME.message), ...
                    'Error', 'icon', 'error');
                app.updateStatus('Error - see message above');
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

% ===== STANDALONE LAUNCH FUNCTION =====
function varargout = MODA(varargin)
    % MODA - Main entry point (maintains backwards compatibility)
    % This function allows the app to be launched with: >> MODA
    
    % Create and run the app
    app = MODAApp(varargin{:});
    
    % Return app handle if requested
    if nargout
        varargout{1} = app;
    end
end
