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
