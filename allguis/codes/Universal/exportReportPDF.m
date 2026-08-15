function ok = exportReportPDF(filepath, viewFig, moduleName, paramStruct)
% EXPORTREPORTPDF  Bundle a module's current results plot and the
% parameters used to produce it into a single one-page PDF report.
%
% This is the "one-click export report" companion to the existing
% per-plot Export/Open View menu items: those save just the current plot;
% this additionally captures which parameter values produced it, so the
% output is self-documenting (useful for write-ups, sharing, or coming
% back to a result later without a saved session file).
%
% INPUT:
% filepath:    destination file path (should end in .pdf)
% viewFig:     a figure handle showing the plot to include (e.g. as
%              returned by a module's buildViewFigure()); NOT closed by
%              this function — the caller owns its lifecycle
% moduleName:  string used in the report title (e.g. 'Time-Frequency Analysis')
% paramStruct: struct of parameter name -> value pairs to list in the report
%
% OUTPUT:
% ok: true if the PDF was written successfully, false otherwise

ok = false;
reportFig = [];
try
    reportFig = figure('Visible', 'off', 'Units', 'inches', ...
        'Position', [1 1 8.5 11], 'PaperUnits', 'inches', ...
        'PaperSize', [8.5 11], 'PaperPosition', [0 0 8.5 11], 'Color', 'w');

    annotation(reportFig, 'textbox', [0.05 0.94 0.9 0.05], ...
        'String', sprintf('MODA Report \x2014 %s', moduleName), ...
        'FontSize', 16, 'FontWeight', 'bold', 'EdgeColor', 'none', ...
        'HorizontalAlignment', 'center');
    annotation(reportFig, 'textbox', [0.05 0.905 0.9 0.03], ...
        'String', ['Generated: ', datestr(now)], ... %#ok<TNOW1,DATST>
        'FontSize', 9, 'EdgeColor', 'none', 'HorizontalAlignment', 'center', ...
        'Color', [0.45 0.45 0.45]);

    % Capture the source figure as a bitmap and display it in the report —
    % robust to any plot layout (single axes, multiple axes, colorbars,
    % legends...) without needing to know its internal structure, unlike
    % copyobj-ing individual axes.
    axSnapshot = axes(reportFig, 'Position', [0.06 0.40 0.88 0.49]);
    frame = getframe(viewFig);
    imshow(frame.cdata, 'Parent', axSnapshot);
    axSnapshot.Visible = 'off';

    fieldNamesList = fieldnames(paramStruct);
    lines = cell(numel(fieldNamesList), 1);
    for k = 1:numel(fieldNamesList)
        v = paramStruct.(fieldNamesList{k});
        if isnumeric(v) || islogical(v)
            v = mat2str(v);
        elseif ~ischar(v) && ~isstring(v)
            v = class(v);
        end
        lines{k} = sprintf('%s:  %s', fieldNamesList{k}, char(v));
    end
    paramText = strjoin([{'Parameters used:', ''}, lines(:)'], newline);
    annotation(reportFig, 'textbox', [0.08 0.04 0.84 0.33], ...
        'String', paramText, 'FontSize', 10, 'FontName', 'FixedWidth', ...
        'EdgeColor', [0.75 0.75 0.75], 'VerticalAlignment', 'top', ...
        'Interpreter', 'none');

    exportgraphics(reportFig, filepath, 'ContentType', 'vector');
    ok = true;
catch
    ok = false;
end
if ~isempty(reportFig) && isvalid(reportFig)
    close(reportFig);
end
end
