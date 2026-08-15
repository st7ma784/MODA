function [plotCanvas, sidebarView] = attachScrollCanvas(parent, designW, designH, sidebarW)
%ATTACHSCROLLCANVAS Split a fixed-pixel module layout into two scroll regions.
%
%   [plotCanvas, sidebarView] = ATTACHSCROLLCANVAS(parent, designW, designH,
%   sidebarW) divides PARENT into an independently scrolling control column
%   on the left (SIDEBARVIEW, sidebarW pixels wide, full height) and the
%   plotting area on the right (PLOTCANVAS).
%
%   Every MODA module lays its components out in absolute pixels on a
%   1600x860 canvas anchored bottom-left, with the control sidebar occupying
%   x = 0..sidebarW and the plot panels x >= sidebarW. In a container shorter
%   than the design — an embedded MODAApp tab, a laptop screen — the *top* of
%   that layout fell off the top edge with no way to reach it.
%
%   Both regions keep the design's TOP edge pinned to the top of the
%   container. PLOTCANVAS is offset by -sidebarW so plot panels keep their
%   original design coordinates unchanged; parent them to PLOTCANVAS exactly
%   as before. The sidebar goes in SIDEBARVIEW and scrolls on its own, so
%   reaching a control at the bottom of the option list no longer drags the
%   graphs off-screen with it. See FITSIDEBARPANEL.

if nargin < 2 || isempty(designW),  designW  = 1600; end
if nargin < 3 || isempty(designH),  designH  = 860;  end
if nargin < 4 || isempty(sidebarW), sidebarW = 330;  end

% Left: control column. Scrolls independently of the plots; its own contents
% are sized to fit by fitSidebarPanel.
sidebarView = uipanel(parent, 'BorderType', 'none', 'Scrollable', 'on', ...
    'Units', 'pixels', 'AutoResizeChildren', 'off');

% Right: plot area, carrying the original fixed-pixel design.
plotView = uipanel(parent, 'BorderType', 'none', 'Scrollable', 'on', ...
    'Units', 'pixels', 'AutoResizeChildren', 'off');

plotCanvas = uipanel(plotView, 'BorderType', 'none', 'Units', 'pixels', ...
    'Position', [-sidebarW, 0, designW, designH]);

setappdata(plotCanvas, 'ScrollCanvasSidebarView', sidebarView);
setappdata(sidebarView, 'ScrollCanvasSidebarWidth', sidebarW);

layoutRegions();
% React to the container (window/tab) changing size. AutoResizeChildren must
% be off first, or MATLAB ignores SizeChangedFcn entirely.
if isprop(parent, 'AutoResizeChildren')
    parent.AutoResizeChildren = 'off';
end
if isprop(parent, 'SizeChangedFcn')
    parent.SizeChangedFcn = @(~,~) layoutRegions();
end

    function layoutRegions()
        if ~isvalid(sidebarView) || ~isvalid(plotView)
            return
        end
        p  = getpixelposition(parent);
        vw = max(1, p(3));
        vh = max(1, p(4));

        sidebarView.Position = [0, 0, min(sidebarW, vw), vh];
        plotView.Position    = [min(sidebarW, vw), 0, max(1, vw - sidebarW), vh];

        % A scrollable container only reaches content that grows UP and to
        % the RIGHT of its origin — anything placed at a negative offset is
        % simply cropped, with no scrollbar to recover it. That is what kept
        % eating the bottom of the option list. So: when the content is
        % taller than the view, sit it at y = 0 and let it overhang upward
        % (scrollable); only when it fits do we lower it to align the tops.
        plotCanvas.Position = [-sidebarW, max(0, vh - designH), designW, designH];

        pinned = getappdata(sidebarView, 'SidebarPanel');
        if ~isempty(pinned) && isvalid(pinned)
            pp = pinned.Position;
            pinned.Position = [pp(1), max(0, vh - pp(4)), pp(3), pp(4)];
        end

        % Start both regions showing the top of the layout, not the bottom.
        scrollToTop(plotView);
        scrollToTop(sidebarView);
    end
end

function scrollToTop(container)
% Show the top of the scrollable area. Wrapped because scroll() errors if
% the container has nothing to scroll yet (e.g. during construction).
try
    scroll(container, 'top');
catch
    % Nothing scrollable yet — the next resize will settle it.
end
end
