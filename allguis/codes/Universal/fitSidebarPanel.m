function fitSidebarPanel(panel, pad)
%FITSIDEBARPANEL Size a control panel to its contents and pin it to the top.
%
%   FITSIDEBARPANEL(panel) grows PANEL so every control it holds fits inside
%   it, then anchors its top edge to the top of the scrolling sidebar column
%   created by ATTACHSCROLLCANVAS. The column scrolls to reach whatever
%   doesn't fit on screen.
%
%   The modules lay their controls out downward from a nominal 795px-tall
%   panel, but the option lists have outgrown that — in Time-Frequency the
%   lowest control sits ~790px *below* the panel's own bottom edge. Those
%   controls were simply unreachable. Measuring the real content extent, and
%   letting the column scroll, is what keeps them reachable without moving
%   the plots.

if nargin < 2 || isempty(pad)
    pad = 8;
end

kids = panel.Children;
kids = kids(arrayfun(@(k) isprop(k, 'Position'), kids));
if isempty(kids)
    return
end

% Children are a heterogeneous handle array (buttons, panels, axes...), so
% gather positions element-wise rather than concatenating properties.
pos = zeros(numel(kids), 4);
for k = 1:numel(kids)
    pos(k,:) = kids(k).Position;
end
minY   = min(pos(:,2));
maxY   = max(pos(:,2) + pos(:,4));
maxX   = max(pos(:,1) + pos(:,3));

% Shift the contents so the lowest control sits `pad` above the panel floor,
% then make the panel exactly tall enough to hold them all.
shift = pad - minY;
if shift ~= 0
    for k = 1:numel(kids)
        cp = kids(k).Position;
        kids(k).Position = [cp(1), cp(2) + shift, cp(3), cp(4)];
    end
end

newH = (maxY - minY) + 2*pad;
newW = panel.Position(3);

sidebarView = panel.Parent;
vh = getpixelposition(sidebarView);
vh = vh(4);

% Width follows the column so the panel never causes sideways scrolling.
storedW = getappdata(sidebarView, 'ScrollCanvasSidebarWidth');
if ~isempty(storedW)
    newW = storedW;
end
% Only widen past the column when a control genuinely overhangs it —
% otherwise a few stray pixels would add a pointless horizontal scrollbar.
if maxX > newW
    newW = maxX + pad;
end

% Never place the panel at a negative offset: a scrollable container cannot
% scroll to content below its origin — that content is just cropped, which is
% exactly how options kept disappearing off the bottom. Overhang upward
% instead (which IS scrollable), and let attachScrollCanvas scroll to the top.
panel.Position = [0, max(0, vh - newH), newW, newH];

% Registered so attachScrollCanvas can re-place it on every resize.
setappdata(sidebarView, 'SidebarPanel', panel);

try
    scroll(sidebarView, 'top');
catch
    % Nothing to scroll (content fits) — nothing to do.
end
end
