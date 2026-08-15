function placeModuleWindow(fig, W, H)
%PLACEMODULEWINDOW Size a module's own window to fit the screen.
%
%   PLACEMODULEWINDOW(fig, W, H) centres FIG at its W x H design size when
%   the screen has room for it, and maximises it otherwise. Maximising lets
%   the operating system leave room for the title bar, menu bar and
%   taskbar; a window sized to the full ScreenSize instead ran off the edge
%   of a laptop screen and cut off the top or bottom of the module.
%
%   The window stays resizable: the layout scrolls to fit any size (see
%   ATTACHSCROLLCANVAS).

fig.Resize = 'on';
ss = get(groot, 'ScreenSize');
% ScreenSize includes the taskbar, and Position excludes the title bar.
chrome = 100;
if ss(3) < W || ss(4) < H + chrome
    % The size to restore to if the user un-maximises it.
    fig.Position = [ss(1), ss(2), min(W, ss(3)), min(H, ss(4) - chrome)];
    fig.WindowState = 'maximized';
else
    fig.Position = [ss(1) + round((ss(3) - W)/2), ss(2) + round((ss(4) - H)/2), W, H];
end
end
