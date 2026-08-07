% MODA GUI settings function
% Compatible with MATLAB R2023a through R2026a.

function handles = MODAsettings(hObject, handles)

% Positioning — only applies when called with an explicit figure handle
% (legacy GUIDE callers). App Designer callers pass hObject=[] and already
% size/position their own uifigure in createComponents; falling back to
% gcf here would resize/create the wrong (or a stray) figure.
if ~isempty(hObject) && ishandle(hObject)
    screensize = get(groot, 'Screensize');
    x = screensize(3);
    y = screensize(4);
    if x < 1600 || y < 860
        GUIsize = [x y];
        set(hObject, 'units', 'pixels', 'position', [0 0 GUIsize(1) GUIsize(2)]);
    else
        GUIsize = [1600 860];
        x2 = x - GUIsize(1);
        y2 = y - GUIsize(2);
        set(hObject, 'units', 'pixels', 'position', [x2/2 y2/2 GUIsize(1) GUIsize(2)]);
    end
end

% Colours
load('cmap.mat');
handles.cmap = cmap;
handles.linecol = cmap([1,18,40,50,60,64,15],:);
handles.line2width = 2;

% Logos — guarded so the function works when logo axes are absent.
% Target the uiaxes handle directly (image(ax, C)); legacy axes()/gca-style
% targeting doesn't work reliably against App Designer uiaxes and causes the
% image to be drawn into a brand-new standalone figure instead.
if isfield(handles, 'logo') && ishandle(handles.logo)
    try
        matlabImage = imread('physicslogo.png');
        image(handles.logo, matlabImage);
        axis(handles.logo, 'off');
        axis(handles.logo, 'image');
    catch
    end
end

if isfield(handles, 'nbmplogo') && ishandle(handles.nbmplogo)
    try
        matlabImage = imread('MODAbanner5.png');
        image(handles.nbmplogo, matlabImage);
        axis(handles.nbmplogo, 'off');
        axis(handles.nbmplogo, 'image');
    catch
    end
end

% Fonts
h = findall(0, 'Type', 'uicontrol');
set(h, 'FontUnits', 'points');
set(h, 'FontSize', 8);
set(h, 'FontUnits', 'normalized');

% Default calculation types
handles.calc_type = 1;
handles.plot_type = 2;

handles.it = 0;
