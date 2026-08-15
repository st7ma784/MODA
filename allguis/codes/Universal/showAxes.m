function showAxes(ax, on)
%SHOWAXES Show or hide an axes together with everything drawn in it.
%
%   SHOWAXES(ax, true|false). Setting only an axes' Visible to off hides
%   its box and ticks, but its contents, colorbar and legend still draw, so
%   a hidden view showed through the one laid over it.

v = matlab.lang.OnOffSwitchState(on);
ax.Visible = v;
set(ax.Children, 'Visible', v);
if ~isempty(ax.Colorbar), ax.Colorbar.Visible = v; end
if ~isempty(ax.Legend),   ax.Legend.Visible   = v; end
end
