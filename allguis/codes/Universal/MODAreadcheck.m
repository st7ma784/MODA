% MODA data loading check function
%
% Confirms before overwriting already-loaded data. [it] is a load counter:
% >1 means data is already present, so ask before erasing it. Shared by every
% module, so [handles] may be a classic GUIDE struct (any field can be added)
% or an App Designer app object (only *declared* properties can be read/set).
% A module that doesn't declare [it] can't persist the counter, so treat it as a
% first load (allow, no prompt) rather than throwing "Unrecognized field it".

function [handles,A]=MODAreadcheck(handles)

hasIt = (isstruct(handles) && isfield(handles,'it')) || ...
        (~isstruct(handles) && isprop(handles,'it'));

if hasIt
    if isempty(handles.it), handles.it = 0; end
    handles.it = handles.it + 1;
    count = handles.it;
else
    count = 1;   % no counter available → behave as the first load
end

A=0;
if count>1
   choice = questdlg('Loading new data will erase unsaved data. Continue?', ...
        'Data Import','Yes','No','No');
   switch choice
       case 'Yes'
           A=1;
       case 'No'
           A=0;
   end
else
    A=1;
end
