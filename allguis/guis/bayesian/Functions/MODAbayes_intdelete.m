function handles = MODAbayes_intdelete(hObject, eventdata, handles)
% Delete the selected interval row from both interval lists and all
% associated data arrays.
% Compatible with MATLAB App Designer (classdef) and legacy GUIDE handles.

% ---- UI-agnostic helpers ----------------------------------------
    function idx = listIdx(h)
        % Return 1-based numeric selection index regardless of UI type.
        try
            idx = get(h,'Value');
            if ~isnumeric(idx); error('not numeric'); end
        catch
            items = h.Items;
            val   = h.Value;
            if ischar(val) || isstring(val); val = {char(val)}; end
            idx = find(strcmp(items, val{1}), 1);
            if isempty(idx); idx = 1; end
        end
    end

    function items = listItems(h)
        try
            items = get(h,'String');
            if ischar(items); items = {items}; end
        catch
            items = h.Items;
        end
    end

    function setListByIdx(h, idx)
        try
            set(h,'Value', idx);   % GUIDE
        catch
            items = h.Items;
            if idx >= 1 && idx <= numel(items)
                h.Value = items{idx};
            end
        end
    end

    function setListItems(h, items)
        try
            set(h,'String', items);   % GUIDE
        catch
            h.Items = items;          % App Designer
        end
    end

    function tf = hasField(s, name)
        % Works for both structs (isfield) and objects (isprop).
        try
            tf = isfield(s, name);
        catch
            try
                tf = isprop(s, name);
            catch
                tf = false;
            end
        end
    end
% ---- end helpers -------------------------------------------------

interval_selected = listIdx(handles.interval_list_1);
n = 1:handles.c;

if isempty(interval_selected); return; end

ne = n(1:end ~= interval_selected);

if ~hasField(handles,'pinput')
    handles.int1 = handles.int1(ne,:);
    handles.int2 = handles.int2(ne,:);
end
handles.winds            = handles.winds(ne);
handles.pr               = handles.pr(ne);
handles.ovr              = handles.ovr(ne);
handles.forder           = handles.forder(ne);
handles.ns               = handles.ns(ne);
handles.confidence_level = handles.confidence_level(ne);

if hasField(handles,'tm') && size(handles.tm,2) == handles.c
    handles.tm        = handles.tm(:,ne);
    handles.cc        = handles.cc(:,ne);
    handles.e         = handles.e(:,ne);
    handles.cpl1      = handles.cpl1(:,ne);
    handles.cpl2      = handles.cpl2(:,ne);
    handles.cf1       = handles.cf1(:,ne);
    handles.cf2       = handles.cf2(:,ne);
    handles.mcf1      = handles.mcf1(:,ne);
    handles.mcf2      = handles.mcf2(:,ne);
    handles.surr_cpl1 = handles.surr_cpl1(:,ne);
    handles.surr_cpl2 = handles.surr_cpl2(:,ne);
    handles.p1        = handles.p1(:,ne);
    handles.p2        = handles.p2(:,ne);
end

% Update interval_list_1
new_idx = max(1, interval_selected - 1);
setListByIdx(handles.interval_list_1, new_idx);
list1 = listItems(handles.interval_list_1);
list1(interval_selected) = [];
setListItems(handles.interval_list_1, list1);

% Update interval_list_2 (same index)
interval_selected2 = listIdx(handles.interval_list_2);
new_idx2 = max(1, interval_selected2 - 1);
setListByIdx(handles.interval_list_2, new_idx2);
list2 = listItems(handles.interval_list_2);
list2(interval_selected) = [];
setListItems(handles.interval_list_2, list2);

if hasField(handles,'bands') && ~isempty(handles.bands)
    try
        handles.bands(:, interval_selected) = [];
    catch
        % Ignore out-of-range deletion errors.
    end
end

drawnow;
handles.c = handles.c - 1;
try; guidata(hObject, handles); catch; end

end
