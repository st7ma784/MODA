function handles=MODAbayes_loadfilt(hObject,eventdata,handles,ty)

% ty=1 - load filtered phases, 1 sig, 2 bands
% ty=2 - load filtered phases, 2 sigs, 1 band
handles.pinput=1;
setVisible(handles.curr_time,'off');

linkaxes([handles.time_series_1 handles.time_series_2],'x');
setStr(handles.status,'Importing Signal...');    

if ty==1
    [filename,pathname] = uigetfile('*.mat','Load phases');
    name = fullfile(pathname,filename);
    sig = load(name);    
    sig = struct2cell(sig);
    sig = cell2mat(sig);
    
    if isfield(sig,'Ridge_recon')
        phiall=sig.Ridge_phase;
    else
        phiall=sig.Filtered_phases;
    end
    
    S=size(phiall,1);

    for j=1:S
        phi1(j,:)=phiall{j,1};
        phi2(j,:)=phiall{j,2};
    end
    
    handles.int1=sig.Freq_bands{1,1};
    handles.int2=sig.Freq_bands{2,1};
    handles.sampling_freq=sig.Sampling_frequency;
    
    
elseif ty==2
    [filename,pathname] = uigetfile('*.mat','Load phase 1');
    name = fullfile(pathname,filename);
    sig1 = load(name);    
            
    if isfield(sig1.Filtered_data,'Ridge_recon')
        phi1=cell2mat(sig1.Filtered_data.Ridge_phase);    
    else
        phi1=cell2mat(sig1.Filtered_data.Filtered_phases);    
    end
    
    [filename,pathname] = uigetfile('*.mat','Load phase 2');
    name = fullfile(pathname,filename);
    sig2 = load(name);   
    
    if isfield(sig2.Filtered_data,'Ridge_recon')
        phi2=cell2mat(sig2.Filtered_data.Ridge_phase);    
    else
        phi2=cell2mat(sig2.Filtered_data.Filtered_phases);    
    end
    
    handles.int1=sig1.Filtered_data.Freq_bands{1,1};
    handles.int2=sig2.Filtered_data.Freq_bands{1,1};
    handles.sampling_freq=sig1.Filtered_data.Sampling_frequency;
    
    
    if sig1.Filtered_data.Sampling_frequency~=sig2.Filtered_data.Sampling_frequency
        errordlg('Data sets must have the same sampling frequency','Parameter Error');
      return;
    else
    end
    
end

handles.sig=[phi1;phi2];
handles.sig_cut=handles.sig;
handles.phase1=phi1;
handles.phase2=phi2;
setStr(handles.freq_1,handles.int1);
setStr(handles.freq_2,handles.int2);

time = linspace(0,length(handles.sig)/handles.sampling_freq,length(handles.sig));
handles.time_axis = time;
handles.time_axis_cut = time;

%% Create signal list
list = cell(size(handles.sig,1)/2,1);
list{1,1} = 'Signal Pair 1';
    
for i = 2:size(handles.sig,1)/2
    list{i,1} = sprintf('Signal Pair %d',i);
end
setListItems(handles.signal_list,list);
    
%% Plot time series
linkaxes([handles.time_series_1 handles.time_series_2],'x'); % Ensures axis limits are identical for both plots
plot(handles.time_series_1,handles.time_axis,handles.sig(1,:),'color',handles.linecol(1,:));
xlim(handles.time_series_1,[0 handles.time_axis(end)]);
plot(handles.time_series_2,handles.time_axis,handles.sig(1+size(handles.sig,1)/2,:),'color',handles.linecol(1,:));
xlim(handles.time_series_2,[0 handles.time_axis(end)]);
xlabel(handles.time_series_2,'Time (s)');
ylabel(handles.time_series_1,'Sig 1');
ylabel(handles.time_series_2,'Sig 2');

setEnable(handles.plot_TS,'on');

setStr(handles.status,'Select data and define parameters');

end % MODAbayes_loadfilt

% ---- UI-agnostic helpers ----------------------------------------
function setStr(h, s)
    try; set(h,'String',s); catch; h.Value = s; end
end

function setEnable(h, s)
    try; set(h,'Enable',s); catch; h.Enable = s; end
end

function setVisible(h, s)
    try; set(h,'Visible',s); catch; h.Visible = s; end
end

function setListItems(h, items)
    try; set(h,'String',items); catch; h.Items = items; end
end
