function ok = savePreset(filepath, moduleName, paramStruct)
% SAVEPRESET  Save an analysis-parameter preset to a .mat file.
%
% A preset is just a snapshot of a module's current parameter field values
% (e.g. fmin/fmax/wavelet type), NOT the loaded signal or any computed
% results — see loadPreset.m for the counterpart, and each module's
% save/loadPresetButtonPushed callback for what fields are actually saved.
%
% INPUT:
% filepath:    full path to write to (caller is responsible for prompting
%              the user for this, e.g. via uiputfile)
% moduleName:  string identifying which module saved this preset (e.g.
%              'TimeFrequencyAnalysis'), stored so loadPreset can warn if
%              a preset is applied to a different module than it came from
% paramStruct: struct of parameter name -> value pairs to save
%
% OUTPUT:
% ok: true if the file was written successfully, false otherwise

ok = false;
try
    preset = struct('moduleName', moduleName, 'params', paramStruct, 'savedAt', datestr(now)); %#ok<NASGU,TNOW1,DATST>
    save(filepath, 'preset');
    ok = true;
catch
    ok = false;
end
end
