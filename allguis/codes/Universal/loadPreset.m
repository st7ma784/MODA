function [params, moduleNameSaved, ok] = loadPreset(filepath)
% LOADPRESET  Load an analysis-parameter preset previously written by
% savePreset.m.
%
% INPUT:
% filepath: full path to a .mat file to read (caller is responsible for
%           prompting the user for this, e.g. via uigetfile)
%
% OUTPUT:
% params:          struct of parameter name -> value pairs (empty struct if
%                  the file could not be read or isn't a valid preset)
% moduleNameSaved: the module name the preset was saved from, so the
%                  caller can warn the user if it doesn't match the module
%                  they're loading into (fields may not correspond 1:1)
% ok:              true if a valid preset was loaded, false otherwise

params = struct();
moduleNameSaved = '';
ok = false;

if ~isfile(filepath)
    return;
end
try
    data = load(filepath);
catch
    return;
end
if ~isfield(data,'preset') || ~isfield(data.preset,'params') || ~isfield(data.preset,'moduleName')
    return;
end
params = data.preset.params;
moduleNameSaved = data.preset.moduleName;
ok = true;
end
