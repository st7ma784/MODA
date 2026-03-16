# MODA Refactor Guide: GUIDE to App Designer Migration

**Version:** 1.0  
**Date:** March 5, 2026  
**Target Platform:** MATLAB R2023a through R2024b  
**Status:** Ready for Developer Implementation

---

## Table of Contents

1. [Overview & Getting Started](#overview)
2. [Architecture Pattern](#architecture)
3. [Step-by-Step Refactor Process](#step-by-step)
4. [File-by-File Implementation](#file-by-file)
5. [Data I/O Modernization](#data-io)
6. [Testing & Validation](#testing)
7. [Common Pitfalls & Solutions](#pitfalls)
8. [Troubleshooting](#troubleshooting)

---

## Overview & Getting Started {#overview}

### What Has Changed

MODA.m has been successfully refactored from GUIDE (v2.5, 2017) to App Designer (2026). This guide covers refactoring the remaining 5 analysis modules and their supporting functions.

### Modules to Refactor

| Module | Current Status | Priority | Effort |
|--------|---|---|---|
| TimeFrequencyAnalysis.m | GUIDE + csvwrite | 🔴 High | 40h |
| CoherenceMulti.m | GUIDE + csvread | 🔴 High | 35h |
| Filtering.m | GUIDE | 🔴 High | 30h |
| Bispectrum.m | GUIDE | 🔴 High | 25h |
| Bayesian.m | GUIDE | 🔴 High | 30h |
| Supporting functions | Deprecated functions | 🟠 Medium | 30h |
| **TOTAL** | | | **~190 hours** |

### Prerequisites

- MATLAB R2023a or later (tested through R2024b)
- All three required toolboxes installed
- Git or version control for tracking changes
- Understanding of GUIDE vs App Designer differences

### Setup Instructions

```bash
# 1. Create backup branch
git checkout -b feature/app-designer-migration

# 2. Create subdirectory for refactored GUIs
mkdir -p allguis/guis_refactored

# 3. Verify MATLAB path
% In MATLAB Command Window:
addpath(genpath(pwd))
```

---

## Architecture Pattern {#architecture}

### GUIDE vs App Designer

#### Old Pattern (GUIDE - Do NOT use)

```matlab
function varargout = TimeFrequencyAnalysis(varargin)
    gui_Singleton = 1;
    gui_State = struct('gui_Name', mfilename, ...);
    gui_mainfcn(gui_State, varargin{:});
end

function TimeFrequencyAnalysis_OpeningFcn(hObject, eventdata, handles, varargin)
    handles.output = hObject;
    set(handles.plot_TS, 'Enable', 'off');
    guidata(hObject, handles);
end
```

#### New Pattern (App Designer - USE THIS)

```matlab
classdef TimeFrequencyAnalysisApp < matlab.apps.AppBase
    % Props declared in classdef
    properties (Access = public)
        UIFigure matlab.ui.Figure
        PlotTS matlab.ui.control.UIAxes
    end
    
    properties (Access = private)
        % Private data
        SignalData (:,:) double
    end
    
    methods (Access = public)
        function app = TimeFrequencyAnalysisApp(varargin)
            % Constructor
            createComponents(app);
            registerAppComponents(app);
            runStartupFcn(app, @startupFcn);
        end
    end
    
    methods (Access = private)
        function startupFcn(app)
            % Called when app launches
            set(app.PlotTS, 'Enable', 'off');
        end
    end
end
```

### Key Differences

| Aspect | GUIDE | App Designer |
|--------|-------|---|
| **File Type** | .fig + .m | .mlapp or classdef .m |
| **Data Storage** | handles structure | class properties |
| **Callbacks** | String names + function pointer | Method references |
| **Layout** | Manual positioning | Grid/Flow/Absolute layout manager |
| **Component Access** | guidata(), guihandles() | Direct property access (app.ComponentName) |
| **Initialization** | OpeningFcn callback | startupFcn method |
| **Execution** | gui_mainfcn() | Direct instantiation (app = ClassName()) |

---

## Step-by-Step Refactor Process {#step-by-step}

### Phase 1: Planning & Setup (2 hours)

#### 1.1 Before You Start

```matlab
% Checklist for each module to refactor:
% [ ] Backup original GUIDE file
% [ ] Identify all callbacks (Search for "_Callback")
% [ ] List all handles used (handles.component_name)
% [ ] Document data flow between functions
% [ ] Identify deprecated functions (csvread, csvwrite, strfind)
% [ ] Plan property structure
```

#### 1.2 Analysis Template

Create a file `REFACTOR_ANALYSIS.md`:

```markdown
# TimeFrequencyAnalysis Refactor Analysis

## Original GUI Structure
- Main figure with 10 axes
- 15 pushbuttons for analysis
- 20 callback functions
- Uses handles for state management

## Deprecated Functions Found
- csvwrite() - line 827 (write export)
- strfind() - line 665 (string search)
- uipickfiles() - custom dialog

## Properties Needed (App Designer)
- SignalData (signal array)
- SamplingFrequency (numeric)
- SelectedSignal (index)
- AnalysisResults (struct)

## Methods to Create
- startupFcn() - initialization
- loadSignalCallback() - file loading
- plotSignalCallback() - visualization
- analyzeCallback() - core algorithm
- exportCallback() - data export
```

---

### Phase 2: Component Conversion (20-40 hours per module)

#### 2.1 Create Skeleton

For each module, create a new .m file with this structure:

```matlab
classdef TimeFrequencyAnalysisApp < matlab.apps.AppBase
    % TimeFrequencyAnalysisApp (v2.0)
    % Modernized from legacy GUIDE app (Oct 2017)
    %
    % Changelog:
    % - Converted from GUIDE to App Designer (R2023a+)
    % - Replaced csvwrite with writematrix
    % - Replaced strfind with contains
    % - Modernized handle management
    
    properties (Access = public)
        UIFigure matlab.ui.Figure
    end
    
    properties (Access = private)
        % UI Components
        MainGrid matlab.ui.container.GridLayout
        ControlPanel matlab.ui.container.Panel
        VisualizationPanel matlab.ui.container.Panel
        
        % Data
        SignalData (:,:) double = []
        SamplingFrequency (1,1) double = 100
        AnalysisResults struct = struct()
        
        % State
        IsAnalyzing (1,1) logical = false
        CurrentFigure matlab.ui.Figure = []
    end
    
    methods (Access = public)
        function app = TimeFrequencyAnalysisApp(varargin)
            createComponents(app);
            registerAppComponents(app);
            runStartupFcn(app, @startupFcn);
        end
        
        function delete(app)
            % Cleanup
        end
    end
    
    methods (Access = private)
        function createComponents(app)
            % Build UI here
        end
        
        function startupFcn(app)
            % Initialize on launch
        end
    end
end

function varargout = TimeFrequencyAnalysis(varargin)
    % Backwards compatible launcher
    app = TimeFrequencyAnalysisApp(varargin{:});
    if nargout
        varargout{1} = app;
    end
end
```

#### 2.2 List All Components

Extract from original GUIDE .fig or .m:

```matlab
function createComponents(app)
    % 1. Figure
    app.UIFigure = uifigure('Visible', 'off');
    app.UIFigure.Position = [100 100 1200 800];
    app.UIFigure.Name = 'Time-Frequency Analysis v2.0';
    app.UIFigure.CloseRequestFcn = createCallbackFcn(app, @closeFcn, true);
    
    % 2. Main grid layout
    app.MainGrid = uigridlayout(app.UIFigure);
    app.MainGrid.ColumnWidth = {'200px', '1x'};
    app.MainGrid.RowHeight = {'50px', '1x', '50px'};
    
    % 3. Control panel
    app.ControlPanel = uipanel(app.UIFigure);
    app.ControlPanel.Layout.Row = 1;
    app.ControlPanel.Layout.Column = [1 2];
    app.ControlPanel.Title = 'Analysis Controls';
    
    % ... continue for each component
end
```

#### 2.3 Convert Callbacks

For EACH callback in original:

**Before (GUIDE):**
```matlab
function load_signal_Callback(hObject, eventdata, handles)
    [filename, pathname] = uigetfile('*.csv');
    if isequal(filename, 0)
        return;
    end
    data = csvread(fullfile(pathname, filename));
    handles.signal = data;
    guidata(hObject, handles);
    set(handles.status_text, 'String', 'Signal loaded');
end
```

**After (App Designer):**
```matlab
function loadSignalCallback(app, event)
    [filename, pathname] = uigetfile('*.csv');
    if isequal(filename, 0)
        return;
    end
    try
        % Use modern function instead of deprecated csvread
        app.SignalData = readmatrix(fullfile(pathname, filename));
        app.StatusText.Value = 'Signal loaded successfully';
    catch ME
        uialert(app.UIFigure, ...
            sprintf('Error loading file:\n%s', ME.message), ...
            'Load Error', 'icon', 'error');
    end
end
```

---

### Phase 3: Data I/O Modernization (5 hours per module)

#### 3.1 Deprecated Functions Reference

**CSVREAD → READMATRIX**

```matlab
% OLD (REMOVED in R2024a)
M = csvread(filename);

% NEW (Use this)
M = readmatrix(filename);

% If you need table with headers:
T = readtable(filename);

% If you need specific type:
M = readmatrix(filename, 'OutputType', 'double');
```

**CSVWRITE → WRITEMATRIX**

```matlab
% OLD (REMOVED in R2024a)
csvwrite(filename, data);

% NEW (Use this)
writematrix(data, filename);

% If exporting with headers:
T = array2table(data, 'VariableNames', {'Signal', 'Time'});
writetable(T, filename);
```

**STRFIND → CONTAINS**

```matlab
% OLD (DEPRECATED R2022b, works but not recommended)
if ~isempty(strfind(str, 'pattern'))
    % do something
end

% NEW (Use this)
if contains(str, 'pattern')
    % do something
end

% With options:
if contains(str, 'pattern', 'IgnoreCase', true)
    % case-insensitive search
end
```

#### 3.2 Audit All File I/O

Search each module for:
```
grep -r "csvread\|csvwrite\|strfind" allguis/guis/
```

Create replacement table:

| File | Line | Function | Replacement | Priority |
|------|------|----------|-------------|----------|
| read_from_csv.m | 6 | csvread | readmatrix | 🔴 Critical |
| TimeFrequencyAnalysis.m | 827 | csvwrite | writematrix | 🔴 Critical |
| ecurve.m | 593 | strfind | contains | 🟡 Medium |

---

### Phase 4: Algorithm Testing (10 hours per module)

#### 4.1 Unit Test Template

Create `tests/test_TimeFrequencyAnalysis.m`:

```matlab
function tests = test_TimeFrequencyAnalysis
    tests = functiontests(localfunctions);
end

% Test 1: App launches without error
function testAppLaunch(testCase)
    app = TimeFrequencyAnalysisApp();
    testCase.verifyInstanceOf(app, 'TimeFrequencyAnalysisApp');
    testCase.verifyTrue(isvalid(app.UIFigure));
    delete(app);
end

% Test 2: Signal loading works
function testSignalLoad(testCase)
    app = TimeFrequencyAnalysisApp();
    
    % Create test signal
    testSignal = sin(2*pi*(1:1000)/100);
    testFile = tempname;
    writematrix(testSignal, [testFile '.csv']);
    
    % Load it
    [filename, pathname] = fileparts(testFile);
    % Simulate user selection
    data = readmatrix([testFile '.csv']);
    app.SignalData = data;
    
    testCase.verifySize(app.SignalData, [1000 1]);
    
    delete(app);
    delete([testFile '.csv']);
end

% Test 3: Wavelet transform produces expected output
function testWaveletTransform(testCase)
    signal = sin(2*pi*(1:1000)/100);
    fs = 100;
    
    [WT, freq] = wt(signal, fs, 'Display', 'off');
    
    testCase.verifyTrue(size(WT, 1) == length(signal));
    testCase.verifyTrue(all(freq > 0));
    testCase.verifyTrue(~any(isnan(WT(:))));
end
```

Run tests:
```matlab
% In MATLAB command window
runtests('tests/test_TimeFrequencyAnalysis.m')
```

---

### Phase 5: Integration Testing (5 hours per module)

#### 5.1 Full Workflow Test

```matlab
function test_TimeFrequencyAnalysisFullWorkflow
    % End-to-end workflow test
    
    % 1. Launch app
    app = TimeFrequencyAnalysisApp();
    assert(isvalid(app.UIFigure), 'App failed to launch');
    
    % 2. Load sample signal
    t = (0:0.01:10)';
    signal = sin(2*pi*1*t) + 0.5*sin(2*pi*2*t);
    app.SignalData = signal;
    app.SamplingFrequency = 100;
    
    % 3. Run analysis
    % Simulate button press
    feval(app.AnalyzeButton.ButtonPushedFcn, app.AnalyzeButton, []);
    
    % 4. Verify results generated
    assert(~isempty(app.AnalysisResults), 'No analysis results');
    assert(isfield(app.AnalysisResults, 'WT'), 'Missing WT field');
    
    % 5. Test export
    exportFile = tempname;
    % Simulate export button
    feval(app.ExportButton.ButtonPushedFcn, app.ExportButton, []);
    
    cleanup:
    delete(app);
end
```

---

## File-by-File Implementation {#file-by-file}

### Refactor Order (Recommended)

1. **read_from_csv.m** (Simplest - 30 min)
2. **read_from_mat.m** (Simple - 30 min)
3. **Filtering.m** (Medium - 6-8 hours)
4. **Bispectrum.m** (Medium - 6-8 hours)
5. **CoherenceMulti.m** (Medium-Hard - 8-10 hours)
6. **TimeFrequencyAnalysis.m** (Most Complex - 10-12 hours)
7. **Bayesian.m** (Complex - 8-10 hours)

Start with simplest to build confidence!

### Template for Each Module

For `[ModuleName]App.m`:

```matlab
classdef [ModuleName]App < matlab.apps.AppBase
    % [ModuleName]App - [Description]
    %
    % Original GUIDE version: [Date]
    % App Designer refactor: March 2026
    % Compatible with: R2023a, R2023b, R2024a, R2024b
    %
    % Major changes:
    % - Converted from GUIDE to App Designer
    % - [List deprecated function replacements]
    % - [List UI improvements]
    
    properties (Access = public)
        UIFigure matlab.ui.Figure
    end
    
    properties (Access = private)
        % UI Components - [Group by functionality]
        
        % Data - [Document variable types]
        
        % State - [Document app state flags]
    end
    
    methods (Access = public)
        function app = [ModuleName]App(varargin)
            createComponents(app);
            registerAppComponents(app);
            runStartupFcn(app, @startupFcn);
        end
        
        function delete(app)
            % Cleanup on close
        end
    end
    
    methods (Access = private)
        function createComponents(app)
            % Build all UI components here
        end
        
        function startupFcn(app)
            % Initialize on first launch
        end
        
        % Callback methods (one per button/control)
    end
end

% Backwards compatibility launcher
function varargout = [OldModuleName](varargin)
    app = [ModuleName]App(varargin{:});
    if nargout
        varargout{1} = app;
    end
end
```

---

## Data I/O Modernization {#data-io}

### Complete Data I/O Refactor

#### read_from_csv.m → Modern Version

```matlab
% ORIGINAL (BROKEN in R2024a+)
function M = read_from_csv
    [filename, pathname, filterindex] = uigetfile('*.csv');
    if isequal(filename, 0)
        return;
    end
    name = fullfile(pathname, filename);
    M = csvread(name);  % <-- REMOVED in R2024a
end

% MODERN REPLACEMENT
function M = read_from_csv
    % Read CSV file with modern MATLAB approach
    % Compatible with R2023a+
    
    [filename, pathname] = uigetfile('*.csv', 'Select CSV file');
    if isequal(filename, 0)
        return;  % User cancelled
    end
    
    filepath = fullfile(pathname, filename);
    
    % Option 1: Numeric data only (best performance)
    try
        M = readmatrix(filepath);
    catch
        % Option 2: Mixed data types (returns table)
        T = readtable(filepath);
        M = table2array(T);
    end
end

% USAGE IN APP
function loadSignalCallback(app, event)
    M = read_from_csv();
    if ~isempty(M)
        app.SignalData = M;
        app.StatusText.Value = sprintf('Loaded: %d samples', size(M,1));
    end
end
```

#### Export Functions

```matlab
% ORIGINAL (BROKEN)
csvwrite(filename, array);

% MODERN APPROACH 1: Simple numeric export
writematrix(array, filename);

% MODERN APPROACH 2: With headers and formatting
T = array2table(array, ...
    'VariableNames', {'Signal', 'Time'});
writetable(T, filename);

% MODERN APPROACH 3: Custom formatting
opts = detectImportOptions(filename);
opts.VariableNames = {'signal', 'time'};
writetable(array2table(array), filename, opts);
```

### String Handling Modernization

```matlab
% DEPRECATED PATTERNS
if ~isempty(strfind(DispMode, 'plot'))
    % Old code
end

% MODERN EQUIVALENT
if contains(DispMode, 'plot')
    % New code
end

% WITH OPTIONS
if contains(DispMode, 'PLOT', 'IgnoreCase', true)
    % Case-insensitive
end

% MULTIPLE PATTERNS
if contains(DispMode, {'plot', 'show', 'display'})
    % Matches any of these
end
```

---

## Testing & Validation {#testing}

### Test Checklist

For EACH refactored module:

#### Functional Testing

- [ ] App launches without errors
- [ ] All buttons are clickable
- [ ] File dialogs work (Load, Save, Export)
- [ ] Signals load from CSV and MAT files
- [ ] Data export to CSV works
- [ ] All plots render correctly
- [ ] Status messages display properly
- [ ] Error messages display for invalid input
- [ ] App closes cleanly without warnings
- [ ] Memory is properly cleaned up

#### Cross-Version Testing

- [ ] Works on R2023a
- [ ] Works on R2023b
- [ ] Works on R2024a
- [ ] Works on R2024b

#### Algorithm Accuracy

- [ ] Numerical results match R2017a version (>10 decimal places)
- [ ] Plots match original visually
- [ ] Performance is acceptable (<2x slower)
- [ ] Large datasets handled without crashes

#### Automated Test Suite

```matlab
% tests/test_all_modules.m
function tests = test_all_modules
    tests = functiontests(localfunctions);
end

function testAllAppsLaunch(testCase)
    apps = {
        'TimeFrequencyAnalysisApp'
        'CoherenceMultiApp'
        'FilteringApp'
        'BispectralApp'
        'BayesianApp'
    };
    
    for i = 1:length(apps)
        try
            appName = apps{i};
            app = feval(appName);
            testCase.verifyTrue(isvalid(app.UIFigure));
            delete(app);
        catch ME
            testCase.verificationFailed(sprintf('%s failed: %s', appName, ME.message));
        end
    end
end

function testDataIOModernized(testCase)
    % Verify no deprecated functions in refactored code
    requiredFunctions = {
        'read_from_csv'
        'read_from_mat'
    };
    
    for i = 1:length(requiredFunctions)
        fname = requiredFunctions{i};
        code = fileread([fname '.m']);
        
        testCase.verifyEmpty(strfind(code, 'csvread'), ...
            sprintf('%s still contains csvread', fname));
        testCase.verifyEmpty(strfind(code, 'csvwrite'), ...
            sprintf('%s still contains csvwrite', fname));
    end
end
```

Run all tests:
```matlab
runtests('tests/test_all_modules.m', 'Recursively', true)
```

---

## Common Pitfalls & Solutions {#pitfalls}

### Problem: Application crashes on startup

**Symptom:** "Uninitialized 'UIFigure'" or similar

**Solution:**
```matlab
% WRONG - Accessing UIFigure before createComponents
function app = MyApp(varargin)
    app.UIFigure.Name = 'Title';  % <-- ERROR!
    createComponents(app);
end

% RIGHT - Create components FIRST
function app = MyApp(varargin)
    createComponents(app);         % <-- Creates UIFigure
    app.UIFigure.Name = 'Title';   % <-- Now OK
    registerAppComponents(app);
    runStartupFcn(app, @startupFcn);
end
```

### Problem: Updated but values not displayed

**Symptom:** App state changes but UI doesn't update

**Solution:**
```matlab
% After changing app properties, trigger UI update
app.SignalData = newData;  % Change data
drawnow('update');         % Force UI refresh

% Or update specific component
app.StatusText.Value = 'Updated';
app.PlotAxis.Title.String = 'New Title';
```

### Problem: csvwrit/csvread functions still being called

**Symptom:** "Undefined function or variable 'csvread'"

**Solution:** Complete list to find and replace:
```bash
# Find all deprecated functions in codebase
grep -r "csvread\|csvwrite" allguis/

# Replace all instances
sed -i 's/csvread/readmatrix/g' allguis/**/*.m
sed -i 's/csvwrite/writematrix/g' allguis/**/*.m
```

### Problem: Event handlers not being called

**Symptom:** Buttons don't work

**Solution:**
```matlab
% WRONG - String reference (doesn't work in App Designer)
set(button, 'Callback', @button_Callback);

% RIGHT - Use createCallbackFcn
button.ButtonPushedFcn = createCallbackFcn(app, @buttonCallback, true);

% Define callback as method
methods (Access = private)
    function buttonCallback(app, event)
        % Handle button press
    end
end
```

### Problem: Handle structure not found

**Symptom:** "Reference to non-existent field 'handles.signal'"

**Solution:**
```matlab
% OLD (GUIDE)
guidata(hObject, handles);
set(handles.plot_axis, 'Next', 'add');

% NEW (App Designer)
% Use properties instead of handles structure
app.signal = [];           % Declare in properties
app.PlotAxis.NextPlot = 'add';  % Direct component access
```

---

## Troubleshooting {#troubleshooting}

### Debugging Tips

```matlab
% Add debugging to callback
function buttonCallback(app, event)
    try
        % Your code here
        disp('Button pressed successfully');
    catch ME
        % Show error in UI
        uialert(app.UIFigure, ...
            sprintf('Error:\n%s', ME.message), ...
            'Error', 'icon', 'error');
        
        % Also log to command window for debugging
        rethrow(ME);
    end
end

% Check app state during execution
function debugAppState(app)
    disp('=== APP STATE ===');
    disp(['Signal size: ', mat2str(size(app.SignalData))]);
    disp(['Sampling rate: ', num2str(app.SamplingFrequency)]);
    disp(['Analysis results: ', mat2str(fieldnames(app.AnalysisResults)')]);
    disp('==================');
end
```

### Common MATLAB Version Issues

```matlab
% Issue: Feature not available in R2023a
% Solution: Check version and use fallback

if verLessThan('matlab', '24.1')  % Before R2024b
    % Use older function/approach
    result = oldFunction(data);
else
    % Use newer, better function
    result = newFunction(data);
end
```

### Testing on Multiple Versions

```bash
# Automated testing script (bash)
#!/bin/bash
for version in "R2023a" "R2023b" "R2024a" "R2024b"; do
    echo "Testing on $version..."
    "/Applications/MATLAB_$version/bin/matlab" -batch ...
        "addpath(pwd); runtests; exit"
done
```

---

## Rollout Plan

### Phase 1: Completion (Target: Week 3-4)

- [ ] All 5 sub-modules converted to App Designer
- [ ] All deprecated functions replaced
- [ ] All modules tested and validated
- [ ] Documentation updated

### Phase 2: Beta Release (Target: Week 5)

- [ ] Release as v2.0-beta
- [ ] Collect user feedback
- [ ] Fix issues found
- [ ] Performance optimization

### Phase 3: Official Release (Target: Week 6)

- [ ] Release MODA v2.0
- [ ] Publish on GitHub
- [ ] Update user documentation
- [ ] Announce to community

---

## Verification Checklist

- [ ] No GUIDE files remain (no .fig files)
- [ ] No calls to gui_mainfcn(), guidata(), guihandles()
- [ ] No csvread() function calls
- [ ] No csvwrite() function calls
- [ ] All apps launch with >> AppName
- [ ] All algorithms produce identical numerical results
- [ ] All tests pass (runtests)
- [ ] All modules work on R2023a through R2024b
- [ ] Documentation updated
- [ ] Version bumped to 2.0

---

## Success Criteria

✅ **Complete when:**
1. All 5 modules are App Designer-based
2. 100% test pass rate
3. Numerical results agree to 10+ digits with v1.5
4. Documentation is comprehensive
5. Code review is complete
6. Version is tagged as v2.0

---

**For questions or issues, refer to MATLAB_VERSION_REVIEW.md or contact the development team.**