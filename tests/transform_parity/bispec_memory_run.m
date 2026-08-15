function bispec_memory_run(repo, baseDir, side, minutes)
% bispec_memory_run  Peak memory and run time of bispecWavNew on a long recording.
%
%   bispec_memory_run(repo, baseDir, side)            % 20 minutes at 40 Hz
%   bispec_memory_run(repo, baseDir, side, minutes)
%
% side  'new'  - the current bispecWavNew (one wtAtf2_batch call per row)
%       'base' - upstream's bispecWavNew (one wtAtf2 call per cell)
%
% Run each side in its own MATLAB process: the figure reported is the
% growth of the process's peak working set during the call, which only
% means something in a fresh process. Windows only (.NET Process API).
% Prints one line: side, samples, bins, seconds, peak MB above baseline.

if nargin < 4, minutes = 20; end
addpath(genpath(fullfile(repo,'allguis')));
addpath(fullfile(baseDir,'swap'));

fs = 40; L = round(minutes*60*fs); t = (0:L-1)'/fs;
rng(9);
s1 = cos(2*pi*0.9*t + 0.3*sin(2*pi*0.05*t)) + 0.5*cos(2*pi*2.3*t) + 0.3*randn(L,1);
s2 = cos(2*pi*0.9*t + 1.1) .* (1 + 0.4*cos(2*pi*2.3*t)) + 0.3*randn(L,1);
args = {'fmin',0.1,'fmax',5,'f0',1,'Display','off','Plot','off'};

switch side
    case 'new',  fn = @bispecWavNew;
    case 'base', fn = @bispecWavNew_base;
    otherwise, error('side must be ''new'' or ''base''');
end

% warm up on a short signal so JIT, FFT plans and path caching are not
% counted against the long run
fn(s1(1:4000), s2(1:4000), fs, args{:});
clear transformCached

p = System.Diagnostics.Process.GetCurrentProcess();
p.Refresh(); peak0 = double(p.PeakWorkingSet64); ws0 = double(p.WorkingSet64);
tic; [B, f] = fn(s1, s2, fs, args{:}); sec = toc;
p.Refresh(); peak1 = double(p.PeakWorkingSet64);

fprintf('RESULT side=%s samples=%d bins=%d seconds=%.1f peak_growth_MB=%.0f peak_above_start_MB=%.0f\n', ...
    side, L, numel(f), sec, (peak1-peak0)/2^20, (peak1-ws0)/2^20);
save(fullfile(baseDir, sprintf('bispec_long_%s.mat', side)), 'B', 'f', '-v7.3');
end
