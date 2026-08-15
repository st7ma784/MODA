function updateWaitbarSafe(h, idx, total, labelPrefix)
% UPDATEWAITBARSAFE  Update a waitbar's progress if it still exists.
%
% Intended as a parallel.pool.DataQueue 'afterEach' callback so a parfor
% loop's progress can be reported to a waitbar living on the client: parfor
% workers cannot touch graphics handles directly (a waitbar is a graphics
% object), but they CAN send() plain data through a DataQueue, which is
% received and processed back on the client — this function is what
% actually moves the waitbar once that notification arrives.
%
% INPUT:
% h:           waitbar handle
% idx:         which unit of work just completed
% total:       total number of units of work
% labelPrefix: text shown before the "(idx/total)" suffix (default: 'Calculating')

if nargin < 4
    labelPrefix = 'Calculating';
end
if ishandle(h)
    waitbar(idx/total, h, sprintf('%s (%d/%d)', labelPrefix, idx, total));
end
end
