function C = build_cases()
%BUILD_CASES  Deterministic case matrix shared by the baseline and production runs.
%   Every case is fully described by this file, so both runs execute an
%   identical list in an identical order.

fs = 40;
C = struct('id',{},'fn',{},'kernel',{},'f0',{},'args',{},'L',{});

% -- deterministic test signals, even and odd length -----------------------
lens = [256, 257];

wavelets = {'Lognorm','Morlet','Bump','Morse-3'};
windows  = {'Gaussian','Hann','Blackman','Exp','Rect','Kaiser-3'};
f0s      = [1, 2];
prep     = {'on','off'};
cuts     = {'on','off'};
pads     = {'predictive', 0, 'symmetric'};
bands    = {'default','narrow'};

k = 0;
for iL = 1:numel(lens)
  for iF = 1:numel(f0s)
    for iP = 1:numel(prep)
      for iC = 1:numel(cuts)
        for iD = 1:numel(pads)
          for iB = 1:numel(bands)

            base = {'Display','off','Plot','off', ...
                    'Preprocess',prep{iP},'CutEdges',cuts{iC},'Padding',pads{iD}};
            if strcmp(bands{iB},'narrow')
              band = {'fmin',0.3,'fmax',8};
            else
              band = {};
            end

            for iW = 1:numel(wavelets)
              k = k+1;
              C(k).id     = sprintf('wt|%s|f0=%g|pre=%s|cut=%s|pad=%s|band=%s|L=%d', ...
                             wavelets{iW}, f0s(iF), prep{iP}, cuts{iC}, ...
                             padname(pads{iD}), bands{iB}, lens(iL));
              C(k).fn     = 'wt';
              C(k).kernel = wavelets{iW};
              C(k).f0     = f0s(iF);
              C(k).args   = [base, band, {'Wavelet',wavelets{iW},'f0',f0s(iF)}];
              C(k).L      = lens(iL);
            end

            for iN = 1:numel(windows)
              k = k+1;
              C(k).id     = sprintf('wft|%s|f0=%g|pre=%s|cut=%s|pad=%s|band=%s|L=%d', ...
                             windows{iN}, f0s(iF), prep{iP}, cuts{iC}, ...
                             padname(pads{iD}), bands{iB}, lens(iL));
              C(k).fn     = 'wft';
              C(k).kernel = windows{iN};
              C(k).f0     = f0s(iF);
              C(k).args   = [base, band, {'Window',windows{iN},'f0',f0s(iF)}];
              C(k).L      = lens(iL);
            end

          end
        end
      end
    end
  end
end

% -- non-default code paths ------------------------------------------------
% Custom fwt-only cell wavelet (Morlet-like in frequency, no time form).
fwtOnly = {@(xi)exp(-(1/2)*(xi-2*pi).^2), [0 Inf], [], []};
k=k+1;
C(k) = mk('wt|custom-fwt-only', 'wt', 'custom-fwt', 1, ...
      {'Display','off','Plot','off','Wavelet',fwtOnly}, 256);

% wft custom twf-only branch (no frequency form).
twfOnly = {[], [], @(t)exp(-(1/2)*t.^2), [-4 4]};
k=k+1;
C(k) = mk('wft|custom-twf-only', 'wft', 'custom-twf', 1, ...
      {'Display','off','Plot','off','Window',twfOnly}, 256);

% Handle that rejects matrix input -> must trip the probe and fall back to
% the serial loop. Identical results are what prove the fallback is correct.
rejects = {@(xi)rejectMatrix(xi), [0 Inf], [], []};
k=k+1;
C(k) = mk('wt|handle-rejects-matrix', 'wt', 'reject', 1, ...
      {'Display','off','Plot','off','Wavelet',rejects}, 256);
k=k+1;
C(k) = mk('wft|handle-rejects-matrix', 'wft', 'reject', 1, ...
      {'Display','off','Plot','off','Window',rejects}, 256);

% Handle producing NaNs on the support -> retry / ouflag overflow path.
nanny = {@(xi)nanOnSupport(xi), [0 Inf], [], []};
k=k+1;
C(k) = mk('wt|handle-nan-on-support', 'wt', 'nan', 1, ...
      {'Display','off','Plot','off','Wavelet',nanny}, 256);
k=k+1;
C(k) = mk('wft|handle-nan-on-support', 'wft', 'nan', 1, ...
      {'Display','off','Plot','off','Window',nanny}, 256);

end

function s = mk(id, fn, kernel, f0, args, L)
s = struct('id',id,'fn',fn,'kernel',kernel,'f0',f0,'args',{args},'L',L);
end

function n = padname(p)
if ischar(p), n = p; else, n = num2str(p); end
end

function y = rejectMatrix(xi)
if ~isvector(xi)
    error('customWavelet:matrixInput','this handle only accepts vectors');
end
y = exp(-(1/2)*(xi-2*pi).^2);
end

function y = nanOnSupport(xi)
y = exp(-(1/2)*(xi-2*pi).^2);
y(abs(xi-2*pi) < 1e-3) = NaN;   % force NaNs inside the support
end
