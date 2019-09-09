function ToFrequency(x,fs,doPlot)

% Set default sampling rate:
if nargin < 2
    fs = 1;
end
% Set default plotting behavior:
if nargin < 3
    doPlot = true;
end
%-------------------------------------------------------------------------------

N = length(x);
t = (1:1/fs:N/fs)';
xTable = timetable(seconds(t),zscore(x));

[pxx,f] = pspectrum(xTable);

if doPlot
    plot(f,pow2db(pxx),'k')
    grid('on')
    xlabel('Frequency (Hz)')
    ylabel('Power Spectrum (dB)')
end

end
