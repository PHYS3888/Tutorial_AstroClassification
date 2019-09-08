function ToFrequency(x,doPlot)

% Set default plotting behavior:
if nargin < 2
    doPlot = true;
end
%-------------------------------------------------------------------------------

N = length(x);
Fs = 1;
t = (1:N)';
xTable = timetable(seconds(t),zscore(x));

[pxx,f] = pspectrum(xTable);

if doPlot
    plot(f,pow2db(pxx),'k')
    grid('on')
    xlabel('Frequency (Hz)')
    ylabel('Power Spectrum (dB)')
end

end
