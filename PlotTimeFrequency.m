function PlotTimeFrequency(TimeSeries,indicesToPlot,numToPlot,maxL)

% Settings:
if nargin < 3
    numToPlot = 5; % Number of examples to plot
end
if nargin < 4
    maxL = 500; % Plot up to this maximum number of samples in the time domain
end
%-------------------------------------------------------------------------------

% Generate the plots:
f = figure('color','w');
for i = 1:numToPlot
    indexToPlot = indicesToPlot(i);

    % Time Series
    subplot(numToPlot,2,(i-1)*2+1)
    plotTimeSeries(TimeSeries,indexToPlot,maxL);

    % Power Spectrum
    subplot(numToPlot,2,i*2)
    ToFrequency(TimeSeries.Data{indexToPlot},true);
end

end
