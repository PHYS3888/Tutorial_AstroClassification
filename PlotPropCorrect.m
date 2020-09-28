function PlotPropCorrect(propCorrect,classNames)

bar(propCorrect);
ax = gca;
ax.XTick = 1:length(classNames);
ax.XTickLabel = classNames;
xlabel('Type of star')
ylabel('Proportion correctly classified')

end
