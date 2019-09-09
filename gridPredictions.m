function gridPredictions(trainedModel,dataMatrix,outputLabels)
% Evaluate a trained model's prediction at each point in
% a grid of input space.

%-------------------------------------------------------------------------------
% Define a grid across the range of each feature:
% (if you have generated a custom feature outside the [0,1] range,
%   modify the range accordingly)
numPointsPerAxis = 50;
feat1_grid = linspace(0,1,numPointsPerAxis);
feat2_grid = linspace(0,1,numPointsPerAxis);
[F1,F2] = meshgrid(feat1_grid,feat2_grid);

%-------------------------------------------------------------------------------
% Evaluate the model at each grid-point:
modelPrediction = zeros(numPointsPerAxis);
for i = 1:numPointsPerAxis
    for j = 1:numPointsPerAxis
        % Fill in the model to predict from:
        modelPrediction(i,j) = predict(...,[F1(i,j),F2(i,j)]);
    end
end

%-------------------------------------------------------------------------------
% Plot how the predictions vary across the feature space:
figure('color','w');
hold('on')
% Fill in the quantity to plot as a color background:
pcolor(feat1_grid,feat2_grid,...)
colormap(hot)
gscatter(dataMatrix(:,1),dataMatrix(:,2),outputLabels)
xlabel('My Feature 1')
ylabel('My Feature 2')

end
