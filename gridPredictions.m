function gridPredictions(trainedModel,dataMatrix,outputLabels,doPosterior)
% Evaluate a trained model's prediction at each point in
% a grid of input space.

if nargin < 4
    doPosterior = false;
end

%-------------------------------------------------------------------------------
% Define a grid across the range of each feature:
numPointsPerAxis = 50;
feat1_grid = linspace(min(dataMatrix(:,1)),max(dataMatrix(:,1)),numPointsPerAxis);
feat2_grid = linspace(min(dataMatrix(:,2)),max(dataMatrix(:,2)),numPointsPerAxis);
[F1,F2] = meshgrid(feat1_grid,feat2_grid);

%-------------------------------------------------------------------------------
% Evaluate the model at each grid-point:
[modelPrediction,posteriorProb] = predict(trainedModel,[F1(:),F2(:)]);

%-------------------------------------------------------------------------------
% Reshape back to square matrix:
modelPrediction = reshape(modelPrediction,numPointsPerAxis,numPointsPerAxis);
posteriorProb = reshape(posteriorProb(:,1),numPointsPerAxis,numPointsPerAxis);

%-------------------------------------------------------------------------------
% Plot how the predictions vary across the feature space:
figure('color','w'); hold('on')

if doPosterior
    % Plot posterior probability of each class in the space:
    contourf(feat1_grid,feat2_grid,posteriorProb)
    giveMeTurboMap()
    % Plot data points on top:
    gscatter(dataMatrix(:,1),dataMatrix(:,2),outputLabels,'kw','xo')
else
    % Plot class prediction at each point in the feature space:
    pcolor(feat1_grid,feat2_grid,double(modelPrediction))
    colormap(flipud(hot))
    % Plot data points on top:
    gscatter(dataMatrix(:,1),dataMatrix(:,2),outputLabels)
end


xlabel('My Feature 1')
ylabel('My Feature 2')

end
