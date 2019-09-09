# Tutorial: Statistical Learning

In this final tutorial of the course, we will learn how to analyze a big dataset of light curves measured from [NASA's _Kepler_ mission](https://www.nasa.gov/mission_pages/kepler/overview/index.html).
You can learn more about this mission in [this YouTube video](https://www.youtube.com/watch?v=3yij1rJOefM).

![](img/starryKepler.png)

Gone are the days of manually observing and storing a handful of stars--modern astronomy is characterised by datasets of unprecedented size and complexity.
It is unfeasible to manually sift through datasets of this magnitude.
Correspondingly, modern research methods have adapted, requiring researchers to develop expertise in statistical learning methods that allow patterns to be found and quantified in big datasets.

In this tutorial, you will work through the techniques and concepts introduced in the lecture to find characteristic of different stars from the Kepler mission.
By the end of the tutorial, you will have developed a way to automatically detect different types of stars from their light curves.

### Background

Today we will focus on the problem of predicting a star's identity from the properties of its light curve.

Is this problem a supervised or unsupervised problem?
Is it a classification or regression problem?

The light curve is the brightness of a point in the sky, measured repeatedly through time.
As shown in the plot below, we have seven classes of object that we're interested in detecting (labels of each star are given in parentheses):

1. Detached binary ('detached')
2. Contact binary ('contact')
3. RR Lyrae ('RRLyr')
4. Gamma Dor ('gDor')
5. Delta Scuti ('dSct')
6. Rotating binary ('rot')
7. Non-variable star ('nonvar')

Take a moment to inspect a representative example of each class in the time-domain and frequency-domain plots below.
Just looking at the data, what types of properties do you think are going to be useful in distinguishing these seven types of stars?

![](img/classTimeseries.png)

### Exploring the dataset

First we need to understand how our data is structured.
A good place to start is our time-series data, stored in a table, `TimeSeries`, in the Matlab file `Kepler_TimeSeries.mat`.
Load the `TimeSeries` table into your Matlab workspace, and take a look at the first handful:

```matlab
head(TimeSeries)
```

In total there are 1341 time series (`height(TimeSeries)`).
Each corresponds to a star observed by the Kepler telescope.
The data for each is in the Data column.
If you are not familiar with table objects in Matlab, note that you can pick out the time-series data for object `i` as `TimeSeries.Data{i}`, and get its metadata as `TimeSeries(i,:)`.

:question::question::question:
Pick a star to plot, zooming in to see if you can discern any interesting temporal structure.
Put the star's ID (`Name`) and class label (`Keywords`) in the title.

### Feature extraction

Recall from lectures that we need to represent the problem in the form of an observation x feature data matrix (`X`) and a target output vector (`y`).
In this tutorial we are going to represent time series by their different properties in the matrix `X`, and label each by one of seven categories in `y`.

![](img/problemSetUp.png)

### Two-Class Classification

Let's get our confidence up by starting our analysis on a simpler two-class problem.

Start by filtering down to just classes: `contact` and `nonvar`.
You can achieve this by:
1. Define `classesToKeep = {'contact','nonvar'}`
2. Use the `ismember` function on the `Keywords` to find matching indices.
3. Apply the logical filter to generate a new subsetted table, `TimeSeriesTwo`.

You should have 385 matches.

A physicist's first instinct when working with time series is to transform to the frequency domain.
Those of you doing PHYS/DATA3888 should be familiar with numerically estimating the Fourier transform of a time series.
I've worked you up a simple implementation in `ToFrequency`.

```matlab
doPlot = true;
ToFrequency(TimeSeriesTwo.Data{1},doPlot);
```

Fill in the missing code below to plot five examples of a `contact` star in both the time domain and the frequency domain:

```matlab
% Indices of contact binary stars ('contact') (in TimeSeriesTwo)
contactIndicies = find();

% Plot
f = figure('color','w');
numToPlot = 5; % Number of examples to plot
maxL = 500; % Only plot up to this maximum number of samples
for i = 1:numToPlot
    indexToPlot = contactIndicies(i);

    % Time Series
    subplot(numToPlot,2,(i-1)*2+1)
    plot(TimeSeriesTwo.Data{indexToPlot}(1:maxL),'k')
    xlabel('Time (samples)')
    title(sprintf('%s (%s)',TimeSeriesTwo.Name{indexToPlot},...
                    TimeSeriesTwo.Keywords{indexToPlot}))

    % Power Spectrum
    subplot(numToPlot,2,i*2)
    ToFrequency(TimeSeriesTwo.Data{indexToPlot},true);
end
```

Repeat for the non-variable (`nonvar`) stars.

From inspecting just five examples of each type, what types of frequency-domain features do you think will help you classify the two types of stars?

### Choose Your Own Features

Pick two different frequency ranges that you think are going to be informative of the differences between contact binaries and non-variable stars.
Write a new function, `f = MyTwoFeatures(x);` that takes in a time-series, `x`, and outputs two features (real numbers stored in the two-element vector, `f`), that represent the power in your two chosen frequency bands.
The `bandpower` function may be helpful.
A draft function structure has worked up for you, including a pre-processing using the _z_-score.

:fire::fire::fire:
If you feel boxed in by the `bandpower` features, don't!
Feel free to be creative in defining any two features you desire, from properties of the distribution of power spectral densities such as its centroid, to the location of prominent peaks, you da boss!
As long as you construct a function, `MyTwoFeatures`, that outputs two relevant features about the input, `x`.

Now compute these two features for time series in `TimeSeriesTwo`, yielding a 385 x 2 (time series x feature) matrix, `dataMatrix`.

We did it!
Now let's do some statistical learning on this matrix.

#### Plotting
Notice have the two ingredients we need for learning:
1. An observation x feature data matrix (`dataMatrix`)
2. Ground-truth labels for each item (`TimeSeriesTwo.Keywords`).

We can denote these labels as a `categorical` instead of the current string.
This tells Matlab to treat them not as lots of pieces of text, but as one of a set of categorical labels:

```matlab
outputLabels = categorical(TimeSeriesTwo.Keywords);
```

Let's first see how our two features are doing at separating the two classes.
We can use the `gscatter` function to label our observations by their class:

```matlab
gscatter(dataMatrix(:,1),dataMatrix(:,2),outputLabels)
xlabel('My Feature 1')
ylabel('My Feature 2')
```

:question::question::question:
Given how you constructed your two features above, give each axis an appropriate label and upload your plot.

Do your features separate the two classes?
Can you interpret why the two classes are where they are in the feature space you constructed?

#### Training a classifier

Let's train a linear SVM to distinguish:

```matlab
Mdl_linearSVM = fitcsvm(dataMatrix,outputLabels,'KernelFunction','linear');
```

You just did 'machine learning'.
Easy, huh?!

We can now get the predicted labels for the data we fed in:
```matlab
predictedLabels = predict(Mdl_linearSVM,dataMatrix);
```

Use the AND logical condition (`&`) to count how many `nonvar` stars were predicted to be `nonvar` stars?
Repeat for `contact`.
Did the simple SVM classifier do pretty well in automatically distinguishing these two types of stars?

We can construct a confusion matrix as:
```matlab
confusionmat(outputLabels,predictedLabels)
```
Do the results match the diagonal entries you computed manually above?

Does your model's prediction improve for an SVM with a radial basis kernel (`KernelFunction`,`rbf`)?

Evaluate each model across a grid of your feature space using the `predict` function.
This is implemented for you in `gridPredictions`, but you'll need to fill in a few parts (marked `...`) to make sure you understand what's going on.

Does the trained SVM classifier learn to distinguish high-density areas of each class?

Does the boundary change if you allow a nonlinear boundary, by setting the `KernelFunction` to `rbf`?

:question::question::question:
Upload plots of your SVM's predictions and data when using the `'linear'` and the `'rbf'` kernel.

### All classes?

Now that we're got some intuition with a couple of classes, and are now armed with a reasonable two-dimensional feature space(!), let's go back to the full seven-class problem.
