# Tutorial: Statistical Learning

In this final tutorial of the course, we will learn how to analyze a big dataset of light curves measured from [NASA's _Kepler_ mission](https://www.nasa.gov/mission_pages/kepler/overview/index.html).
You can learn more about NASA's _Kepler_ mission in [this YouTube video](https://www.youtube.com/watch?v=3yij1rJOefM).

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

1. Detached binary (`'detached'`)
2. Contact binary (`'contact'`)
3. RR Lyrae (`'RRLyr'`)
4. Gamma Dor (`'gDor'`)
5. Delta Scuti (`'dSct'`)
6. Rotating binary (`'rot'`)
7. Non-variable star (`'nonvar'`)

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

Each row of `TimeSeries` corresponds to a star observed by the Kepler telescope.
Light-curve data is contained in the `Data` column, while the other columns give additional information about each star, including its ID (`Name`) and assigned class (`Keywords`).

Verify that there are 1341 time series in total using `height(TimeSeries)`.
If you are not familiar with table objects in Matlab, note that you can pick out the time-series data for object `i` as `TimeSeries.Data{i}`, and get its metadata as `TimeSeries(i,:)`.

#### Plot a light curve
* Pick a star and plot its light curve, zooming in to see if you can discern any interesting temporal structure
* _Note:_ If you spot some flat lines, these represent artefactual periods or missing data, and have been set to the signal mean.
* Plot the signal using a valid and appropriate time axis with units (e.g., days), given that the Kepler telescope records a measurement every 29.45 minutes.
* Set a variable, `maxL`, that plots only the first `maxL` samples.
* Put the star's ID (`Name`) and class label (`Keywords`) in the title.

Write yourself a function `plotTimeSeries.m` (a template is provided for you) that takes in the `TimeSeries` table, an index of a time series to plot, and a maximum number of samples to plot, `maxL`, and outputs a time-series plot of the selected star.

:question::question::question: __Upload your code.__

### _Context_: Feature extraction

Recall from lectures that we need to represent the problem in the form of an observation x feature data matrix (`X`) and a target output vector (`y`).
In this tutorial we are going to represent time series by their different properties in the matrix `X`, and label each by one of seven categories in `y`.

![](img/problemSetUp.png)

### Two-Class Classification

#### Filtering to a subset

Seven classes are a bit daunting; let's build some confidence by starting with a simpler two-class problem.

Start by filtering down to just classes: `contact` and `nonvar`.
You can achieve this by:
1. Define `classesToKeep = {'contact','nonvar'}`
2. Use the `ismember` function on the `Keywords` column of the `TimeSeries` table to find matching indices.
3. Apply the logical filter on the rows of the `TimeSeries` table to generate a new table, `TimeSeriesTwo`.

You should get 385 matches (check `height(TimeSeriesTwo)`).

#### Plotting in the time and frequency domain

A physicist's first instinct when working with time series is to transform to the frequency domain.
Those of you doing PHYS/DATA3888 should be familiar with numerically estimating the Fourier transform of a time series.
I've worked you up a simple implementation in `ToFrequency`.
To get a valid time axis, you'll need to calculate and specify the sampling frequency, `fs` (Hz).

```matlab
doPlot = true;
fs = ...;
ToFrequency(TimeSeriesTwo.Data{1},fs,doPlot);
```

Fill in the missing code below to plot five examples of a `contact` star in both the time domain (using your `plotTimeSeries` function you constructed above) and in the frequency domain (using `ToFrequency`):

```matlab
% Indices of contact binary stars ('contact') (in TimeSeriesTwo)
contactIndicies = find(...);

% Settings:
numToPlot = 5; % Number of examples to plot
maxL = 500; % Only plot up to this maximum number of samples
fs = ...;

% Generate the plots:
f = figure('color','w');
for i = 1:numToPlot
    indexToPlot = contactIndicies(i);

    % Time Series
    subplot(numToPlot,2,(i-1)*2+1)
    plotTimeSeries(TimeSeriesTwo,indexToPlot,maxL);

    % Power Spectrum
    subplot(numToPlot,2,i*2)
    ToFrequency(TimeSeriesTwo.Data{indexToPlot},fs,true);
end
```

Repeat for the non-variable (`nonvar`) stars.

From inspecting just five examples of each type, what types of frequency-domain features do you think will help you classify the two types of stars?

### Choose Your Own Features

Write a new function, `f = MyTwoFeatures(x);` that takes in a time-series, `x`, and outputs two features (real numbers stored in the two-element vector, `f`), that represent two different properties of the power spectrum.
A draft function structure has worked up for you, including a pre-processing using the _z_-score.
You just need to implement the calculation of two features, as described below.

#### Feature 1: Peak in the power spectrum

Perhaps you noticed that the contact binaries have a characteristic oscillation.
Let's first measure how 'peaky' the power spectrum is.
The simplest metric for quantifying this is to simply take the `max` of the power spectral density.

:fire::fire: Feel free to instead implement an alternative feature, like the `skewness`, or an explicit peak-finding functions like `findpeaks`---anything that implements this idea of generating a _single real number_ that captures the peakiness of the power spectrum.

#### Feature 2: Power across a frequency range

The oscillation can also be detected as increased power across a range of frequencies.

Looking again at the frequency spectra you plotted above, pick a frequency range that you think is going to be informative of the differences between contact binaries and non-variable stars.
Implement this as your second feature in `MyTwoFeatures`.

_Note_: If you use the `bandpower` function, note that the frequency range should be measured in Hz.

#### Feature space

Now loop over all time series in `TimeSeriesTwo`, computing your two features for each.
Store the result as the 385 x 2 (time series x feature) matrix, `dataMatrixTwo`.

:milky_way::stars::satisfied::star2::bowtie::star2::smile::star2::laughing::stars::milky_way:

Have a brief celebration!
You have just generated the main programmatic machinery to convert light curves into an interpretable two-dimensional feature space that will enable the automatic classification of Kepler stars!

#### Plotting
Did you notice?: We now have the two ingredients we need for learning:
1. An observation x feature data matrix (`dataMatrixTwo`)
2. Ground-truth labels for each item (`TimeSeriesTwo.Keywords`).

We can denote these labels as a `categorical` instead of storing them as a string of text.
This tells Matlab to treat them not as lots of pieces of text, but as one of a set of categorical labels:

```matlab
outputLabelsTwo = categorical(TimeSeriesTwo.Keywords);
```

Let's first see how our two features are doing at separating the two classes.
We can use the `gscatter` function to label our observations by their class:

```matlab
figure('color','w')
gscatter(dataMatrixTwo(:,1),dataMatrixTwo(:,2),outputLabelsTwo)
xlabel('My Feature 1')
ylabel('My Feature 2')
```

:question::question::question:
Give each axis an appropriate label (corresponding to how you designed each feature, adding units where appropriate), and upload your plot.

Do your features separate the two classes?
Given your understanding of what your features are measuring, can you interpret why the two classes are where they are in the feature space?
Are there any outliers?

### Training a classifier

Let's train a linear SVM to distinguish:

```matlab
theKernel = 'linear';
Mdl_linearSVM = fitcsvm(dataMatrixTwo,outputLabelsTwo,...
                    'Standardize',true,...
                    'KernelFunction',theKernel);
```

You just did 'machine learning'.
Easy, huh?! :sweat_smile:

We can now get the predicted labels for the data we fed in:
```matlab
predictedLabelsTwo = predict(Mdl_linearSVM,dataMatrixTwo);
```

Use the logical AND condition (`&`) between your assigned labels (`outputLabelsTwo`) and predicted labels (`predictedLabelsTwo`) to count how many `nonvar` stars were predicted to be `nonvar` stars?
Repeat for `contact`.

Does this simple linear SVM classifier do well at distinguishing these two types of stars?

We can construct a confusion matrix as:
```matlab
[confMat,order] = confusionmat(outputLabelsTwo,predictedLabelsTwo)
```
Do the results match the diagonal entries you computed manually above?

If you get less than 100% prediction accuracy, investigate using a radial basis kernel function (`theKernel = 'rbf'`).
Does your model's prediction improve?

Evaluate each model across a grid of your feature space using the `predict` function.
This is implemented for you in `gridPredictions`, but you'll need to fill in a few parts (marked `...`) to make sure you understand what's going on.

Try it out, both with `doPosterior = false` (plotting the predicted class at each point in feature space), and `doPosterior = true` (plotting the estimated probability of being each class at each point in the feature space).

Does the trained SVM classifier learn a sensible prediction profile to distinguish the two classes of stars?

:fire::fire::fire:
__Training a nonlinear boundary.__
How does the boundary change if you allow a nonlinear boundary, by setting the `KernelFunction` to `rbf`?
You can train the `rbf` model as:
```matlab
Mdl_rbfSVM = fitcsvm(dataMatrixTwo,outputLabelsTwo,...
                    'Standardize',true,...
                    'KernelFunction','rbf',...
                    'KernelScale','auto');
```
(Inspect the behavior of the fitted model using `gridPredictions` as before).

:question::question::question:
Upload a plots of the data and your trained SVM's in-sample predictions using either the `'linear'` or `'rbf'` kernel.

:fire::fire::fire:
On this problem, it was not so difficult to get near-to (or precisely) 100% classification accuracy.
For a challenge, play around with randomizing a proportion of labels (e.g., swap the labels on 10% of observations), or add some noise into the feature calculation (e.g., `fNoisy = f + 0.2*randn(2,1)`).
See what happens to your prediction model, and the differences in prediction patterns between the `linear` and `rbf` kernel SVMs.

## The seven-class problem :open_mouth:

Now that we're got some intuition with a couple of classes, and we are now armed with a reasonable two-dimensional feature space, let's go back to the full seven-class problem.
:muscle::metal:

Repeat the steps performed above for the full dataset:
1. __Compute features__: Use your `MyTwoFeatures` function to compute two features for each time series in the full dataset, saving the result to the 1341 x 2 matrix, `dataMatrix`.
2. __Plot feature space__: Extract the categorical class labels as `outputLabels`, and then plot the class distributions for all seven classes in your two-dimensional feature space using `gscatter`.
(_Note:_ you may wish to play around with the formatting of the plot, e.g., showing points as larger crosses: `gscatter(dataMatrix(:,1),dataMatrix(:,2),outputLabels,'','x',10)`).

From visually inspecting the feature space, assess whether this is an easier or more difficult classification problem than the two-class version.

Now we can train a classification model for all seven types of stars!
Let's work with normalized features:
```matlab
dataMatrixNorm = zscore(dataMatrix);
```

The syntax is slightly different in the multi-class setting:
```matlab
% Train a multi-class SVM classification model with a given kernel function:
classNames = unique(outputLabels);
% Pick a base model:
% t = templateSVM('KernelFunction','linear');
t = templateLinear();
trainedModel = fitcecoc(dataMatrixNorm,outputLabels,'Learners',t,...
            'ClassNames',classNames);
```

Then we can compute `predictedLabels` using the `predict` function (as above), and look at the confusion matrix in the commandline (`confusionmat`).
[Don't forget to use `dataMatrixNorm`].

From reading off of the confusion matrix, can you tell whether some classes are being classified better than others?

Does the in-sample performance (e.g., the simplistic metric counting correct classification events) improve in the case of the `'rbf'` kernel relative to the `'linear'` kernel?

:question::question::question:
What was the in-sample classification accuracy of your classification model?
