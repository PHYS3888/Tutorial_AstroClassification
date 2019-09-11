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
3. RR Lyrae variable (`'RRLyr'`)
4. Gamma Doradus variable (`'gDor'`)
5. Delta Scuti variable (`'dSct'`)
6. Rotating variable (`'rot'`)
7. Non-variable (`'nonvar'`)

Take a moment to inspect a representative example of each class in the time-domain and frequency-domain plots below.
Just looking at the data, what types of properties do you think are going to be useful in distinguishing these seven types of stars?

![](img/classTimeSeries.png)

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

We went through a few examples of different classification strategies in the lecture:
boundary-based linear discriminant analysis and decision trees, and neighbor-based _k_-nearest-neighbor (_k_-NN).

Today we're going to mainly use a support vector machine classifier (SVM), which can (extremely simply) be viewed as a souped-up variant of linear discriminant analysis.
We're going to consider two variants:
1. `'linear'` SVM, in which each pair of classes is separated by a linear boundary, and
2. `'rbf'` kernel SVM, in which a local Gaussian distribution centered around each data point can generate more complicated nonlinear boundaries between pairs of classes.

Let's first train a linear SVM to distinguish:

```matlab
Mdl_linearSVM = fitcsvm(dataMatrixTwo,outputLabelsTwo,...
                    'Standardize',true,...
                    'KernelFunction','linear');
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

If you get less than 100% prediction accuracy, investigate using a radial basis kernel function (`theKernel = 'rbf'`):
```matlab
Mdl_rbfSVM = fitcsvm(dataMatrixTwo,outputLabelsTwo,...
                    'Standardize',true,...
                    'KernelFunction','rbf',...
                    'KernelScale','auto');
```

Does your model's prediction improve?

So now you have two trained models, `Mdl_linearSVM` and `Mdl_rbfSVM`.
Evaluate each model across a grid of your feature space using the `predict` function.
This is implemented for you in `gridPredictions`, but you'll need to fill in a few parts (marked `...`) to make sure you understand what's going on.

Try out `gridPredictions`!
1. Set `doPosterior = false`. This mode plots the predicted class at each point in feature space.
2. :fire: Set `doPosterior = true`. This mode plots the estimated probability of being a given class at each point in the feature space.

Does the trained SVM classifier learn a sensible prediction profile to distinguish the two classes of stars?
How does the boundary change between the linear (`Mdl_linearSVM`) and nonlinear (`Mdl_rbfSVM`) SVMs?

:question::question::question:
Upload a plots of the data and your trained SVM's in-sample predictions using either the `'linear'` or `'rbf'` kernel.

:fire::fire::fire:
On this problem, it was not so difficult to get near (or precisely) 100% classification accuracy.
For a challenge, play around with randomizing a proportion of labels (e.g., swap the labels on 10% of observations), or add some noise into the feature calculation (e.g., `fNoisy = f + 0.2*randn(2,1)`).
See what happens to your prediction model, and the differences in prediction patterns between the `linear` and `rbf` kernel SVMs.

## The seven-class problem :open_mouth:

Now that we're got some intuition with a couple of classes, and we are now armed with a reasonable two-dimensional feature space, let's go back to the full seven-class problem.
:muscle::metal:

Repeat the steps performed above for the full dataset:

__Compute features__.
Use your `MyTwoFeatures` function to compute two features for each time series in the full dataset, saving the result to the 1341 x 2 matrix, `dataMatrix`.

__Plot data in the feature space__.
Extract the categorical class labels as `outputLabels`, and then plot the class distributions for all seven classes in your two-dimensional feature space using `gscatter`.
(_Note:_ you may wish to play around with the formatting of the plot, e.g., showing points as larger crosses: `gscatter(dataMatrix(:,1),dataMatrix(:,2),outputLabels,'','x',10)`).

From visually inspecting the feature space, assess whether there is more or less overlap between the class distributions, and thus whether this is an easier or more difficult classification problem than the two-class problem analyzed above.

Now we can train a classification model for all seven types of stars!
:dancers::dancers::dancers:

The syntax is slightly different in the multi-class setting:
```matlab
% Train a multi-class SVM classification model with a given kernel function:
classNames = categories(outputLabels);
% Pick a base model:
t = templateSVM('Standardize',true,'KernelFunction','linear');
trainedModel = fitcecoc(dataMatrix,outputLabels,'Learners',t,...
            'ClassNames',classNames);
```

Then we can compute `predictedLabels` using the `predict` function (as above).

Let's take a quick look at how we did using the `gridPredictions` function:
```matlab
gridPredictions(trainedModel,dataMatrix,outputLabels,false);
```
Does the model learn sensible classification regions for each class?

Now inspect the seven-class confusion matrix in the command line using `confusionmat`.
Are some classes being classified more accurately than others?

How can you compute the number of true examples of each class from the confusion matrix?
For each class, compute the proportion of true examples of a class that are predicted as such: `propCorrect`.
Then plot it as a bar chart:

```matlab
bar(propCorrect);
ax = gca;
ax.XTick = 1:length(classNames);
ax.XTickLabel = classNames;
xlabel('Type of star')
ylabel('Proportion correctly classified')
```

:question::question::question:
Upload your plot.

Take the simplest measure of in-sample classification performance: counting the number of correct classification events.
Does this metric improve in the case of the `'rbf'` kernel relative to the `'linear'` kernel?

:question::question::question:
Does a boost in in-sample accuracy from applying a more complex model always represent an improvement?
Why or why not?

#### :fire::fire::fire: (Optional): Class Imbalance
A classifier that is simply trying to maximize the number of correct classification events will be biased towards over-represented classes.
Look at the classification accuracy of each class (proportion of predictions of that class that match the true examples of that class).
Does this intuition broadly hold true in the case of your classifier?
You can fix this by adding weights, as an additional setting in the `fitcecoc` function (`'Weights',w`), for a weight vector, `w`, that sets the importance of classifying each object in the dataset.
Try setting the weights such that each class contributes the same total weight.
This is called inverse probability weighting.
Compare the resulting `propCorrect` chart--does the classifier learn to treat the smaller classes more seriously?

#### :fire::fire::fire: (Optional): More complex classifiers
Try different base models by altering the line `t = templateSVM('Standardize',true,'KernelFunction','linear');`
For example, try
* k-NN with k = 3: `t = templateKNN('NumNeighbors',3,'Standardize',true);`
* Decision tree: `t = templateTree('Surrogate','off');`
* RBF-SVM: `t = templateSVM('Standardize',true,'KernelFunction','rbf','KernelScale','auto');`

How do the inter-class boundaries look?
Does the in-sample performance improve?

### Predicting new stars

Now we can test our classifier!
New time series are in the `toTest` directory:
1. `5702601.txt`
2. `6042116.txt`
3. `8324268.txt`
4. `9664607.txt`
5. `10933414.txt`

You can load one as:
```matlab
theFiles = {'5702601.txt','6042116.txt','8324268.txt','9664607.txt','10933414.txt'};
numFiles = length(theFiles);
X = cell(numFiles,1);
for i = 1:numFiles
    fileLocation = fullfile('toTest',theFiles{i});
    X{i} = dlmread(fileLocation,'',1,1);
end
```

Compute your two features for each of these five stars, storing your results as a `newStarFeatures` matrix.

Plot the new stars in your two-dimensional feature space, including the class boundaries trained above (`gridPredictions`).

Now we can use the `predict` function to classify each of these stars based on their features, using the patterns we learned and stored in the `trainedModel` above:

```matlab
modelPrediction_star1 = predict(trainedModel,newStarFeatures(1,:));
```



### :fire::fire::fire: A large feature space
Our results above were pretty impressive from using just two simple features of the power spectral density.
Let's see if we can improve our predictions by adding thousands of diverse additional time-series features.
We have already done the calculation for you (:fire: FYI :fire:, you can access the software we used for mass time-series feature extraction [here](https://github.com/benfulcher/hctsa)).
