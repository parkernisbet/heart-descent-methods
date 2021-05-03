# Heart Descent Methods
Partial scratch-implementation of coordinate descent, using a specified feature selector method to choose and later update a coordinate's corresponding weight. This behavior mimics the "fit" function of most descent-compatible machine learning algorithms.

Possible choices for "selector_method" are:

  1. 'a': adaptive, meaning the program will choose based on descending absolute per feature gradient value
  2. 'c': cyclic, meaning progressive looping over the dataset's features in order of increasing index
  3. 'r': random, meaning a random index from the feature set is chosen to update

My implemented loss function is based off the generalized formula for binary cross entropy, deviating to instead provide the averaged loss across all data points used.

Unsurprisingly, the adaptive coordinate selection method came out on top, with a 35% improvement on iterations to convergence. Between the random and cyclic selector methods, cyclic saw slight though not significant improvements on iteration count and time. The most notable difference between the latter two "underperforming" coordinate selector methods was the more smoothed nature of the iterations vs average loss plot for 'c'.
