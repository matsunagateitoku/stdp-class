



# Class 3: Classifying Documents With Supervised ML
## Part 1: Process Data Sets for Supervised Machine Learning Classification Models
Module Introduction: Process Data Sets for Supervised Machine Learning Classification Models
Supervised and Unsupervised Machine Learning Techniques
A Simple Supervised Machine Learning Example
Examples of Machine Learning Problems in NLP


Training and Validating Data
Splitting Data Sets
Split a Simulated Data Set Into Training and Validation Sets
Practice Splitting a Simulated Data Set Into Training and Validation Sets
Split a Data Set of Names Into Training and Validation Sets
Practice Splitting a Data Set of Names Into Training and Validation Sets
Split Newsgroup Data Into Training and Validation Sets
Practice Splitting Newsgroup Data Into Training and Validation Sets
Scikit-Learn Submodules
Splitting Data into Training and Validation Sets
Course Project, Part One — Processing a Data Set for Supervised Machine Learning Classification Model
Module Wrap-up: Process Data Sets for Supervised Machine Learning Classification Models
Classifying Documents With Supervised Machine Learning
Process Data Sets for Supervised Machine Learning Classification Models


familiar with the types of problems you can solve using supervised machine learning. 
examine the process of teaching a model to classify data by dividing a labeled dataset into training and validation sets. 
practice dividing both real and simulated data sets into training and validation sets using scikit-learn. 

There are two main machine learning techniques: supervised and unsupervised learning methods. In supervised learning, our algorithm learns the relationship between input feature x and output label y. After the model is trained, we can predict an outcome y for a new input x. A regression model predicts a quantity, and we also have a classification model which predicts a class or category, which may not be numeric. In unsupervised learning, the algorithm learns the relationship between features or between unlabeled observations. The former is used to reduce dimensionality. For example, we might compress a 300-dimensional word vector into two-dimensional points to visualize this on a plot or to feed this into some other predictive model. This compression is not lossless and requires care. The observations can also be clustered according to their similarities or distances. Often, unsupervised methods prepare a data matrix for supervised learning as part of a machine learning pipeline. A bit on terminology: The label y is also synonymous with "response," "target," "outcome," or "dependent variable." Likewise, the input x is also known as a "predictor," "feature," "independent variable," "regressor," "covariate," or "explanatory variable." Much of this depends on the context you are working with. Maybe in physics they use some terminology that is different from statistics. We typically package inputs into a data matrix or design matrix, where columns are features with homogeneous value types and rows are individual observations with possibly heterogeneous value types. This is similar to a tabular format of a SQL table.
 
This section is all about splitting dataset into train, test, and validate sets. 

## Part 2: Train a Classification Model To Categorize Text

Module Introduction: Train a Classification Model To Categorize Text
What Is a Model?
A model can be either parametric or non-parametric. 
A parametric model, such as logistic regression, has a fixed set of parameters. These parameters are learned on the sample of observations contained in inputs and outputs. 
A non-parametric model does not have a fixed set of parameters or may not have parameters at all. K-Nearest Neighbors, or kNN, is a non-parametric model with no parameters and no model training stage. 
A decision tree is an example of a non-parametric model with variant set of parameters. A tree structure doesn't have a fixed number of branches, for example, which are learned from a given training set. So the structure of a tree can vary dramatically from one training set to another. 
Note that there are two types of parameters. 
Model parameters are coefficients — or model coefficients that are automatically learned by an underlying optimization algorithm. For example, an intercept or a slope in logistic regression are such parameters. We want the model to remain flexible in a particular task, but with fewer parameters, because each parameter will require estimation, and that adds uncertainty to the model. You want a balance between the flexibility and parameters that the model needs to estimate. 
Hyperparameters are those controlled by human experts. For example, we could explicitly control the depth of a decision tree or whether logistic regression learns an intercept or not. 
Parametric Classification Models
A classifier is a model that takes an observation x and helps us determine the corresponding category y. A simple binary classifier takes a scalar input and assigns it to one of two classes. For example, we might train a model to take the numeric attributes of a particular book and classify it as either "fiction" or "science" genre. Some features might include the fraction of Greek letters per page or number of chapters or number of pages and publication year and so on. Different types or families of models learn different patterns and report results differently. Let's review a few parametric models. If you remember parametric models, you have a fixed set of parameters that need to be learned during training. In logistic regression, it is intuitive to have the output classes of zero and one. The model outputs the probability of the observation x being in class one, or we call it positive class, and its complementary probability of x being in class zero. The two probabilities always sum up to one. In our book example, we may give a model a vector for a book with two Greek letters per page, 10 chapters, 500 pages, and 2020 publication year, all packed as the vector 2, 10, 500, and 2020. It may output a 0.7 probability of class "science" and, of course, corresponding or complementary 0.3 probability of class "fiction." A naive Bayes is another amazingly naive and effective algorithm. It assumes no relation among the features. Remember that no relation among the observations is pretty much always present in these types of models. The relation would be assumed in a time series model — that's a different topic altogether. In our book example, the model ignores any relationship between number of pages or number of chapters, although we know that this may not be a completely realistic assumption. Nonetheless, it is very effective. Just like logistic regression does, naive Bayes computes the so-called posterior probability of the class y for the given observation. Because of an independence assumption, this is just the product of so-called likelihoods scaled by the prior probability of the class. Each likelihood is the probability of a feature value given the class y. That's inverse to posterior, backwards posterior, I would say, or inverted. This model scales well with the number of features and can handle many different classes. Neural network is yet another example of a parametric model, where its structure is a set of fixed layers and fixed number of neurons in each layer. Each neuron is basically a logistic regression with a fixed number of parameters that determine the number of inputs and number of outputs in the model in that neuron. Even though we might be training millions of parameters, their number is still fixed.
Non-Parametric Classification Models
Now let's acquaint with a few popular non-parametric models which do not have a fixed number of parameters or may not have any parameters at all. Support vector classifier learns the so-called decision boundary or best separating plane or hyperplane in a multidimensional space between the classes, and it's all done in the vector space. The points on one side of this plane are labeled as minus one or negative class, and the points on the other side are labeled as positive one or they're called positive class. The given new observation is pushed through the model and SVC computes which side of this fitted plane, positive or negative, the point belongs to and returns the corresponding label, minus one or plus one. In our book example, the trained model would have learned a plane that has as many fiction book vectors as possible on its one side, and as many science book vectors as possible on its opposite side. The k-Nearest Neighbor, or kNN, algorithm doesn't have any parameters to train at all. So its training time is zero and all work is done during model querying or model inferencing; we also call it model testing. Given an input vector, kNN looks for the nearest k vectors and returns their majority class. So in the book example, if k equals 10 and our query vector has eight neighboring vectors labeled as "science" and two labeled as "fiction," the majority class "science" is returned for the test case vector. kNN also computes the probabilities of a "science" class, which is 80% in this case, and the probability of a "fiction" class, which is 20% in this case. kNN quality is very sensitive to the hyperparameter k, which we must provide. We can try different values of k to identify the ones that yield the lowest testing error. That's typically done through trial and error of different cases. kNN classifier and the regressor — there's another model for regression called kNN regression — is a very effective model for low-dimensional vectors with dimensions three, four, maybe 10, but to be avoided if vector dimensionality is greater than 30 or 40. A decision tree is a highly flexible model which recursively or iteratively builds a tree of split nodes starting from the root node and all the way down to the leaf nodes. The intermediate nodes are in between. Every node is basically a split, and the split is just a feature that we choose and its value, which are best separating the observations into two classes. So every split produces two subsets, and they have to be as pure as possible in the classes that they contain. Each time the sample of observations is split, we look for the next best split on each of the resulting subsets of observations. This model is highly interpretable and is so powerful that it's nearly useless due to overfitting. To make it usable, we must trim or prune the tree branches so that the tree learns the general pattern or the general structure or distribution, and not every bit of noise in the training observations. A random forests, basically an ensemble or meta model that combines a large set of very small or very shallow decision trees, sometimes even stumps — those are decision trees with just one node or one split. Each stump is just barely better than a random guess. But together, combined, they are exceptionally effective. During training, each stump is trained on a subset of observations and subset of features. So stumps are poorly correlated, which is exactly what we want. We want that the final prediction, which is an average prediction of all the stumps, cancels out all the error of these decorrelated trees and a more reliable and stable prediction is returned.
Parametric and Non-Parametric Models Chart

Choosing an Appropriate Classification Model
The Process of Training a Logistic Regression Model
Apply Sigmoid and Logit Functions for Logistic Regression
Many machine learning algorithms learn simple linear combinations of input vectors which result in most likely outputs. This works well if the underlying input/output relation is in fact linear. If not, then the model or function fit can be severely affected by nonlinearities and extreme values. Specifically, in binary classification, the quantitative input can be any real value, but the output is constrained to classes zero or one. Sigmoid function to the rescue! It maps any real value to a unique number in a 0–1 range. If thresholded at 0.5, then any value above 0.5 becomes plus one and others are mapped to plus zero. So you can see how zero and one classes are convenient here. So the output of the sigmoid function can be thought of as the probability of plus one given its input value. Then the probability of one and class one coincide, which is convenient and intuitive. The logit function is the inverse of the sigmoid function, meaning that it takes probability and returns the positive or negative real value, not the probability anymore. The logistic function is a generalization of the sigmoid function. It takes a vector of values — that product this — with some parameters, and the resulting scalar value is mapped to probability via sigmoid function. Let's see this in code. We are declaring sigmoid and logit. These are inverses of each other. Sigmoid is the s-shaped function from left to right, and logit is that function flipped around diagonal, so it goes from top to bottom. The sigmoid can be declared a number of ways. We're using this single exponent definition, and the logit would be using the logarithm to reverse that effect. So if we run this function and we declare a couple of vectors — these are vectors of real values for the x vector and probability values for the p vector. They don't necessarily have anything to do with each other, but we'll see the effect for each one when we run this through these functions. Notice that in your modeling, the x here is vector of values. But it could be vector of vectors, where each element is actually an observation, a vector of its own. If we run these sigmoid/logit functions on these vectors, then sigmoid maps vector x of individual values to individual probabilities. And logit function maps probabilities to real values. Logit of sigmoid will cancel out the effect that the sigmoid has and return back vector x. Sigmoid of logit will cancel out the effect of logit and return back the probability vector p. And the logit of 0.5 is the zero; it's centered at zero. We can also define these two other functions, logistic and inverse, and they will take a set of parameters, beta-naught, beta-one — beta-naught being an intercept and beta-one a slope. You can have multiple — beta-two, beta-three, and so on. Here we're only using univariate functions which only consider one element, each element of vector x, as an observation. If we run the vector x through logistic function with these parameters, then a vector of probabilities is returned, and the inverse logistic applied to logistic with the same parameters will cancel out the effect and the original vector x is returned. Let's see what these parameters, betas, do to the function, to the logistic function, which also looks like the letter "s." The beta-naught parameter, which is the intercept, controls the horizontal shift of the logistic function. So the lower beta-naught will push the curve to the right and the higher beta-naught will push the curve to the left, horizontally, but it does not affect the slope or the stretch of the function, which is controlled by the slope parameter beta-one. Beta-one will stretch the function if the beta-one is higher and greater than one. If it shrinks towards zero, but still positive, it will make the function steeper or compress the function. And if beta-one is negative, it will completely rotate or flip the function, which is what this red line is.
Practice Applying Sigmoid and Logit Functions for Logistic Regression
Training and Validating a Baseline Classification Model
Practice Training and Validating a Baseline Classification Model
Select a Model To Outperform the Baseline
Practice Selecting a Model To Outperform the Baseline
Training and Validating a Logistic Regression Model
Course Project, Part Two — Train a Classification Model To Categorize Text
Module Wrap-up: Train a Classification Model To Categorize Text

 

In scikit-learn, when you call the `fit` method on a `LogisticRegression` model, it performs several key operations under the hood to train the model on your data. Here’s a breakdown of what happens:


1. **Input Validation**: The method first checks the input data (features and target variable) to ensure they are in the correct format and that the number of samples matches between the features and the target.


2. **Data Preprocessing**: It may apply some preprocessing steps, such as handling missing values or standardizing features (if specified). Logistic regression often benefits from standardized input data, but this is not automatically done unless you explicitly apply scaling before fitting.


3. **Initialization**: The model initializes the weights (coefficients) for the logistic regression algorithm. These weights will be adjusted during the optimization process to minimize the loss function.


4. **Loss Function Definition**: The model uses the logistic (logit) loss function, which is defined as the negative log-likelihood of the observed labels given the predicted probabilities. The goal is to minimize this loss function during training.


5. **Optimization Algorithm**: The `fit` method employs an optimization algorithm (like stochastic gradient descent or a more advanced algorithm such as LBFGS) to update the weights. The optimization process involves:
   - **Forward Pass**: Calculating the predicted probabilities using the current weights and the logistic function.
   - **Loss Calculation**: Computing the loss based on the predictions and the true labels.
   - **Backward Pass**: Calculating gradients of the loss with respect to the weights.
   - **Weight Update**: Adjusting the weights using the gradients and a learning rate.


6. **Convergence Check**: The algorithm checks for convergence by monitoring changes in the loss function or weights. If changes are below a specified threshold or after a set number of iterations, the training stops.


7. **Storing the Model**: After fitting, the learned coefficients (weights) and intercept are stored within the model object, so you can use them for predictions later.


8. **Output**: The `fit` method typically returns the fitted model itself, allowing for method chaining if desired.


Overall, the `fit` method encapsulates a complete training routine that prepares the logistic regression model to make predictions on new data.


