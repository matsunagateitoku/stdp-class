
# Class 4: Topic Modeling With Unsupervised Machine Learning
## Module 1: Identify Keywords and Keyphrases From Text

Key Concept: Colocation 
- CODE: Practice Identifying Collocations
- Find coalitions in text (n-grams) and rank their importance with pointwise mutual information (PMI), which is a degree of increased information from the additional word in the phrase, regardless of word order
do optional practice
- keyword extraction algorithms (KEAs), 
- the nltk implementation of RAKE and the pke implementation of TextRank
- Collocation is the identification of tokens — words, phrases, or other n-grams — that tend to occur together more often than yo
u might expect by chance.
Rapid Automatic Keyword Extraction (RAKE) is a simple algorithm used to determine the relative importance of keywords and keyphrases in a document. It works by identifying content words in a corpus and assigning them a score based on their frequency, as well as a less intuitive score known as degree. Words and phrases that appear frequently and as part of long phrases receive higher scores and are thus identified as more valuable keywords and keyphrases.

| Lesson          |         Discription                                | Colab link    |
|-------------------|----------------------------------------------|------|
| **Module Introduction: Identify Keywords and Keyphrases From Text**   | Identify Keywords and Keyphrases From Text    |                                 
| **Introduction to Text Summarization**   | Identify Keywords and Keyphrases From Text    |                                  |
| **Overview of Unsupervised Learning**   | Identify Keywords and Keyphrases From Text    |                                  |
| **Module Introduction: Identify Keywords and Keyphrases From Text**   | Identify Keywords and Keyphrases From Text    |                                 
| **Review Text Summarization**   | Identify Keywords and Keyphrases From Text    |                                  |
| **Identify Collocations**   | Identify Keywords and Keyphrases From Text    |                                  |
| **Review Text Summarization**   | Identify Keywords and Keyphrases From Text    |                                  |
| **Practice Identifying Collocations**     | Using the NBA's API to track team win totals  |[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matsunagateitoku/stdp-class/blob/main/4.1.1.Practice_Comparing_Texts.ipynb) 
|**Keyword Extraction Methods**||
|**Keyword Extraction With RAKE**||
|**Determine Relative Importance With RAKE**||
|**Practice Extracting Keywords With RAKE**||
|**Keyword Extraction With TextRank**||
|**Implement the TextRank Algorithm for Keyword Extraction**||
|**Practice Extracting Keywords With TextRank**||
|**Interpreting Graph Outputs From TextRank**||
|**Measuring Keyword Quality**||
|**Practice Measuring Keyword Quality**||
|**Course Project, Part One — Identifying Keywords and Keyphrases From Text**||
|**Module Wrap-up: Identify Keywords and Keyphrases From Text**||



## Section 2: 
| Lesson          |         Discription                                | Colab link    |
|-------------------|----------------------------------------------|------|
|**Module Introduction: Identify Topics From Documents** ||
|**Introduction to Topic Modeling** ||
|**Basic Matrix Operations** ||
|**Practice With Basic Matrix Operations** ||
|**Matrix Decomposition** ||
|**Practice Matrix Decomposition** ||
|**Decompose a Matrix Using Singular Value Decomposition (SVD)** ||
|**Practice Decomposing a Matrix Using SVD**||
|**Approximate a Matrix With Truncated SVD**||
|**Dimensionality Reduction**||
|**SVD and Truncated SVD**||
|**Practice Approximating a Matrix With Truncated SVD**||
|**Identify Topics Using Latent Semantic Analysis (LSA)**||
|**Latent Semantic Analysis (LSA)**||
|**Practice Identifying Topics Using LSA**||
|**Identify Topics Using Latent Dirichlet Allocation (LDA)**||
|**Latent Dirichlet Allocation (LDA)**||
|**Practice Identifying Topics Using LDA**||
|**Course Project, Part Two — Identifying Topics From Documents**||
|**Module Wrap-up: Identify Topics From Documents**||

## Section 3: 
| Lesson          |         Discription                                | Colab link    |
|-------------------|----------------------------------------------|------|
|**Module Introduction: Extract Summary Sentences From Documents** ||
|**Introduction to Document Summarization** ||
|**Practice Extracting a Summary With NLTK** || [![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matsunagateitoku/stdp-class/blob/main/class_4/Practice_Extracting_a_Summary.ipynb) |
|**Approaches to Summary Extraction** ||
|**Review Summary Extraction** ||
|**Summary Extraction With LSA** ||
|**Practice Extracting a Summary With LSA** ||
|**Summary Extraction With PageRank** ||
|**Practice Extracting a Summary With PageRank** ||
|**Course Project, Part Three — Extracting Summary Sentences From Documents** ||
|**Module Wrap-up: Extract Summary Sentences From Documents** ||
|**Glossary** ||
|**Course Transcript** ||




CODE: Practice Extracting Keywords With RAKE
from rake_nltk import Rake
r = Rake(language="english", ranking_metric=Metric.DEGREE_TO_FREQUENCY_RATIO, max_length=3, min_length=1)
*** do this practice parts 
TextRank Algorithm for Keyword Extraction
CODE: Practice Extracting Keywords With TextRank
work with graph structures and the TextRank algorithm available through Python Keyphrase Extraction (PKE) library. 
use NetworkX to build, compute and manage graphs, but there are many other ways of storing and manipulating graph data types.
networkx as nx
graphs and Graph Storage
CODE: Practice Measuring Keyword Quality
Extract the top relevant keywords using PKE's TextRank algorithm
 extract the same number of top keywords with RAKE, another KEA.
Compare Using Jaccard Similarity
UDF StemPhrase() below takes a keyword, splits it into words, and re-joins the stemmed words into a phrase again. 
 compare these as sets of words instead,
extra practice: compare the  keyword lists produced by both Rake() and TextRank() using Jaccard Similarity.
Course Project, Part One — Identifying Keywords and Keyphrases From Text
takes an NLTK file ID and returns a Pandas DataFrame of RAKE-extracted keywords as indices and their scores as column values
measure the similarity between two documents based on their matching keywords and their aggregated scores.
takes a query file id (qfid) and computes its similarity metric with each inaugural speech
wrap up
first step of text summarization: keyword extraction. 
techniques for extracting keywords & phrases using the Rake() and TextRank() methods.
evaluating the quality of keywords produced by examining which keywords appeared repeatedly in the results of multiple keyword extraction algorithms.


Module  2 Identify Topics From Documents
topic modeling algorithms Latent Semantic Analysis (LSA) and Latent Dirichlet Allocation (LDA)
singular value decomposition, or SVD
NOTE: I really don’t get SVD particularly Eigenvectors (and Eigenvalues):
CODE: Practice With Basic Matrix Operations
Matrix Decomposition
Single value decomposition (SVD), in which a given matrix C is decomposed into matrices U, D, and VT
Given a matrix 𝐶, you want to find three special matrices, 𝑈,𝐷,𝑉, whose product is  𝐶
The NumPy function np.linalg.svd and scikit-learn function TruncatedSVD compute matrices U,D,V such that  U is a left eigenvector matrix, V is a right eigenvector matrix, and D is a diagonal matrix with eigenvalues on its diagonal. Eigenvectors are special vectors that are orthogonal and point to the directions of the largest variances (which are the corresponding eigenvalues in D). The SVD itself is: 𝐶=𝑈𝐷𝑉⊤
Therefore, in addition to finding the best axial tilt that produces the best eigenvectors, you also need to be able to determine how much signal each of the eigenvectors contain. To do this, you perform a matrix math exercise that results in a number, referred to as an eigenvalue, that quantitatively identifies how much of the predictive signal each eigenvector feature has relative to all of the other eigenvector features in the data.
SVD and Truncated SVD is very heavy, review 
CODE: Practice Approximating a Matrix With Truncated SVD
   '''Plots C, it's SVD components and their product CEst
    C: original matrix which we factorize
    U,D,Vt: left eigenvector matrix, diagonal matrix of eigenvalues, right eigenvector matrix
    CEst: matrix estimated with the product U @ D @ Vt
    figSize: dimensions of the overall plot panel
    nPrecision: number of decimals in matrices    '''
Latent Semantic Analysis (LSA) is an unsupervised learning technique used for topic modeling. 
Silhouette scoring is a commonly used metric that provides a ranged score of -1 to 1 that indicates how well a given document or term aligns with other objects in a given topic based on a distance metric such as Euclidean distance. 


Part Two of the Course Project
In this project, you will identify common topics among 59 presidential inaugural speeches using Latent Dirichlet Allocation (LDA) built on a term frequency-inverse document frequency (TF-IDF) document term matrix (DTM).


Latent Dirichlet Allocation, or LDA, is a Bayesian probabilistic model which computes the probability of a document being in one of the predefined key topics. The latent topics are not observed explicitly, but are assumed to follow a Dirichlet prior distribution. After initialization for each triplet of a document D, word W, and topic T, the algorithm computes the conditional probability of that topic given that document as the fraction of words in the document assigned to the topic. Then we compute the conditional probability of the word given that topic. It is a count of the word in the topic divided by the total frequency of the word. The product of these two probabilities is the probability of our assignment of the given word to the given topic. After a few iterations, the documents are decomposed into topics, and topics are decomposed into words. Since we typically do not know the true number of topics and events, we use hyperparameter search to try different numbers in search of the best value of some metric, either log likelihood score, we want that higher, or some lower value of perplexity score, or complexity of our model. 
Also note that the topics are notoriously difficult to interpret from the list of topic words. The interpretation is often subjective and requires a deep domain expertise to recognize and validate the computed topics. 
Let's look at LDA training, interpretation, and visualization with a package called pyLDAVis, which allows some interactivity with the plot and eases the interpretation and manipulation of topics. 
LDA here is coming from Scikit-learn. ++++++++++++++++++++


Module 3 Extract Summary Sentences From Documents
document summarization approaches, including Latent Semantic Analysis (LSA) and PageRank(). 
CODE Practice Extracting a Summary With NLTK
justg ranking sentences by the total word freq score of words in the sentence 








Course Project, Part Three — Extracting Summary Sentences From Documents


In this project, you will build a graph of sentences from a U.S. President's inaugural speech and apply the PageRank algorithm to rank the sentences by their importance score, referred to as 'Rank' herein. You will compute correlation using the Gramian matrix built from TF-IDF document term matrix (DTM). The more a given sentance is "correlated" with other sentences, the greater its importance.





mSimilarity = mDTM @ mDTM.T   # Gramian matrix = all pairwise dot products of sentence vectors


print(f'mSimilarity.shape = {mSimilarity.shape}')
def PlotMatrix(m:np.array) -> None:
    '''Function to plot a heatmap of a 2D Numpy array m'''
    plt.figure(figsize=(25, 4))
    sns.heatmap(pd.DataFrame(m).round(1), annot=True, cbar=False, cmap='Reds');
PlotMatrix(mSimilarity[:10,:30])








G = nx.from_numpy_array(mSimilarity)   # build similarity graph
G.name = 'Sentence Similarities'
print(nx.info(G))










More About: Special Matrix Types


A zero matrix is an additive identity, assuming matching dimensions. In this matrix all values are zeros but dimensions can vary.
An identity matrix is a multiplicative identity, assuming matching inner dimensions. This is a square matrix of varying dimensions with ones on diagonal and zeros otherwise.
A matrix of ones, where all values are ones.
A diagonal line or elements in a matrix includes all elements with matching row/column indices. So, in a matrix 
[
𝑎
𝑖𝑗
]
[aij], 
𝑎
𝑖𝑖
aii are diagonal elements, and other elements are off-diagonal.
A symmetric matrix where all values are the same about the diagonal line or elements.
A symmetric matrix must be square, i.e., have the same number of rows and columns.
A symmetric matrix has many amazing properties that ease operations on it, such as factorization into two matrices.
A diagonal matrix is a matrix with zeros off-diagonal — above and below the diagonal values. Diagonal values are those with equal indices, but the matrix itself does not need to be square.
Ex. Let 
𝐴:=[
𝑎
𝑖𝑗
]
2×4
A:=[aij]2×4, i.e., a matrix with two rows and four columns and elements 
𝑎
𝑖𝑗
aij. Then 
𝐴
A is diagonal, if 
𝑎
𝑖𝑗
=0,∀𝑖≠𝑗
aij=0,∀i≠j, i.e., for all values with non-equal indices being zeros. The zero matrix is also diagonal by this definition.


special type of matrix decomposition: Single value decomposition (SVD), in which a given matrix C is decomposed into matrices U, D, and VT. 








