# Class 5: Clustering Documents With Unsupervised Machine Learning

## Section 1: Use Metrics To Determine Text Similarity
| Lesson          |         Discription                                | Colab link    |
|-------------------|----------------------------------------------|------|
| **Module Intro**   | - become familiar with two major similarity measures: lexical and semantic similarity practice evaluating the similarity between words by computing the distance virus identification and spell-check problems using the Hamming and Levenshtein distance metrics    Measuring the similarity between words in this module will prepare you for the document categorization     |  |
|**Vid: pare Texts Using Similarity Metrics**|lexical similarity, which is based on syntax, structure, and content, is commonly used for autocomplete and spell check applications. Semantic similarity, by contrast, uses context to evaluate the similarity in meaning between words or documents.|
|**Compare Texts Using Similarity Metrics**|1 how to calculate **Jaccard similarity** and use a correlation function to compare two words. 2 practice applying Jaccard similarity to both characters and words.|
|**Code: Practice Comparing Texts Using Similarity Metrics**|Binary comparison 2 Jaccard similarity 3. Correlation 4 using ord() to convert to ASCII |[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matsunagateitoku/stdp-class/blob/main/4.1.1.Practice_Comparing_Texts.ipynb) |
|**Find Mismatches With Hamming Distance**|**Hamming distance** is another way to measure the distance between words, when working with two strings of equal length number of substitutions needed to make two strings equal|
|**Practice Finding Mismatches With Hamming Distance**|Hamming Distance=opposite of similarity. sequences of equal length, counts positions that corresponding characters are differen | [![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matsunagateitoku/stdp-class/blob/main/Practice_Finding_Mismatches.ipynb)|
|**Similarity Measures**|Quiz|
|**Comparing Sequences With Levenshtein Distance**|Levenshtein distance, counts the number of alterations needed to convert one string into another string using addition, deletion, or substitution| video
|**Comparing Sequences With Levenshtein Distance**|Levenshtein distance, counts the number of alterations needed to convert one string into another string using addition, deletion, or substitution| video
|**Compute Levenshtein Distance**| text explaintion of Levenshtein|
|**Practice Comparing Sequences With Levenshtein Distance**|compute Levenshtein distance using dynamic programming and how to model the results in a 2D matrix|[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matsunagateitoku/stdp-class/blob/main/class_5/Practice_Comparing_Sequences.ipynb)|
|**Use Levenshtein Distance To Autocorrect Text**|built-in Python-Levenshtein package is very fast bsed on C & NLTK implementation of edit distance  timeit function in conjunction with the Brown Corpus dictionary to offer recommendations for reducing the total number of computations and runtime|
|**Practice Using Levenshtein Distance To Autocorrect Text**||[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matsunagateitoku/stdp-class/blob/main/class_5/Practice_Using_Levenshtein.ipynb)|
|**Review: Similarity and Distances Metrics**||[PDF](class_5/cis575_tool-similarity-and-distance-metrics.pdf).
|**Course Project, Part One — Using Metrics To Determine Text Similarity**|1. computes Jaccard similarity 2. ompute Jaccard similarity scores for SsQry set and each presidential speech in NLTK 3. Hamming distance;; Marker(), which takes two same-length strings, query sQry and target sTgt, and computes returns a "marked" string sTgt where the characters matching to corresponding characters in sQry are replaced with _. 4 ankHD(), which takes two strings: query sQry and target sTgt, of varying lengths. Then, for each substring in sTgt with length equal to len(sQry), compute the Hamming distance and the corresponding marked string.|[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matsunagateitoku/stdp-class/blob/main/class_5/Using_Metrics.ipynb)|
|**Module Wrap-up: Use Metrics To Determine Text Similarity**|In this module, you gained a familiarity with several approaches and metrics used to measure the distance between words and documents. You began by comparing two words using Jaccard similarity, and then applied the Levenshtein and Hamming distance metrics to more complex problems. Finally, you built functions computing Jaccard similarity and Hamming distance, which can be used to evaluate the similarity between speeches and rank viral samples. |

## Section 2: Use Metrics To Determine Text Similarity
| Lesson          |         Discription                                | Colab link    |
|-------------------|----------------------------------------------|------|
|**Module Introduction**|Perform Hierarchical Clustering on Sentence Embeddings To Group Similar Texts|In this module, you will be introduced to **clustering**, an unsupervised method you will use to categorize documents based on their similarities. However, before you can practice clustering techniques in code, you will first examine how to **represent documents as numerical vectors**. Then, you will practice using **hierarchical clustering** to categorize movies in a dataset according to genre. Finally, you will use **dendrograms, or tree diagrams**, to visually evaluate the quality of your clustering. You will also evaluate your clustering quality using several quantitative
|**Embed Sentences Into Vectors**|When representing and evaluating many sentences in code, one helpful tool is Sentence Bidirectional Encoder Representations from Transformers (SBERT), which encodes entire sentences or documents as numerical vectors. SBERT can be used for a variety of NLP tasks including language translation and meaningful sentence embeddings.uses the sentence-transformers implementation of SBERT to encode 15 famous quotes. He then evaluates the quotes’ correlation coefficients and cosine similarities using a pre-trained model.|video
|**Practice Embedding Sentences Into Vectors**|Sentence BERT or SBERT incredibly easy to use and is only a few hundred megabytes compared to 8 gigabyte fastText model. Sentence BERT does not store static word vectors. Instead each word vector, if still needed, is generated dynamically from its semantic context. 1. SBERT in code--Sentence Transformers Package. 2. Encode function 3 loading a dictionary with famous quotes 4. Encode function to create 15 different 768-dimensional vectors.5) compute the correlation  [minus 1 and 1]. Next we can query the list of sentences that we have based on their vector representation. And this does not need to have the same-- we do not need to have the same words as we've had previously with TFIDF type of query or document term matrix type of query. Here the semantic meaning is extracted from the sentence we have, language and sight being the query sentence. And that is represented in the next line with a vector, again, 768 dimensional, which we use to compute every one of these cosine similarities for every one sentence.paraphrase-distilroberta-base-v (1330 MB) model. In this activity, you will use a smaller model, paraphrase-albert-small-v2 (~50 MB).  SBERT.encode().  evaluate correlations (i.e., pairwise measures of linear dependence), which can be easily computed using the .corr() method on the transposed dataframe.  **cosine similarity**| [![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matsunagateitoku/stdp-class/blob/main/class_5/Practice_Comparing_Sequences.ipynb)|
|**Compute Similarity of Sentence Embeddings To Find Similar Movies**||video
|**Practice Computing Similarity of Sentence Embeddings To Find Similar Movies**||
|**Methods for Clustering**||
|**Hierarchical Clustering of Movies Based on Their Descriptions**||
|**Practice Hierarchical Clustering of Movies Based on Their Descriptions**||
|**The Anatomy of a Dendrogram**||
|**Interpreting a Dendrogram**||
|**Build Dendrograms Using Different Linkages**||
|**Practice Building Dendrograms Using Different Linkages**||
|**Identify Linkages in Dendrograms**||
|**Measure Clustering Performance**||
|**Practice Measuring Clustering Performance**||
|**Course Project, Part Two — Performing Hierarchical Clustering on Sentence Embeddings To Group Similar Texts**||
|**Module Wrap-up: Perform Hierarchical Clustering on Sentence Embeddings To Group Similar Texts**||

## Section 3: Perform K-Means Clustering on Sentence Embeddings To Group Similar Texts
| Lesson          |         Discription                                | Colab link    |
|-------------------|----------------------------------------------|------|
|**Module Introduction: Perform K-Means Clustering on Sentence Embeddings To Group Similar Texts**||
|**Calculate Centroids and Medoids To Find a Representative Data Point**||
|**Practice Calculating Centroids and Medoids To Find a Representative Data Point**||
|**Compute the Medoid To Find a Representative Movie**||
|**Practice Computing the Medoid To Find a Representative Movie**||
|**Clustering Movies With K-Means and Evaluating Performance**||
|**Practice Clustering Movies With K-Means and Evaluating Performance**||
|**Hierarchical and K-Means Clustering**||
|**Course Project, Part Three — Perform K-Means Clustering on Sentence Embeddings To Group Similar Texts**||
|**Module Wrap-up: Perform K-Means Clustering on Sentence Embeddings To Group Similar Texts**||
|**Glossary**||
|**Course Transcript**||


