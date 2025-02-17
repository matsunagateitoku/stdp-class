
# Class 2: Transforming Text Into Numeric Vectors
## Section One: Create Sparse Document Vectors From Counters

- In this module, you will construct and evaluate two kinds of vectors:
 - the **document term matrix (DTM)**,
 - and the **term frequency-inverse document frequency (TF-IDF) matrix**, a special kind of DTM. 
- You will also work closely with the scikitlearn package, which contains the tools to quickly compute a basic DTM and a more sophisticated TF-IDF DTM.
- You can use scikit-learn's CountVectorizer() object to build a document term matrix (DTM).
- CountVectorizer takes a list of string sentences, parses them into word tokens, builds a vocabulary from the list of unique words in the corpus of sentences, and counts each word in each sentence. The vocabulary words become columns and sentences (or documents) become rows. In addition, this object has many useful arguments and methods for convenient preprocessing, such as stopword removal, lower casing, etc. It can also be customized with a more sophisticated preprocessing tool, such as a lemmatizer. The default word tokenizer pattern is the argument token_pattern='(?u)\b\w\w+\b', which assumes words are made up of multiple word characters, but can be modified for other word separators.
 - cv = CountVectorizer(stop_words='english', lowercase=True) # create object to count pre-processed words
Now that the object is initialized, you can pass the list of sentence strings to the fit_transform() method. Nearly all modeling objects in scikit-learn have fit, transform, and fit_transform methods. Here, fit applies preprocessing, parses words, and learns the vocabulary. Once the vocabulary is learned on a set of documents, you can apply the same set of docs or another set of docs to be transformed to a DTM. Note that the CountVectorizer only "knows" the vocabulary it was fitted on. So, if the transformed set of documents has new words, they will be ignored. For this reason, you should try to have a reasonably comprehensive vocabulary when you apply CountVectorizer. Otherwise, you'll need to refit the CountVectorizer object as you add documents.
The fit_transform applies both methods at once to the same set of documents (or quotes, in this case). A transformation returns a SciPy sparse matrix, which stores indices and values of non-zero elements in a matrix. If the matrix is highly sparse (as is often the case with DTMs), then such data structure is very effective in saving memory. It is interesting that a larger vocabulary will not use any more memory for storing counts in a sparse matrix but will use some memory to store a larger vocabulary list itself.
DT_smatrix = cv.fit_transform(LsQuote)     # create document-term matrix (DTM) in sparse format
DT_smatrix
Part 1: Operations on Vectors
Vectors are a point in Cartesian space
they all start from the origin 
We can perform operations on them add, sub, etc. 
Part 2: Operations on Matrices
NumPy Library
np.array() object
.T attribute
SciPy Library
csr_matrix() object
csr_matrix().toarray() method
dense and sparse matrix
The rows in a matrix are sometimes referred to as "observations" while the columns are sometimes referred to as "features," "variables," or "dimensions."
For example, a matrix with three rows and five columns would be referred to as a 3x5 matrix. This matrix:  would also be referred to as being a data set with five dimensions
from scipy.sparse import csr_matrix
vectorization: print(Y[Y[:,0]==0])
Dot Product
Part 3: Operations on Pandas DataFrames
pandas used numpy under the hood. slower but more user friendly 
Part 4: Document-Term Matrix (DTM)
Scikit-Learn Library
CounterVectorizer object
.fit_transform learns vocabulary, returns DTM
.get_feature_names retrieves vocabulary
SciPy Library
csr_matrix().toarray() method  converts to dense matrix format
Seaborn Library
heatmap plots colored grid
bag-of-words type of model
DTMs store documents on the rows of a matrix and the frequency of specific words in each document on the columns. Since both rows and columns of a matrix can be manipulated as vectors, storing matrices in this manner opens up a wide range of analytic possibilities.
This matrix is commonly constructed with CountVectorizer object of Scikit-Learn library.
cv = CountVectorizer(stop_words='english', lowercase=True) # create object to count pre-processed words
Now that the object is initialized, you can pass the list of sentence strings to the fit_transform() method. Nearly all modeling objects in scikit-learn have fit, transform, and fit_transform methods. Here, fit applies preprocessing, parses words, and learns the vocabulary. Once the vocabulary is learned on a set of documents, you can apply the same set of docs or another set of docs to be transformed to a DTM. Note that the CountVectorizer only "knows" the vocabulary it was fitted on. So, if the transformed set of documents has new words, they will be ignored. For this reason, you should try to have a reasonably comprehensive vocabulary when you apply CountVectorizer. Otherwise, you'll need to refit the CountVectorizer object as you add documents.
The fit_transform applies both methods at once to the same set of documents (or quotes, in this case).
A transformation returns a SciPy sparse matrix, which stores indices and values of non-zero elements in a matrix. If the matrix is highly sparse (as is often the case with DTMs), then such data structure is very effective in saving memory. It is interesting that a larger vocabulary will not use any more memory for storing counts in a sparse matrix but will use some memory to store a larger vocabulary list itself.
DT_smatrix = cv.fit_transform(LsQuote)     # create document-term matrix (DTM) in sparse format
dfDT = pd.DataFrame(DT_smatrix.toarray(), index=LsQuote, columns=LsVocab)
make a pandas df from strings & list of vocab 
Seaborn
ax = sns.heatmap(dfDT, annot=True, cmap='coolwarm', cbar=False);
tmp = ax.set_title('Counts Matrix');
Part 5: Term Frequency-Inverse Document Frequency (TF-IDF)
TF-IDF is also a document-term matrix, or DTM. In that sense, it's still is troubled by high sparsity or lots of zeros and large vocabulary. Now, the TF-IDF does not remove words from vocabulary, but we can detect stopwords by thresholding the weights in TF-IDF matrix.
scikitlearn Library
tfidfTransformer() object
.fit_transform()
get_feature_names()
smTFIDF1 = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True).fit_transform(smDT)
tv = TfidfVectorizer()
smTFIDF2 = tv.fit_transform(LsQuotes[:-1]) # fit and transform only 14 quotes, return sparse matrix format


dfTFIDF2 = pd.DataFrame(smTFIDF2.toarray(), columns=tv.get_feature_names())
Part 6: Thresholding a TF-IDF Matrix
fit_transform()
Use the object's fit_transform() method, which fits and transforms the documents:
In fitting, the object learns the new vocabulary and inverse document frequency (IDF).
In transforming, the object uses learned vocabulary and IDF to transform documents.
The method returns a document–term matrix (DTM) with TF–IDF weights (not counts) in SciPy's sparse matrix format.
Wrap up: You often want to compare documents by expressing their degree of similarity, and one relatively simple way of achieving this is to transform text into numeric vectors, then count (or somehow measure) the overlapping tokens. In this module, we constructed and evaluated two kinds of vectors: the document term matrix (DTM) and the term frequency-inverse document frequency (TF-IDF) matrix. We did this by working closely with the scikitlearn package.
Next, we will extend this idea to represent words as dense vectors, which we can aggregate to represent sentences as dense vectors; this allows us to measure similarity between any two sentences, regardless of whether they have the same words or not. In fact, we can even measure similarity between words and sentences and in different languages. All that is possible when we convert our text to mathematical vector spaces.


## Note: I think there are ways we could get a lot more value from TF-IDF than word clouds. 
- In particular TF-IDF ignores the words that appear in all the docs as stopwords, but those words can be very helpful in understanding what the result set is about.
- Maybe compare the autogenerated stop word list with a preset stop word list?
- How does this relate to topic modeling?
- Would a matrix display be better

## Section Two: Create Dense Doc Vectors From Pre-Trained Models
Word2Vec model  does not count words in a given document but uses predictive deep learning on very large corpora to create representations of words that capture contextual and semantic similarity. 
The resulting numeric vectors are dense rather than sparse and consequently have much lower dimensionality.
Part 1: Dense Matrices and Word2Vec
Sparse vector representations are computationally inefficient. Recently, a dense representation of text have been developed where text is encoded into 50 to 500 dimensions instead of hundreds of thousands of dimensions required in sparse representation of DTM and TF-IDF. 
Much of it started around 2013 when Thomas Mikolov, employed by Google at that time, was able to train a Word2Vec model on almost 2 billion words. This was a simple neural network which learned associations between neighboring words in its hidden layer of coefficients. These coefficients were precisely the numeric vectors for words fed through the model. It was surprising to find out that the model captured syntactic and semantic relationships among words into dimensions of the vectors. These relationships relate countries to their capitals, currencies, nationalities. They also associate various forms of verbs, nouns, adjectives, and other morphological structures or derivatives. It understands synonyms, antonyms, comparative word forms like "great" or "greater," and much more. The biggest limitations of Word2Vec model were slow sequential training, limited vocabulary, and static vectors regardless of the word's meaning in a sentence. For example, if a word, such as my last name, was not part of the training corpus, there would be no way to get a vector for that word. Also the word "jaguar," which would be defined by the same trend vector regardless of the context it appears in but it could mean car in one sentence and it could mean an animal in another. However, both senses — car and animal — of this "jaguar" word would be captured by a single vector. And there would be dimensions that are presenting more of an animalness of that word and more of a carness of that word. Since then, the models have improved dramatically. But for now, let's see how the word vectors are loaded with Gensim package in the code. We need to download the Gensim library and — Gensim library trained Word2Vec model, and this could be brought in from the GitHub, from the RaRe Technologies. It's the smallest models that they have of 50-dimensional vectors. We can even see the size of this model, which is roughly 17 megabytes. It's a .gz or zipped format, and we can uncompress it, actually look inside of this trained model. It's just a text file. It's a text file that indicates on the first line how many word vectors there are in the file — there are 400,000 — and these are 50-dimensional words. This word "the" that appears in the first line is the article word that has this following representation of numbers or the vector, which is 50-dimensional. So if we scroll all the way to the right, this is the 50th — or 49th if we start from 0 position; all the coefficients that was trained in the neural network in the Word2Vec — or actually GloVe model, which is very similar. There are 400,000 of these if we scroll all the way to the bottom; 400 001 was the first line. And we can search for a particular word, like "cornell," and find its vector is a vector starting with negative 0.929. There are other words that are also here and are represented by a vector. Loading this model or this 70-megabyte file — it's actually unzipped 170 megabytes — takes a little bit of time, so we preloaded this for you. When you print it out, it just says it's a Gensim model of keyed vectors. It's basically a dictionary with additional features or attributes and methods that we can use, one of which is extraction of a vector for a particular given word. Word_vec method will extract a vector negative 0.929 that we just saw in a text file. All it does, it just pulls that line out of the text file. We can actually refer to the object wv, the model object, as we refer to a typical dictionary and extract the vector coefficients in the same way. There is another attribute that is convenient to use; you can count number of coefficients here and derive 50, but you can also extract vector_size and that will tell you the dimensionality of a vector
Part 2: Practice With Dense Matrices and Word2Vec
Gensim-trained Word2Vec model 'glove-wiki-gigaword-50.gz'. This is the smallest package in the library and includes about 400,000 words.
contains a matrix of weights, where each line is a word vector with the word itself starting the line
It contains 50 values somewhere between -2 and +2.
All values are floats with 32-bit precision.
There are no zeros, so it can be considered a dense vector (a vector comprised of mostly non-zero values).
Each value represents a dimension but is not necessarily interpretable by humans.
Large (in magnitude) values, such as 1.73 or -1.3797, may relate to education, university, academia, technology, or some other broad category in which Cornell has great presence.
Part 3: Word2Vec Vectors
gensim Library
KeyedVectors object
.load_word2vec_format()
.vectors()
.vocab
.keys()
Gensim package is rich with functionality, not just for Word2Vec but for several other word-embedding models such as GloVe and FastText. 
It offers a standardized interface for extracting Word vectors, computing similarities, searching for vocabulary — or searching for words in vocabulary, and much more. 
GloVe model, which is very similar to Word2Vec and it's trained on Wikipedia. This is the smallest model that is out there, typically found with 50-dimensional vectors and just 400,000 lowercased words. We are loading the model into the wv object. This is the returned result; it's loaded as the keyed vector, so basically a dictionary with additional attributes and methods. Notice that because the words in this dictionary — or the keys in this dictionary are all lowercased, if you tried to search for uppercase word and extract its vector, you will get an error message; we're capturing this error message with a try-and-except to make sure it doesn't crush the code. But the error is essentially "word 'cornell' is not in vocabulary." The extraction could be improved a little bit where we can wrap it into your function, and if the word is in the dictionary, we would extract the word, just like we would extract value based on the key from any other dictionary. Or if it's not there, we could return the zero-vector, 50-dimensional vector, and it can be as easy as taking an arbitrary vector — let's say the first one, the zero vector — and multiplying all its coefficients by 0, basically zeroing them out. You can also use NumPy 0 to create a 50-dimensional vector of zeros. The type command will tell us what the type of the wv.vocab attribute is, and it's a dictionary. We can extract its keys. The keys are precisely the vocabulary in the order that we've seen them in a text file, starting with "the," then a comma, period, and so on; 400,000 words that we put double spaces in between. We can also look at the vectors that represent this vocabulary. Every row here, just like in the text file, represents a word. This first row is for the word "the," and so on. There are 50 dimensions and 50 columns for each vector. No one really knows what these columns represent; it's an open area of research. But they are helpful in identifying the similarities between words, as we will demonstrate later. The NumPy array could be nicely wrapped into a DataFrame where each row will get a word identifier or identification or label, and the columns will be numbered from 0 to 49 to indicate 50-dimensional vectors. There are 400,000 of this; they're nicely formatted, and it's a bit difficult to tell where the numbers relate and where they do not. We can further just focus on a few words, such as "cornell," "graduate," "university," "professor," "jazz," and extract just the vectors for these words and present them with the Seaborn heatmap in this nicely colored background table. Now we can see that the words, such as "cornell, "graduate," "university," and "professor," have high coefficients in some of the columns, and these coefficients are different from "jazz," "music," and "wave." So the columns 37 and 38 stand out. This has some dimension or dimensions that relate to university or relate to education or relate to Cornell and they are highlighting that fact by values different from the values of the music-related concepts. Some music-related dimensions might be column 9, where the coefficients are high for the music words. There are some other dimensions which may indicate different properties or different meanings, different senses for these different words, and it's very interesting to investigate.


Part 4: Arithmetic With Word Vectors
gensim Library
KeyedVectors object
.load_word2vec_format()
.most_similar()
.most_similar_to_given()
.similarity()
scikitlearn Library
PCA() object
.fit_transform()
king + woman - man? As you'll see in this video, it is queen
We're now ready to apply vector algebra we learned earlier to word vectors. We do not need to explicitly compute — add or subtract vectors — because Gensim library provides this functionality for us. But we will explore how it's done — how it can be done and what other methods are available for us to operate on vectors. Let's look at the code. Here we have the GloVe word to vector model loaded. Again, it has 400,000 words lowercase and 50-dimensional vectors representing each one of them. We can load it as usual into wv object in Python. This is mostly a dictionary — a Python dictionary with some additional methods and attributes. One of these methods is most_similar, and it'll take a word, "cornell," and return topn — top number —10; in this case, words that are most similar. By "most similar," we mean they have the greatest cosine similarity. We'll learn about this later; it's a number between negative 1 and 1 indicating how similar the two words are. It's a pairwise metric, so it needs to take two words; one of them is "cornell" and the other one is — every word in the vocabulary is tried; every one of the 400,000 words is tried in 400,000 computations. Then they're sorted or ordered by this similarity and the top 10 are returned. We can limit this computation to just a few selected words, and if we want to know which of the words in this given list is most similar to the word "cornell," we can do the most_similar_to_given and provide a list of words. The word "professor" is returned; we don't see the similarity score that was used to bring this word up to top word on that list. But here's a way where we can actually iterate over the words in the list we have, list of string words, via list comprehension, and compute similarity on the "cornell" and each one of those words. At the end, we apply a sorted method on the similarities that is returned as part of each one of these tuples, and we order by the similarity to "cornell." The "cornell" and "cornell" have a similarity of 1, of course, but the next one up is "professor," as we've seen earlier. There's a very nice way to display a 50-dimensional vector in two-dimensional plane. For that, we use something called principal component analysis, which is the projection of this word vector's multidimensional space to a two-dimensional surface. We'll learn about PCA later, but for now, what it does, it allows us to plot the word vectors, or their two-dimensional presentation, and evaluate how close the words are based on the distance between their vectors. So the education-related or university-related vectors are all clustered together, which makes sense. Same goes for the "music," "guitar," "concert." Notice that in the vertical dimension, the university-related vectors are similar to music. If you project all of this to the left on this vertical line, they are similar, but they are different in this horizontal space. We can do more. The most_similar method takes positive as a list of words that we want to — for which we want to add their vectors together. And the "negative," it's not a negative word; it's a negative operation. So we will take the "man" vector or vector representing "man" and subtract it from the vector that arises from adding vectors for "king" and "woman." In some sense, this is an operation of "king" vector minus "man" vector plus a "woman" vector. The resulting vector may not be in the trained model. But what we will do next is extract the word that has the closest vector to the resulting vector and compute the similarity. So the word "queen," which makes sense, has similarity of 0.85. We can do more. "Actor" plus "woman" minus "man" would give us actress. Another way to state it is Obama to Clinton is as Reagan to Bush is found to be the closest result of this comparison. Finally, we can find the word that is least related, related to the list of words that we have. "Breakfast," "dinner," and "lunch" are all similar in a sense, but "milk" doesn't seem to fit in as much. We can split this sentence and use the doesn't_match method to find the word that is least sensible to this list.
Part 5: Work With Word Vector Arithmetic
Using Cosine Similarity to Identify Similar Words
One metric for identifying semantically similar words is cosine similarity. Cosine similarity values range from -1 to 1, with the score of 1 representing perfect similarity (such as the similarity of a word with itself) and a score of -1 representing perfect dissimilarity.
From the Gensim library, the most_similar() method identifies the top 𝑛 most similar words for a given word. The method converts the given word into a word vector and then computes cosine similarities of that vector with each of the 400K vectors in the model. Similarities are then ordered in decreasing order and the highest 
𝑛
𝑛 cosine similarity scores along with the associated word are returned. You can choose how many of the most similar words to return with the topn argument.

Part 6: How Word2Vec Is Trained
To understand the capabilities and limitations of Word2Vec, you need to understand how the model is trained. In this video, Professor Melnikov discusses the two algorithms that were originally used to train this model, walks through model hyperparameters, and discusses benchmarks for evaluating results
the biggest challenge in training Word2Vec model was not the neural network architecture; it was the difficulty of scaling the training of billions of words sequentially as they are fed through the model one by one. This is a notoriously very slow situation. 
Several novel and clever sampling methods made it possible. Two algorithms were considered for the training — for the sampling. Each one of them required some sort of sliding window that would go over the sequence of words. In this window, say of size 5, the center word is called a target word and the surrounding words are called context words. In the continuous bag of words, or CBOW model, the algorithm tries to predict the target word from the surrounding words as soon as there is a relationship between words that are in the window. In the skip-gram algorithm, the model tries to do the opposite: It predicts the surrounding or the context words, given the target word, which seems to be a more difficult problem but turns out to give better results and trains faster. Whenever the model makes an incorrect prediction, the model coefficients are adjusted accordingly so as to lower the error the next time these words appear in the window. Once the training is finished, the trained coefficients from this model — either one of these two algorithms — are precisely the word vectors we use. Before the training begins, the scientist needs to specify the hyperparameters for the model. They are also called tuning parameters. They include the window size or the word vector dimensionality or the learning rate with which the coefficients are updated or the number of iterations over the input corpus or many other ones. These are nontrivial and typically require experimentation to identify suitable parameters. For example, nobody really knows whether a sliding window of size 5 or 50 or 500 needs to be used or which one is more appropriate, or whether the vector of size 10, or of length 100 or 1,000 is better suited for a particular corpus. Some intuition, however, can be drawn from the size and complexity of the corpus itself. For example, vectors might require a double length if two languages are considered for training. This, of course, depends on some intuition that you might develop with a single-language vector length. The others also need to decide on the preprocessing and corpus cleaning methods. It might lowercase or leave the case intact, or they might remove punctuation or leave the punctuation there or do any de-accenting and so on. Finally, the model results are evaluated and compared against the latest benchmarks in a particular NLP task, such as question-and-answer task. For example, quantitative, semantic, and syntactic accuracy needs to be evaluated against other leading neural network and classical NLP models to determine whether the model has achieved its potential and can be used in the real world.  
Part 7: The Long Tail Problem for Out-of-Vocabulary Words
Libraries/Methods
collections Library
Counter() object
.most_common()
pandas Library
DataFrame() object
.set_index()
.freq
.count_values()
One way to reduce the training size of a Word2Vec model is to remove infrequent words that have insufficient information for learning their vector embeddings. Other approaches, such as lowercasing and lemmatization, can help increase count frequencies for words that appear in different casing or in different morphological forms; e.g., "Happy," "happy," "happiness."
hink about the distribution of words in a document. If we order words by their frequency, we might observe the head of the distribution with 20 percent of vocabulary making up 80 percent of text. The long tail captures the remaining 80 percent of vocabulary with rare appearances. Infrequent words express few associations with other words and result in poor quality word vectors. We might drop rare words out of vocabulary and save some space in Word2Vec model. For example, any word with frequency less than six might be dropped out due to insufficient information about its co-occurrence with other words. Let's build this distribution in our code. We are loading Gutenberg's "Alice in Wonderland" and looking at just the top 20 tokens in the text in the order that they appear. "Alice's Adventures in Wonderland" by Lewis Carroll, 1865. We can use the collections.Counter object with the list passed to it to get the most common — or the distribution of the words starting with the most frequent and in decreasing order of frequency. So the period appears almost 2,000 times followed by single quote and then the word "the," which would probably be removed as a stopword, and and so on. We can look at the same information in the DataFrame, which is little bit nicer, more presentable, and we can scroll all the way to the right and look at the tail of this distribution with the word "THEIR" and "happy" appearing just once in the whole text. "Their," of course, might appear more times in a different capitalization. For lowercase, our words — "THEIR" will be collapsed with another "their" and the count will go up. So the word vector would be more meaningful. But the word "happy" is problematic. Unless there was a — "happy" was upper capitalization, like a title word, this is by the long tail that we do not want to have in the training of word vectors. Here's how the distribution looks like, where most of the words that we have — almost 3,000 of them — are appearing only once in the text. Just a few words appear an enormous amount of times; almost 2,000. Let's do a bit of preprocessing. What we have here is a list comprehension where we iterate through every single word in the Ls, in list of words, that we pulled out of the "Alice in Wonderland" text. We're only keeping the ones that are longer than three characters and the ones that have only letters in the word. Keep that in mind because some of the words with dashes will be dropped out and they might be legitimate words. We're lowercasing and placing everything back into LsWords2, and these are the first 20 words that we're observing all lowercase and much cleaner than the previous set. We are further creating the similar object, a list of tuples with a word from the vocabulary. It's only a unique version of that word and the frequencies that is corresponding to that word in the document. So the word "said" appears 462 times, "alice" appears almost 400 times — this is lowercased — and so on. When presented in the DataFrame format with the word and frequency as the indices of these two rows and the first row word is set as an index — so it's a little bit nicer and easier to observe — we have almost 2,500 words; fewer than 3,000 that we had before. The word "happy" is still here, so there is no other capitalized version of the word "happy." There might be a word "happiness," but those are two different words that would probably benefit from lemmatization. This is still long tail and this is still problematic. Let's see how many words appear at least six times. We have this masking or marking vector which marks old words that appear more than six times. We can actually use that in the picture and put a threshold of word or words that are most frequent and less frequent with five or fewer appearances. It's about 20/80; just like I've said before, 20 percent of words appear six times or more. This is what we would consider a head of this distribution and the rest is the tail. This tail is still problematic, but if we have a larger text like the Wikipedia corpus or if we add other books, the word "happy" would appear more frequently and would have a more meaningful vector because it appears in different contexts and we can get — or we can learn its meaning or its sense. Here the two words that are problematic for this particular training corpus: "happy" appears only once and the word "sad" appears no times whatsoever. This would be out of vocabulary, or we would call them OOV; out-of-vocabulary words.


Part 8: Address the Long Tail Problem
In the previous video, Professor Melnikov introduced the problem of long tail distribution of word counts in NLP. In the "Review" section of this ungraded coding exercise, you will use these techniques as Professor Melnikov presented them in the video. In the "Optional Practice" section of this exercise, you will explore word frequency distributions.


Part 9: Generating Subwords With FastText
Libraries/Methods
nltk Library
nltk.ngrams()
As you discovered in the previous video, infrequent words create a real problem in the original Word2Vec models. The FastText model is a newer type of Word2Vec model that can assign a vector embedding to almost any word by training on subwords (i.e., n-grams generated from a character sliding window) of words. The vectors that represent these subwords can then be aggregated to form the vector of the original word or be combined with other subwords to create new words and vector representations.
Follow along as Professor Melnikov discusses how the FastText model works, the challenges it overcomes, and how it changes the word frequency distribution.
Long tail distributions are problematic in NLP. However, we might try to leverage the commonality among words. Many words in vocabulary have common word parts, such as subwords or n-grams, which include word roots and affixes. This words should be similar in some way. For example, if we train a vector for "Cornell," then it should be similar to the vector for "eCornell," which might be absent from the training corpus. If we train a vector for "learn," "ing," and "ed," then we can sum or average these vectors to derive vectors for "learning" or "learned." We can even derive a reasonable sum vector for the phrase "eCornell learning." This is precisely what FastText model does: It decomposes words into all possible 3-grams, 4-grams, 5-grams, and 6-grams of characters, and trains vectors for n-grams as well as the word itself. This allows recovering average vector for nearly any word of three or more characters, even if the word was never part of the original training vocabulary; so-called out-of-vocabulary words. it can have a sensible word vector presentation. For example, a query for a word most similar to my last name, Melnikov, yields other Russian last names and concepts. Let's evaluate long tail for a distribution of subword frequencies in code. We are basically partitioning a word "alice" through this function into 3-grams. As a result, we get a list of substrings that we can generate from word "alice"; we can pass any other word into it. There's a function in NLTKs which will do just that, except it returns a generator. Generator can only be passed over once. It's not a list; you can only go from start to end one time. You can wrap it into a list and then actually observe the contents of the generator. They are a list of tuples of single characters which can be joined together to generate the same result we've seen before, so 3-grams for the word "alice." We can then use it in a function that will take a word and will partition it into n-grams starting from the nmin and ending with nmax. This will return several different lists for different calls of the function with all possible n-grams in that range. The way it does it, it's just a loop that iterates for the n-values from nmin to nmax and calls the function, which is fairly fast because it's internally built in with the generator, which are fast components of Python. There's another wrapper function for that, which will take a list of words and generate n-grams for that list and pack it all as a single list of all n-grams for those words. This now can be used for this "Alice in Wonderland" document or text, book, which we clean a little bit by removing all words that are too short; maybe fewer than three characters. We are getting rid of anything that contains dashes, underscores, numbers, punctuations, and we'll lowercase all the words. Now we have this list of the original words of about 14,000 words and a list of all the n-grams generated from those words, which is about 117,000. They contain duplicates so if you want to get rid of the duplicates and look at the vocabulary of words and vocabulary of n-grams or the subwords, the subwords is pretty large, but that is not a problem here. What's important is that the subwords, even though there are many more of them, are now appearing more frequently in the text. We can have a subword appearing more than six times, and that's meaningful for a training of a vector in a FastText model. Here's a distribution of these subwords. We can see that previously we had 20 percent of words with frequency six or higher. Now we're having 28 percent of subwords in the head of the distribution. So the long tail has shrunk in comparison or in a percentage basis and we can have — these subwords have meaningful vectors that can add up into a word vector that the subwords came from.


++++++++++++++++++++++
## Part Two of the Course Project
- Goals: complete a set of functions that retrieve word vectors from a Word2Vec model, process the model's vocabulary to work better with similarity analyses, and then use these functions to analyze similarity of pairs and groups of words. As you use these functions, you will work with the glove-wiki-gigaword-50 pre-trained Word2Vec model that you've worked with in this module.
- Begin by loading the required libraries and printing the versions of NLTK, Gensim, and NumPy using their __version__ attribute.
- Note: Since word-embedding models are a rapidly changing area of NLP, changes in library versions may break older code. Pay attention to library versions and, as always, carefully read error messages. We will note where the functionality diverges from that demonstrated in the videos and provide alternative methods you can use to complete the task.


Measure Similarity Between Document Vectors
So far in this course, you have 
created both sparse and dense document vectors, 
and you compared different models to examine the size of the vectors and 
observe how dense vectors capture semantic information. 
To express the similarity between document vectors, we can use several different metrics that result in a numeric similarity score.
Higher similarity measures imply greater similarity between paired vectors or objects. 
Metrics can be standardized in different ways to ease their interpretation. 
For example, dot product is not standardized and can result in any real value. 
Cosine similarity, however, is standardized to be in [-1, 1] interval, which makes it far easier to understand. Two vectors with a cosine similarity 1 are perfectly similar and two vectors with a cosine similarity of -1 are perfectly dissimilar. 
Part 1: Similarity
Similarity is a numeric measure of likeness among two objects. In NLP, we compute similarity among words, paragraphs, documents
No similarity metric is perfect, but each has its own purpose and advantages
Typically, we use them to identify related documents; to query related relevant web pages; to compare viral DNA strings; to find synonyms, antonyms, words that rhyme; and much more. 
A distance is opposite to similarity. Higher distance implies lower similarity and vice versa.
To improve interpretation of similarity, we often standardize it to be in some finite range or to be a fraction of some base number. For example, correlation and cosine similarity are in the interval minus 1 to 1. 
While Jaccard similarity and Hamming distance can be scaled to be in a range of 0 to 1. Thus, given the similarity of 0.95, we know the documents are highly similar. 
On the other hand, non-standardized similarities, such as dot product or covariance, are preferred in the algorithms because of simpler and faster computation but are less interpretable to humans. 
Part 2: Similarity and Distance Metrics
Similarity is a numeric measure of likeness among two objects. In NLP, we compute similarity among words, paragraphs, documents, or any two sequence of elements, such as DNA sequences comprised of ordered nucleotides, ACTG, and so on. No similarity metric is perfect, but each has its own purpose and advantages. Typically, we use them to identify related documents; to query related relevant web pages; to compare viral DNA strings; to find synonyms, antonyms, words that rhyme; and much more. A distance is opposite to similarity. Higher distance implies lower similarity and vice versa. Often, we can derive one from another. To improve interpretation of similarity, we often standardize it to be in some finite range or to be a fraction of some base number. For example, correlation and cosine similarity are in the interval minus 1 to 1. While Jaccard similarity and Hamming distance can be scaled to be in a range of 0 to 1. Thus, given the similarity of 0.95, we know the documents are highly similar. On the other hand, non-standardized similarities, such as dot product or covariance, are preferred in the algorithms because of simpler and faster computation but are less interpretable to humans. Similarity metrics vary in the way they detect pattern differences and in the inputs they take. Binary similarity is the most generic and returns True or 1 if two input objects are matching exactly and False or 0 otherwise. Other similarities can measure the structural differences of inputs, which can be arbitrary sets or strings of varying length or even numeric vectors. In this module, we focus on vector similarity and learn to measure the degree of semantic proximity among words and documents. In its simplest form, documents with higher match count of important words are closer to each other in vector space built from a document-term matrix we've seen earlier. A more precise and robust measure can be drawn from word and document vectors, which are built to capture semantic and syntactic representation.
Part 3: Examples of Similarity Metrics
Let's review a few different ways we can compute vectors and distance similarities in Python. So here we have two vectors, A and B. "Dot product" between them would indicate some sort of similarity in some sense, and that's just some of the element-wise multiplications. Because these are packed as a list, we cannot apply element-wise functions, such as those in NumPy library, but we can still use some operations in Python to do the element-wise products. For that, we would wrap these two lists into a single list of tuples, where the tuples represent the pairs in the corresponding positions of the original list, A and B. After that, we would iterate through each element of the list so each tuple is retrieved and the elements of the tuple are multiplied. Finally, all these different products — 1 multiplied by 0, 1 multiplied by 1, 1 multiplied by 2 — are added up together into a number 3; that's dot product. There are several different ways to compute that directly with the NumPy array; if you convert A and B to NumPy vectors or representations, you can then multiply A by B, and that will do the element-wise multiplication for you automatically. That could be summed up at the end. There's also this very nice notation with the @ symbol, which will do the dot product for the vectors and matrices. Then there is np.dot and np.inner, which is little bit more general function but still does the same multiplication and summation at the end. So all this produces the same results: 3, 3, 3, 3. The cosine similarity is basically a dot product for scaled vectors. First, we're computing the length or magnitude of vector A and vector B, then we're rescaling vector A and B — and we're not changing direction — we're just rescaling them to have unit length — and doing dot product at the end, which gives us 0.77 as the cosine similarity. Remember, this is the cosine of the angle between the two vectors A and B. It's close to 1, so we have an idea that these two vectors are fairly similar with respect to this metric. We can compute the radians and even degrees between the two vectors by first applying the arc cosine function, which is the inverse cosine, to the cosine similarity. This will cancel out the cosine effect, and we'll just return the angle in radians, which is 0.68. Then we can apply the math degrees, which will give us 39 degrees for the 0.68 radians. Alternatively, we can divide the radians by pi, multiply by 180, and get the same representation in degrees. There's a function in scipy.spatial.distance called cosine. This is a distance function, not the similarity. If we subtract it from 1, we will get the similarity of 0.77 just like before. There is a very nice function that will compute pairwise cosine similarities for a set of vectors. It will give us a matrix where each row represents one vector, XYZ, and the column represents XYZ as well, so the number in this matrix is just the cosine similarity for the row and column representation. On diagonal, we have all 1s always, and that's because the vector cosine similar with itself has 0 degrees — 0 in radians or 0 in degrees angle, and cosine of 0 is always 1. So that's why we have 1s. The 0s imply orthogonality or orthogonal vectors, and they are orthogonal by design, so we picked vectors that reside on the axis; [1, 0, 0] and [0, 1, 0], and [0, 0, 1], they all lie on the axis. And they can be of any length, and that would still produce a 90-degree angle between them. We can see that those angles are 90 degrees if we convert this matrix of cosine angles to just angles. Now, we can move on to the distance function, and having two vectors, A and B — [0, 1] and [0, 2] — We can compute the distance between them. Again, they lie or reside on the y-axis, and the distance between is just 1, which is exactly what we can compute with four different ways, using norm function or Euclidean function, or just computing the Euclidean distance ourselves by subtracting the vectors, multiplying their elements or squaring their elements, adding them up, and square-rooting at the end. Here's a nice visual representation of several different vectors that we pack into a dictionary called vecs. Vectors b, c, d, e, and f are all spread out. We are looking for the vector that is closest to a. And "closest" could have multiple meanings; it could be either distance close or cosine similarity close. From the cosine similarity perspective, we can order them by cosine similarity. So by iterating through this dictionary, pulling every vector representation and its name, we can find the cosine similarity between that vector's coordinates and the coordinates of the vector a of interest to us, and then taking the results that are rounded and sorting them by a decrease in cosine similarity. We'd have vector b as the closest to a, and this makes sense. If we look at the picture, the angle between b and a is the smallest. Doing the same thing, same type of shuffling, and applying a different function, Euclidean distance would get us the closest vector f in this two-dimensional space. That also makes sense because the distance between a and f is the closest among all the other distances between a and any other vector.
Part 4: 
Part 5: Generate Similarity Metrics
how to implement three key metrics — dot product, cosine similarity, and Euclidean (or L2) distance — with Python.
Finding Similar Documents With TF-IDF
a document-term matrix or TF-IDF matrix is a set of sparse row vectors representing documents. For any pair of such document vectors, we can compute their cosine similarity. Thus, a given document vector can give us the similarity to another document in the set of documents in that matrix. 
We can also find a pair of two most similar documents in our set. A major drawback in this process is that we must recompute TF-IDF matrix every time we need a vector or some sort of analysis for a new document; we'd have to add that document into the corpus and recompute TF-IDF.
 We can create TF-IDF matrix, which is a sparse matrix, and then represent it as a dense matrix by wrapping it into a DataFrame. We have all these different sentences and we have all these different words that are cleaned up a little bit lowercase then with stopwords removed. Some of these have similar roots or lemmas, and our goal here is to find sentences that are most similar to the first sentence, "A different language is a different vision of life." If we look at the coefficients, some of the coefficients do not have any correlation with any other value for a word like "different," but some other ones' language are probably similar — make this first sentence similar to many other sentences. Yet there is one word, "life," that makes it similar to another document containing the word "life" as well. So "One language sets you in a corridor for life. Two languages open every door along the way." These two must be similar. How can we compute this similarity automatically? How can we find the second sentence automatically? We can compute the cosine similarities using the cosine similarity function in Scikit-Learn, and this will pairwise create similarities for every vector in our original DTM or TF-IDF matrix. This particular number here, 0.033, is computed by taking the first row vector and the second row vector and computing the cosine similarity between them. 0.15 is done the same way between the first sentence and the third sentence that is represented by a column. This is how the matrix is filled up and it is symmetric. So 0.39 here makes two sentences very similar to each other. These are these longer sentences. And the one on the diagonal means that the sentence is perfectly similar to itself. If we look at the row of this matrix — which is a square matrix, of course, 15 by 15; that's how many sentences or quotes we have here — the biggest value of this matrix in the first row is between the first sentence and the third one, and that's the one that we found earlier to be most similar to each other. All the other ones have smaller values. We can also identify the clusters of documents that are most similar like this vector, the similarity of the vectors representing these two sentences. Notice that if we have a query document or a phrase which was never in the original set before, then we'll have problems because we don't have a vector for it. In order for us to have a TF-IDF vector for this phrase "language, nature and birds," we would have to edit to the set, recompute TF-IDF, and then find the similarity between this sentence or a document and every other document in the corpus.






FastText Model Formats and Transfer Learning
The introduction of FastText by Tomas Mikolov and Facebook in 2016 was a major breakthrough in use of NLP because it maps virtually any word to some vector, as long as at least one subword was present in the training vocabulary. 
The training model is distributed in over 150 languages and in two file formats. The smaller .vec text file is about 300 to 600 megabytes trained model. It offers pretty much the same functionality, albeit limited, as Word2Vec does. A larger .bin file is roughly three to eight gigabytes and contains a binary representation of the full pre-trained model. The file size depends greatly on the training vocabulary size; vector size, which can typically be between 100 and 1,000 scalar coefficients or model parameters; and numeric precision of these parameters. 
We always prefer the larger model if production computers have sufficient memory to store and process it. This binary model handles out-of-vocabulary words, which is highly desirable. 
These files are trained on different languages and different corpora, so you'll need to pick the model that is most relevant to your domain. 
It is also important to note that you can post-train binary models on your highly specific corpus, and Gensim package makes it easy. This is called transfer learning, when we fine-tune the vectors and expand the model trained on generic corpus to our specific domain. For example, we can take a model pre-trained on Wikipedia corpus and Google News corpus, and post-train it on medical records for breast cancer patients where highly specialized terminology and abbreviations are used.
Building Sentence Vectors With FastText
There are many ways to convert a full document to a FastText vector, not just a single-word vector. A simple technique is to feed the whole string to FastText's library, which will parse it into subwords, map them to vectors, and return an equally weighted average vector representing a full sentence as a bag of words. This may generate too many irrelevant subwords continuing punctuation and spaces and weaken the signal in the vector. Instead, we can speed it up by first tokenizing the document into words and then retrieving FastText vectors for each word individually. We would still need to compute the average vector, but we can compute and assign weights ourselves in any desired way. For example, weights could be provided by TF-IDF with rare-word vectors weighted more heavily. Let's go to the code. We are loading a pre-trained library from the internet public domain; wiki.simple is probably the simplest FastText pre-trained library you can find, and the zip file contains both the .vec and .bin, and the second one or the later is the one we'll use. "Wiki" means that it was trained on Wikipedia. This is about 2.5-gigabyte file, and it takes about 1.5 minutes to download it from the internet. It takes next 25 seconds to load it into Gensim's FastText object. We are loading wiki.simple.bin, which is already the expanded file, and specifying the Unicode encoding UTF-8, because some of the words in the model are unicoded. The vector size can be viewed with the function and it's 300 — just verifying — and this tells us about the complexity of the words and how much resources this will utilize. Next, we're loading some libraries that we'll be using, one of which is a function cosine distance. To turn it into cosine similarity, we're subtracting it from 1 and wrapping it up into a function or lambda function. The next cell takes a query, "Learn a new language and get a new soul." This is the query we'll be working with and comparing to itself, embedded in two different formats. One format is to parse it into words, as we're doing in this function GetSentVec. We are taking each word, passing it to FastText library, which returns a vector, except for the words that are in the stopword list if that's provided. Then we're finding the mean of all these different vectors. We might as well just find a sum because cosine similarity will standardize all vectors to unit length anyway. The second approach is to use FastText's library directly to pass the full sentence to that library and get a vector. We will normalize both of these vectors so that visually we can compare the coefficients and their magnitudes and directions. Let's run the file again. Here are the two vectors side by side, and we can see the coefficients, for the most part, have the same magnitude and the same direction. But the cosine similarities are rather low; 0.66, we'd expect much higher, especially because this is the same sentence, just embedded slightly with different approaches. One uses all subwords for the full string and one uses words then their subword presentations, and all that is aggregated back to the sentence-level vector. This is a graphical representation with scalars. What we're looking for in this 300-dimensional representation of these two sentences is that, again, the magnitude and the direction represented by strength of the color and the color itself is similar; observe the blues tend to match up and the reds or orange colors tend to match up as well in the same positions for both of these rows. Then we're taking all these different quotes that are now packaged as a dictionary, where the key is the name of the person and the quote is the value. We're taking all the different values — here they are — and embedding each one of those values with a sentence vector. We're using the GetSendVec function we wrote earlier to embed each one and compute the cosine similarity between the sentence vector of the quote to the sentence vector of the query that we had above. All that is packaged as a list of tuples and wrapped into a DataFrame to present nicely. We're also sorting the values or the rows by the cosine similarity. So if the closest quote we can find is the original quote itself, naturally it has cosine similarity 1; "Learn a new language and get a new soul" is the precise quote we had. Then the second one has 0.84 probably because it has the word "soul" and the word "language" in it and so on. Now we can see which ones are more alike or not. Keep in mind that this treats every sentence as a bag of words regardless of the position in the sentence and relevancy to other words.
Finding Similar Documents With FastText
We might actually be ready for real-world application of NLP in document search.
convert each document to a 300-dimensional vector
compute and evaluate a cosine similarity and matrix among all pairs of documents to identify most-related documents. 
Finally, convert a query string and look for a document closest to it. 
So let's look at the code. 
We are bringing a zip file with a pre-trained FastText model from the public domain and that's about 2.6-gigabyte file. 
then we're loading it into a Gensim FastText object. 
We'll specifically load in the binary model because that will handle out-of-vocabulary words, indicating the encoding to be UTF-8 — the Unicode encoding because some of the words there are Unicode, using Unicode characters. 
Then we use three different versions of the word "language" and the word "life"; We want to see which ones are similar and how similar they are. We generate a vector for each one of these words. 
Notice that each one brings a vector back regardless of capitalization, regardless of the plural or different form or morphology of these words. The first three words relating to "language" appear to be similar; they have similar magnitude and direction of the coefficients. If we're looking at the zero dimension, the magnitude is very large positive for the first three and "life" has close to zero. Then the next dimension has very large negative for the first three words and close to zero for the word "life." This can also be observed in color with this matrix. The colors will be similar for the first three and different for the last one. Will not always be the case, but for the most part we'll see three blues and a red or three reds and a blue or a light red. Now, we can load this dictionary of keys as the individual names and the values as the quotes that they produce — or generate — or ideate. We're interested in observing the same coefficients, except we don't want to look at these coefficients individually; let's look at them in aggregated way. This is 15 rows for 15 different quotes — we don't see all the quotes here — and 300 dimensions for the dimensions of the FastText vectors. We will see that some of the dimensions are similar in color, and that's because they all deal with language, so wherever you see that, that's probably the aspect of language that is being captured. Some of them are different and some of these words would probably result in greater similarity to some other ones; not words but sentences. This matrix is dimensions 15 by 300 as we expected. Now let's generate a pairwise cosine similarity using cosine similarity from SciKit-Learn. We're wrapping the cosine similarity into a DataFrame, giving it index names and column names as the quotes that we use. It's a little bit easier to identify which numbers are generated — or which sentences generate which numbers. The way you read this symmetric matrix is it's full of cosine similarities of the intersections of two words — or two sentences; one is in a row and one is in a column. This 0.73 is the cosine similarity between "Language is to the mind more than light is to the eye" in this sentence here, which is difficult to read because it's vertical. But we have 1s on the diagonal, and that's because the sentences are exactly similar to themselves so the cosine similarity is 1. Then we have blues where the cosine similarity is low. So "A mistake is to commit a misunderstanding" is a sentence that appears to be very low correlated to all the other sentences, except maybe this sentence here. And there is this 0.91 of two sentences that are strongly similar to each other for some reason. We're looking at semantic representations of the words. Notice that we're not taking the order of the words into account; there are other models that we'll look at in the course that will do that. Finally, we want to do a search. This search will allow us to use any type of phrase and return ordered sentences by their cosine similarity. So the top match will be at the top and the lowest match will be at the bottom. So "gardening ideas"; we know we didn't have the word "gardening" before, but we have "gardening ideas" and it brings up something about growth. We can change this to misspelled version of this — let's say "gardening ideass" — and run it, and we still have something about "grow" at the top. We can change this to "anotherSoul"; "anotherSoul" may be misspelled, maybe "Soul" was a capital word, and we are still getting a relevant phrase at the top, something unrelated to the second "soul." This is how robust this model is; it will generate a vector — a reasonably meaningful vector from anything you give it.


+++++++++++++


Module Introduction: Measure Similarity Between Document Vectors
Similarity
Similarity and Distance Metrics
Examples of Similarity Metrics
Similarity Metrics
Generate Similarity MetricsFinding Similar Documents With TF-IDF
Find Similar Documents With TF-IDF
FastText Model Formats and Transfer Learning
Building Sentence Vectors With FastText
Build Sentence Vectors With FastText
Finding Similar Documents With FastText
Find Similar Documents With FastText
Course Project, Part Three — Measuring Similarity Between Vectors
Module Wrap-up: Measure Similarity Between Vectors




+++++++++++++=
Project 3: Measuring Similarity Between Vectors
Understanding the limitations of word vectors will help you build realistic and practical real-world models. In this project, you'll explore some functionality — and limitations — of word vectors via the Gensim library in the context of speeches given by various Presidents of the United States.
You'll define functions that will help you use NLP to build and examine document vectors. 
build a document vector for each speech (i.e., document) that will be the average of all word vectors you identify within the document. 
identify words which are semantically close to the document vector and measure similarity among documents based on their vector representations. 
deduce that document vectors built from an averaged bag-of-word vectors lose their semantic representation as the size of the document increases.
Document Vector
In this task, you will write a function to calculate the centroid of the vector embeddings for lowercased vectorizable (LCV) words in the given document.
This is an important task because, while a single word vector is useful, embedding the meaning of the whole text in a single numerical vector can help you compare document similarity. 
One simple way to do this is via arithmetic (element-wise) averaging (i.e., mean) of all LCV words from a document. This document vector would have exactly the same dimension as the dimensions of the vectors used to produce it. Here is a quick example with made-up 3-dimensional vectors.
Take the phrase "I like NLP_." Suppose after tokenization and conversion to lowercase, "nlp" is not in a Word2Vec model, but "i" and "like" have vector representations 

Note: You will use all words in a document as a bag of words, without regard to their positioning, order, or distances to their neighboring words. This dramatically reduces the quality of the resulting document vector. However, the repetitive words can strongly influence the position of the document vector in the 50-dimensional vector space, so if the document discusses mostly cats and the other document discusses mostly dogs, then their vectors will represent cats and dogs, respectively.


















++++++++++++++++++++++++++++++
General Python notes
lamba is a nameless function 
tuple: immutable ordered list of objects
list a_list = [1, 2, 3, 4, 5]
dictionary 
a_dictionary = {}      another_dictionary = dict()
keys() : The keys() method on a dictionary will return a list of the keys.
values() : The values() method on a dictionary will return a list of the values.
items() : The items() method on a dictionary returns a list of tuples, each of which holds a respective key and value.(!!)




f string with {}     print(f'Item class: {type(item)}\tValue: {item}')
for loops 
try except 
List Comprehensions 
numbers = [1, 2, 3, 4, 5]
even_squares = [x * x for x in numbers if x % 2 == 0]
print(even_squares)  # Output: [4, 16]




Libraries 
NumPy
my_array = np.array([1, 2, 3, 5, 7, 11, 13, 17])
numpy_array.shape
numpy_array.reshape
numpy_array.ndim
genfromtxt()
re
search()
sub()
split()
findall()
IGNORECASE()
methods 
print()
help()
str()	converts any object to a string 
list()
type()
enumerate()
s.join()
s.lower()
len(),
s.find()
s.startswith(), s.endswith(),
s.replace(), s.strip(), s.split()
count() method
Counter() object




