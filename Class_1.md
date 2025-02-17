# Class 1: Natural Language Processing Fundamentals 

## Mod 1: Finding Simple Patterns in a Body of Text

| Lesson          |         Discription                                | Colab link    |
|-------------------|----------------------------------------------|------|
| **??**   | ??    |                                 
| **??**   | ??    |    




- Lots of basic string manipulation (lower, split, etc). Then using regex to find patterns in the text.
-    lower()
-    split()
- Finally, counting what is found (Counter, most_common)

most_common()
Counter()   a class from the Python collections module that is used to count the frequency of elements in an iterable (such as a list or a string). It provides a convenient way to keep track of counts of hashable objects.
regex re()    Parsing Strings with Regular Expressions
made a function that retrieves an alphabetically sorted list of unique tokens from string documents. 
function that returns a list of counts in decreasing-order, from the most frequent words to the least frequent words in a string document. Thei 




++++++++++++++++++++++++++++++++++++++++++++++++
Part Two — Preprocess Text to Reduce Vocabulary
This section is all about string parsing and document cleaning to preprocess. This is key to reduce the vocabulary
introduces the nltk library 
parse a document into token 
breaking a document into sentences and counting them
refining with regex to deal with newlines etc. 
downloading text
   _ = nltk.download(['brown'], quiet=True)
 Ss6 = {s.lower() for s in nltk.corpus.brown.words()}
turn the document into a string of tokens with stop word removed with each separated by a space 
The enumerate() function in Python is used to iterate over a sequence (such as a list, tuple, string, etc.) while keeping track of the index of each item. It returns an enumerate object which yields pairs of indexes and elements, where each pair is represented as a tuple.
checking the length of the sentence to make sure it seems correct 
working with characters
ASCII
unicode  -- UTF-8  u’string’
encode()
decode()
    def Norm(self) -> object:
        self.LsWords = [unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8') for word in self.LsWords]


dealing with contractions
simple version can be done via regex or a map
re library unContract_generic()
contractions library 
fix()
text correction
punkt
spelling correction
textblob library 
correct () and spellckeck() 
removing HTML tags  
stemming and lemmatization 
nltk stem.PorterStemmer()
nltk WordNetLemmatizer()
lemma = dictionary form of a word 
removing stopwords 


Pipe() class, which is a Python class for building custom text preprocessing pipelines.
call it like this:
pp = Pipe(LsBookWords, SsStopWords='nltk', SsLex='nltk').Low.Norm.Exp.Words.Stem.Stop.NoNum
The class methods containing preprocessing code are exposed as properties (with @property decorator). The properties can be called without parenthesis, which is convenient and visually attractive. Every preprocessing step logs the task name and some basic stats to the dictionary DStat, which is stored internally in the instantiated Pipe object. So, if needed, one can evaluate the compression of the original document's lexicon at each step of the pipeline.


In Python, the @property decorator is a built-in decorator that allows you to define a method as a property of a class. Properties enable you to define special behavior when accessing attributes of an object, such as performing calculations, validating inputs, or controlling access.


Task 1: Initialize attributes
Initialize the class attributes (self.LsWords, self.DStat, self.LsStep) in __init__().
Task 2: Format Output String
Complete the Out() method to format the output string.
Tasks 3 - 10: String Preprocessing Methods
Complete the Low(), NoNum(), Words(), Stop(), Norm(), Exp(), Stem(), Lem() properties.
assert isinstance(LsWords, list)




12 functions that make up a preprocessing pipeline that you can use to tag, lemmatize, and parse a document, as well as uncover its hierarchical dependencies. 
use each function individually to investigate a mix of large and small documents 
note: %time: This is a special IPython (Jupyter) magic command used to time the execution of a single statement. It measures and prints the elapsed wall-clock time taken to execute the statement.


sDoc is a string document with at least one sentence
Ex: a string with 2 sentences, 'I do. We go.'
LsSents is a list of string sentences
Ex: ['I do.', 'We go.']
LLsWords is a list of lists of words of sentences
Ex: [['I','do','.'],['We','go','.']]
LLTsWordPOST is a list of lists of tuples of word & Penn POS tag pairs
Ex: [[('I','PRP'),('do','VBP'),('.','.')],  [('We','PRP'),('go', 'VBP'),('.', '.')]]
Wordnet lemmatizer uses WordNet POS tags: 'a':adjective, 'n':noun (and is the default), 'r':adverb, 'v':verb
Ex: [[('I','n'),('do','v'),('.','n')],  [('We','n'),('go', 'v'),('.', 'n')]]
++++++++++++++++++++++++++++++++++++
Class 1, Task 3: Tagging and Parsing a Document 
The task is to tag and parse a document 
you need tags to lemmatize and it help to reduce the size of the vocab
mainly using nltk library 
stem.wordLemmatizer() definition
pos_tag() method
word.tokenize() method
Freq.Dist() method
Standard Library
Counter object
most.common() method
use nltk.sent_tokenize() to break text into sentences 
use nltk.word_tokenize() to break text into words
use nltk.pos_tag() to tag
covert Penn tags to Wordnet tags 
Note things get funky with this tags because nltk.pos_tag outputs Penn Treebank POS tag set and 
NLTK (Natural Language Toolkit) does not have a direct lemmatize function in its core library like nltk.lemmatize. so, lemmatization in NLTK is typically performed using the WordNetLemmatizer class from the nltk.stem module. The WordNetLemmatizer in NLTK accepts Part-of-Speech (POS) tags as inputs to specify the context in which a word should be lemmatized.
note the nltk does not use the UPenn tags, it uses wordnet_tag
use WordNetLemmatizer()  limmatize() to lemmtize words with tag
use nltk.RegexParser
use this to chuck text    
ChunkTree = nltk.RegexpParser(sGrammar).parse(LTsPOST)sGrammar="VP: {<V.*>+}")
ChunkTree = nltk.RegexpParser(sGrammar).parse(LTsPOST)
Chunk() to find chunk 
sGrammar = r'''NP1: {<DT>? <JJ>? <NN.*>+}'''
note: The collection Module in Python provides different types of containers. A Container is an object that is used to store different objects and provide a way to access the contained objects and iterate over them. Some of the built-in containers are Tuple, List, Dictionary, etc.
Punkt Sentence Tokenizer. This tokenizer divides a text into a list of sentences by using an unsupervised algorithm to build a model for abbreviation words, collocations, and words that start sentences. It must be trained on a large collection of plaintext in the target language before it can be used.
chunking (shallow parsing )= finding phrases. 
This can be done manually with regex on tags
better to use spacy
Tree graphs 
syntactic parse 
dependency parse 
constituency parse 


nltk.word_tokenize() 
 nltk.sent_tokenize()
nltk.pos_tag()
WordNetLemmatizer()
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
