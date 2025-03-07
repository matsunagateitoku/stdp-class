# Conducting Semantic and Sentiment Analysis
Course Description
We have all been misunderstood when sending a text message or email, as tone often does not translate well in written communication. Similarly, computers can have a hard time discerning the meaning of words if they are being used sarcastically, such as when we say “Great weather” when it’s raining. If you are automatically processing reviews of your product, a negative review will have many of the same key words as a positive one, so you will need to be able to train a model to distinguish between a good review and a bad review. This is where semantic and sentiment analysis come in.
In this course, you will examine many kinds of semantic relationships that words can have (such as hypernyms, hyponyms, or meronyms), which go a long way toward extracting the meaning of documents at scale. You will also implement named entity recognition to identify proper nouns within a document and use several techniques to determine the sentiment of text: Is the tone positive or negative? These invaluable skills can easily turn the tide in a difficult project for your team at work or on the path toward achieving your personal goals.


## Module 1: Conduct Semantic Analysis Using WordNet
 
### key concepts
- **wordnet**: WordNet is a large lexical database of the English language, created by linguist George A. Miller and his colleagues at Princeton University in the mid-1980s. The project aimed to reflect the way humans organize and understand words, grouping them into sets of synonyms called "synsets," each representing a distinct concept. These synsets are interlinked through various semantic relationships, such as hypernyms (generalizations), hyponyms (specific instances), meronyms (part-whole relationships), and antonyms (opposites). The first version of WordNet was released in 1990, and it has since become a key resource in natural language processing (NLP), machine learning, and AI, helping computers better understand and process human language.
- Homophones sound the same, but also have different senses.　Example: Buy/By/Bye, Eye/I
- Homographs are spelled identically, but have different senses.　Examples: Bat (animal)/Bat (used to play baseball), river bank/financial bank
- Homographs and homophones are homonyms, which is their superset.
- Homophones sound the same, but have different senses.
Homographs are written the same way, but have different senses.
A holonym is a whole consisting of parts (meronyms).

### questions 

| Lesson          |         Discription                                | Colab link    |
|-------------------|----------------------------------------------|------|
| **Module Intro**   | One way to compare the contents of many texts is by examining their vocabularies. However, even texts relating to the same topic may not include the exact same set of words — synonyms, part of speech variations, and the use of broad versus specific terms can make it challenging to compare textual vocabulary alone. The process of semantic analysis can help provide insight into the meaning, or sense, behind terms, as well as the relationship between different terms. A computer can then use these relationships to compare document content and categorize documents at scale. | In this module, you will be introduced to many types of lexical semantic relationships, which you will practice retrieving in Python using the lexical database WordNet. Then, you will use these relationships to compare various texts and evaluate their similarity.
| Introduction to Semantic Analysis **Video** | You can perform a semantic analysis using techniques to extract, represent, and store the meaning of texts. To allow for automated methods that give more discrete semantic classifications than simply extracting meaning from collocated words, experts created WordNet, which is a lexical database containing semantic relationships and meanings for words and phrases.  Here, Professor Melnikov introduces the WordNet database, describes its architecture, and discusses how it’s used in semantic analysis.| - gives background on wordnet
| Retrieve Synsets and Lemmas Using WordNet **Video**     |  Synsets, which are sets of synonymous lemmas*, are an important part of semantic analysis in WordNet. In this video, Professor Melnikov demonstrates how to access, interpret, and list attributes for synsets and lemmas in WordNet both in English and using the multilingual feature in Python.     | 1. load wordnet from NLTK 
| Practice Retrieving Synsets and Lemmas Using WordNet     | attributes and methods for synsets and lemmas in wordnet   |[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matsunagateitoku/stdp-class/blob/main/class_6/Synsets_Lemmas.ipynb)|
| Retrieve Lexical Semantic Relationships Using WordNet      | analyze semantic relationships in synsets. evaluate **entailments**, which are verbs causally evolved from some other verb.  **homographs**  many different meanings even though it's written the same. **hyponym** of another sense if the former is more specific than the latter, which is called a **hypernym**. vehicle is hypernym of a car which is a hyponym of a vehicle in this parent-child relationship.**paths** evaluate  words respect -whole relation. lettuce and tomato are **meronyms** of a sandwich, which is their **holonym**. 
| Practice Retrieving Lexical Semantic Relationships Using WordNet    |  Previously, Professor Melnikov demonstrated how you can retrieve and analyze different lexical semantic relationships in WordNet. In the "Review" section of this ungraded coding exercise, you will use these techniques as Professor Melnikov presented them in the video. In the "Optional Practice" section of this exercise, you will practice using WordNet to analyze entailment, antonyms, and hyponyms, and how to compute path similarity between a word and two homographs.    | [![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matsunagateitoku/stdp-class/blob/main/class_6/relationships_WordNet.ipynb)|
| Use Cases for Lexical Semantic Relationships     |      |  [text](./class_6/relationships.pdf)  
| Semantic Analysis Terminology     |      |
| Compute the Similarity of Lexical Semantic Relationships Using WordNet     |      |
| Practice Computing Similarity of Lexical Semantic Relationships Using WordNet     |  demonstrated how to compute the similarity score and find the closest hypernym between a pair of synsets. In the "Review" section of this ungraded coding exercise, you will use these techniques as Professor Melnikov presented them in the video. In the "Optional Practice" section of this exercise, you will practice evaluating the similarity between different senses of the same word.    |   [![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matsunagateitoku/stdp-class/blob/main/class_6/Lexical_Relationships.ipynb)|
| Compute Document Similarity Using WordNet     |      |
| Practice Computing Document Similarity Using WordNet     |  demonstrated how to compute the average similarity between two lists of synsets in order to find document similarity in WordNet. In the "Review" section of this ungraded coding exercise, you will use these techniques as Professor Melnikov presented them in the video. In the "Optional Practice" section of this exercise, you will practice computing the similarity score to find document similarity between multiple documents.   |  [![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matsunagateitoku/stdp-class/blob/main/class_6/P.ipynb)|  
| Course Project, Part One — Conducting Semantic Analysis Using WordNet   | In Part One of the course project, you will find related sysnets to a lemma and their hypernyms. Then you will compute the path similarity score for two lemmas. The tasks you will perform are based on videos and coding activities in the current module but may also rely on your preparation in Python and basic math. Use the tools from this module to help you complete this part of the course project. Additionally, you may consult your course facilitator and peers by posting in the project forum that is linked in the sidebar.
Task 1 Complete UDF Lemmas(), which takes sLemma lemma word, finds all related synsets SS and returns the set of lemma names from all synsets in SS. Task 2 Complete UDF Hypernyms(), which takes sLemma lemma word, finds all related synsets SS and returns the set of hypernym's lemma names from all synsets in SS. Task 3 Complete UDF Sim(), which takes two lemmas and for each combination of their synsets computes a path similarity score. If ReturnBest is selected, then only the topmost similarity pairs are returned (which can be more than one). Sim('teacher', 'tiger') return the following dataframe (ordered by columns ss1, ss2):
|[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matsunagateitoku/stdp-class/blob/main/class_6/Conduct.ipynb)| 
| Module Wrap-up: Conduct Semantic Analysis Using WordNet   |   |

## Module 2: Train a Model To Predict Named Entity Tags Within a Text
### key concepts:
### questions 
| Lesson          |         Discription                                | Colab link    |
|-------------------|----------------------------------------------|------|
|Intro   |  Another way to compare documents is by noting whether they reference the same person, location, or date. Named entity recognition (NER) is a technique for extracting and categorizing these key entities, which are often proper nouns. This module will introduce you to a variety of named entities before you practice identifying them within a document. After training a model to predict NER tags, you will then test its performance on an unfamiliar document.  |
|Introduction to Named Entity Recognition **video**   |    | A related task is named entity linking or disambiguation, NEL or NAD. relates to coreference resolution.   state-of-the-art systems can achieve 93% or higher F-measure performance on some standard annotated datasets compared to human performance of about 97%
|   |    |
|   |    |
| Course Project, Part Two — Training a Model to Predict Named Entity Tags Within a Text  | train a CRF model to predict NE tags to recognize movie related named entities and then measure the models performance. The tasks you will perform are based on videos and coding activities in the current module, but they may also rely on your preparation in Python and basic math.   | [![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matsunagateitoku/stdp-class/blob/main/class_6/Train.ipynb)|
|Wrap up   | Now you have evaluated the content of texts using both lexical semantic analysis and NER. In this module, you were first introduced to several named entity tags available through the SpaCy library. With the help of the GMB Corpus, you were able to practice implementing NER and IOB tagging before building your own NER tagger. Finally, you trained, tested, and applied your tagger to predict named entity tags in an unfamiliar corpus, then evaluated its performance. |　Train a Model To Predict Named Entity Tags Within a Text


## Module 3: Conduct Semantic Analysis Using WordNet

### key concepts
### questions 
| Lesson          |         Discription                                | Colab link    |
|-------------------|----------------------------------------------|------|


