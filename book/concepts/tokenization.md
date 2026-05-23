# Tokenization

## What it is
Tokenization is the process of breaking raw text into smaller units such as words, subwords, or characters.

## The key intuition
Text becomes model input only after it has been turned into discrete tokens. The tokenization step is where words and phrases become the atomic units your model can reason over.

## Why it matters
If tokenization is poor, the model gets a bad representation of the text. Good tokenization preserves meaning while keeping the vocabulary size manageable.

## When to use it
Use tokenization whenever you convert raw text into features or feed text into a model.

## Example notebooks in this repo
- `Identifying Keywords and Keyphrases From Text.ipynb`
- `Practice_Identifying_Collocations.ipynb`
- `class_6/Lexical_Relationships.ipynb`

If you want, add a short notebook named `tokenization.ipynb` here later for the exact code patterns.
