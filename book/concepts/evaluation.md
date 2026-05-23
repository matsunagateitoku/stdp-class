# Evaluation

## What it is
Evaluation tells you whether your model is doing the right thing.

## The key intuition
The only way to know if your model works is to measure it against the right criteria.

## Why it matters
A model can look good in one metric and still fail in real use. Evaluation forces you to compare model predictions against actual outcomes.

## Metrics to remember
- Accuracy, precision, recall
- Confusion matrix
- Loss and validation performance
- Similarity metrics such as Jaccard, cosine, Levenshtein

## Example notebooks in this repo
- `Evaluating the Performance of a Classification Model.ipynb`
- `Practice Creating and Analyzing a Confusion Matrix.ipynb`
- `Practice Calculating Metrics To Evaluate Classification Model Performance .ipynb`
- `Practice Improving a Model Using Hyperparameter Tuning.ipynb`

## Remember this
Always link the metric back to the task. For classification, use accuracy/precision/recall. For text similarity, use a distance or similarity score.
