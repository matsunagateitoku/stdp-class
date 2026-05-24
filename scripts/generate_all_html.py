#!/usr/bin/env python3
"""
Convert all class notebooks to HTML, create Colab wrapper pages,
and update class_1–6 HTML files so every notebook link opens the
website version and has a Colab badge.

Run from /workspaces/stdp-class:
    python3 scripts/generate_all_html.py
"""

import re, subprocess, urllib.parse
from pathlib import Path

ROOT   = Path("/workspaces/stdp-class")
OWNER  = "matsunagateitoku"
REPO   = "stdp-class"
BRANCH = "main"

BADGE = (
    ' <a href="{url}" target="_blank" rel="noopener">'
    '<img src="https://colab.research.google.com/assets/colab-badge.svg"'
    ' alt="Open in Colab" style="vertical-align:middle;margin-left:.3em"></a>'
)


def safe_stem(nb_filename):
    s = Path(nb_filename).stem
    s = re.sub(r'[ ()\[\]]', '_', s)
    s = re.sub(r'[^A-Za-z0-9._-]', '_', s)
    s = re.sub(r'_+', '_', s)
    return s.strip('_')


def colab_url(rel_path):
    return (
        f"https://colab.research.google.com/github/{OWNER}/{REPO}"
        f"/blob/{BRANCH}/{urllib.parse.quote(rel_path)}"
    )


def convert_nb(nb_abs, out_dir, stem):
    out_html = out_dir / f"{stem}.html"
    if out_html.exists():
        return True
    r = subprocess.run(
        ["jupyter", "nbconvert", "--to", "html", str(nb_abs),
         "--output", stem, "--output-dir", str(out_dir)],
        capture_output=True, text=True)
    if r.returncode == 0:
        print(f"  Converted: {nb_abs.name}")
        return True
    print(f"  FAILED:    {nb_abs.name}: {r.stderr[:120]}")
    return False


def make_wrapper(out_dir, stem, title, curl):
    wp = out_dir / f"{stem}_index.html"
    if wp.exists():
        return
    wp.write_text(
        f'<!doctype html>\n<html>\n<head>\n'
        f'  <meta charset="utf-8">\n'
        f'  <meta name="viewport" content="width=device-width,initial-scale=1">\n'
        f'  <title>{title}</title>\n'
        f'  <style>body{{font-family:system-ui,sans-serif;margin:0;padding:1rem}}</style>\n'
        f'</head>\n<body>\n'
        f'  <p>\n'
        f'    <a href="{curl}" target="_blank" rel="noopener">\n'
        f'      <img src="https://colab.research.google.com/assets/colab-badge.svg"'
        f' alt="Open in Colab">\n'
        f'    </a>\n'
        f'  </p>\n'
        f'  <iframe src="{stem}.html"'
        f' style="width:100%;height:78vh;border:0"></iframe>\n'
        f'</body>\n</html>')
    print(f"  Wrapper:   {wp.name}")


# ── Registry ──────────────────────────────────────────────────────────────────
# (href_in_html, rel_path_from_root, out_class)
REGISTRY = [
    # ── Class 1 ───────────────────────────────────────────────────────────────
    ("Text_preprocessing_pipeline.ipynb",
     "Text_preprocessing_pipeline.ipynb", "class_1"),

    # ── Class 2 ───────────────────────────────────────────────────────────────
    ("Practice With Basic Matrix Operations.ipynb",
     "Practice With Basic Matrix Operations.ipynb", "class_2"),
    ("Practice Matrix Decomposition.ipynb",
     "Practice Matrix Decomposition.ipynb", "class_2"),
    ("Means.ipynb", "Means.ipynb", "class_2"),
    ("Practice Splitting a Simulated Data Set Into Training and Validation Sets.ipynb",
     "Practice Splitting a Simulated Data Set Into Training and Validation Sets.ipynb", "class_2"),
    ("Practice Splitting Newsgroup Data Into Training and Validation Sets.ipynb",
     "Practice Splitting Newsgroup Data Into Training and Validation Sets.ipynb", "class_2"),

    # ── Class 3 root-level ────────────────────────────────────────────────────
    ("Train a Classification Model To Categorize Text.ipynb",
     "Train a Classification Model To Categorize Text.ipynb", "class_3"),
    ("Train a Classification Model To Categorize Text (1).ipynb",
     "Train a Classification Model To Categorize Text (1).ipynb", "class_3"),
    ("Practice Training and Validating a Baseline Classification Model.ipynb",
     "Practice Training and Validating a Baseline Classification Model.ipynb", "class_3"),
    ("Practice Creating and Analyzing a Confusion Matrix.ipynb",
     "Practice Creating and Analyzing a Confusion Matrix.ipynb", "class_3"),
    ("Practice Improving a Model Using Hyperparameter Tuning.ipynb",
     "Practice Improving a Model Using Hyperparameter Tuning.ipynb", "class_3"),
    ("Practice Improving a Model Using Hyperparameter Tuning (1).ipynb",
     "Practice Improving a Model Using Hyperparameter Tuning (1).ipynb", "class_3"),
    ("Evaluating the Performance of a Classification Model.ipynb",
     "Evaluating the Performance of a Classification Model.ipynb", "class_3"),
    ("Practice Calculating Metrics To Evaluate Classification Model Performance .ipynb",
     "Practice Calculating Metrics To Evaluate Classification Model Performance .ipynb", "class_3"),
    ("Practice Calculating Metrics To Evaluate Classification Model Performance  (1).ipynb",
     "Practice Calculating Metrics To Evaluate Classification Model Performance  (1).ipynb", "class_3"),
    ("Practice Applying Sigmoid and Logit Functions for Logistic Regression.ipynb",
     "Practice Applying Sigmoid and Logit Functions for Logistic Regression.ipynb", "class_3"),
    ("Practice Calculating the Accuracy for a New Logistic Regression Model.ipynb",
     "Practice Calculating the Accuracy for a New Logistic Regression Model.ipynb", "class_3"),
    ("Practice Selecting a Model To Outperform the Baseline.ipynb",
     "Practice Selecting a Model To Outperform the Baseline.ipynb", "class_3"),
    ("Processing a Data Set for Supervised Machine Learning Classification Model.ipynb",
     "Processing a Data Set for Supervised Machine Learning Classification Model.ipynb", "class_3"),

    # ── Class 4 root-level ────────────────────────────────────────────────────
    ("4.1.1.Practice_Comparing_Texts.ipynb",
     "4.1.1.Practice_Comparing_Texts.ipynb", "class_4"),
    ("Practice_Identifying_Collocations.ipynb",
     "Practice_Identifying_Collocations.ipynb", "class_4"),
    ("Practice Measuring Keyword Quality.ipynb",
     "Practice Measuring Keyword Quality.ipynb", "class_4"),
    ("Identifying Keywords and Keyphrases From Text.ipynb",
     "Identifying Keywords and Keyphrases From Text.ipynb", "class_4"),
    ("Identifying Keywords and Keyphrases From Text (1).ipynb",
     "Identifying Keywords and Keyphrases From Text (1).ipynb", "class_4"),
    ("Identifying Topics From Documents.ipynb",
     "Identifying Topics From Documents.ipynb", "class_4"),
    ("Identifying Topics From Documents (1).ipynb",
     "Identifying Topics From Documents (1).ipynb", "class_4"),
    ("Identifying Topics From Documents (2).ipynb",
     "Identifying Topics From Documents (2).ipynb", "class_4"),
    ("Extracting Summary Sentences From Documents.ipynb",
     "Extracting Summary Sentences From Documents.ipynb", "class_4"),
    ("Extracting Summary Sentences From Documents (1).ipynb",
     "Extracting Summary Sentences From Documents (1).ipynb", "class_4"),
    # Class 4 subdirectory
    ("class_4/Practice_Extracting_a_Summary.ipynb",
     "class_4/Practice_Extracting_a_Summary.ipynb", "class_4"),

    # ── Class 5 root-level ────────────────────────────────────────────────────
    ("Practice_Finding_Mismatches.ipynb",
     "Practice_Finding_Mismatches.ipynb", "class_5"),
    ("Using Metrics To Determine Text Similarity.ipynb",
     "Using Metrics To Determine Text Similarity.ipynb", "class_5"),
    # Class 5 subdirectory (HTML already generated; ensure wrappers exist and links updated)
    ("class_5/Calculating-Centroids.ipynb",
     "class_5/Calculating-Centroids.ipynb", "class_5"),
    ("class_5/Dendrograms.ipynb",
     "class_5/Dendrograms.ipynb", "class_5"),
    ("class_5/Find_Movies.ipynb",
     "class_5/Find_Movies.ipynb", "class_5"),
    ("class_5/Hierarchical_Clustering.ipynb",
     "class_5/Hierarchical_Clustering.ipynb", "class_5"),
    ("class_5/Measuring_Clustering.ipynb",
     "class_5/Measuring_Clustering.ipynb", "class_5"),
    ("class_5/Medoid_Movie.ipynb",
     "class_5/Medoid_Movie.ipynb", "class_5"),
    ("class_5/Movies_K.ipynb",
     "class_5/Movies_K.ipynb", "class_5"),
    ("class_5/Perform K-Means Clustering on Sentence Embeddings To Group Similar Texts (1).ipynb",
     "class_5/Perform K-Means Clustering on Sentence Embeddings To Group Similar Texts (1).ipynb",
     "class_5"),
    ("class_5/Perform_K.ipynb",
     "class_5/Perform_K.ipynb", "class_5"),
    ("class_5/Performing Hierarchical Clustering on Sentence Embeddings To Group Similar Texts (1).ipynb",
     "class_5/Performing Hierarchical Clustering on Sentence Embeddings To Group Similar Texts (1).ipynb",
     "class_5"),
    ("class_5/Performing_Hierarchical.ipynb",
     "class_5/Performing_Hierarchical.ipynb", "class_5"),
    ("class_5/Practice_Comparing_Sequences.ipynb",
     "class_5/Practice_Comparing_Sequences.ipynb", "class_5"),
    ("class_5/Practice_Embedding.ipynb",
     "class_5/Practice_Embedding.ipynb", "class_5"),
    ("class_5/Practice_Using_Levenshtein.ipynb",
     "class_5/Practice_Using_Levenshtein.ipynb", "class_5"),
    ("class_5/UsingMetrics.ipynb",
     "class_5/UsingMetrics.ipynb", "class_5"),
    ("class_5/Using_Metrics.ipynb",
     "class_5/Using_Metrics.ipynb", "class_5"),

    # ── Class 6 subdirectory ──────────────────────────────────────────────────
    ("class_6/Computing WordNet.ipynb",
     "class_6/Computing WordNet.ipynb", "class_6"),
    ("class_6/Lexical_Relationships.ipynb",
     "class_6/Lexical_Relationships.ipynb", "class_6"),
    ("class_6/Synsets_Lemmas.ipynb",
     "class_6/Synsets_Lemmas.ipynb", "class_6"),
    ("class_6/relationships_WordNet.ipynb",
     "class_6/relationships_WordNet.ipynb", "class_6"),
    ("class_6/NER.ipynb",
     "class_6/NER.ipynb", "class_6"),
    ("class_6/Conduct.ipynb",
     "class_6/Conduct.ipynb", "class_6"),
    ("class_6/Conduct2.ipynb",
     "class_6/Conduct2.ipynb", "class_6"),
    ("class_6/Conduct_Sentiment.ipynb",
     "class_6/Conduct_Sentiment.ipynb", "class_6"),
    ("class_6/TextBlob.ipynb",
     "class_6/TextBlob.ipynb", "class_6"),
    ("class_6/Practice Conducting Sentiment Analysis With the TextBlob Model.ipynb",
     "class_6/Practice Conducting Sentiment Analysis With the TextBlob Model.ipynb", "class_6"),
    ("class_6/VADER.ipynb",
     "class_6/VADER.ipynb", "class_6"),
    ("class_6/Train.ipynb",
     "class_6/Train.ipynb", "class_6"),
    ("class_6/Train1.ipynb",
     "class_6/Train1.ipynb", "class_6"),
    ("class_6/P.ipynb",
     "class_6/P.ipynb", "class_6"),
    ("class_6/final.ipynb",
     "class_6/final.ipynb", "class_6"),
]

# class_3.html already links to notebooks/class_3/..._index.html for some
# notebooks — keep those hrefs but still add the Colab badge.
CLASS3_EXISTING_WRAPPERS = {
    "notebooks/class_3/Train_a_Classification_Model_To_Categorize_Text_index.html":
        "Train a Classification Model To Categorize Text.ipynb",
    "notebooks/class_3/Train_a_Classification_Model_To_Categorize_Text__1__index.html":
        "Train a Classification Model To Categorize Text (1).ipynb",
    "notebooks/class_3/Practice_Training_and_Validating_a_Baseline_Classification_Model_index.html":
        "Practice Training and Validating a Baseline Classification Model.ipynb",
    "notebooks/class_3/Practice_Creating_and_Analyzing_a_Confusion_Matrix_index.html":
        "Practice Creating and Analyzing a Confusion Matrix.ipynb",
    "notebooks/class_3/Practice_Improving_a_Model_Using_Hyperparameter_Tuning_index.html":
        "Practice Improving a Model Using Hyperparameter Tuning.ipynb",
    "notebooks/class_3/Practice_Improving_a_Model_Using_Hyperparameter_Tuning__1__index.html":
        "Practice Improving a Model Using Hyperparameter Tuning (1).ipynb",
}

# ── Build lookup: href_in_html → (web_href, colab_url) ───────────────────────
lookup = {}
for href_in_html, rel_path, out_class in REGISTRY:
    stem    = safe_stem(Path(rel_path).name)
    web     = f"notebooks/{out_class}/{stem}_index.html"
    curl    = colab_url(rel_path)
    lookup[href_in_html] = (web, curl)

# Add class_3 already-converted wrappers (keep href, just badge)
for web_href, nb_name in CLASS3_EXISTING_WRAPPERS.items():
    lookup[web_href] = (web_href, colab_url(nb_name))


# ── Step 1: Convert notebooks ─────────────────────────────────────────────────
print("=== Step 1: Convert notebooks to HTML ===")
for href_in_html, rel_path, out_class in REGISTRY:
    nb_abs = ROOT / rel_path
    if not nb_abs.exists():
        print(f"  MISSING: {rel_path}")
        continue
    out_dir = ROOT / "notebooks" / out_class
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_stem(nb_abs.name)
    convert_nb(nb_abs, out_dir, stem)
    make_wrapper(out_dir, stem, nb_abs.name, colab_url(rel_path))


# ── Step 2: Update class HTML files ───────────────────────────────────────────
print("\n=== Step 2: Update class HTML files ===")

def update_html_file(fname):
    path = ROOT / fname
    original = path.read_text()
    content  = original

    def replacer(m):
        href       = m.group(1)
        extra_attr = m.group(2)   # e.g. ' target="_blank"'
        link_text  = m.group(3)
        if href not in lookup:
            return m.group(0)
        web_href, curl = lookup[href]
        badge = BADGE.format(url=curl)
        return f'<a href="{web_href}"{extra_attr}>{link_text}</a>{badge}'

    content = re.sub(
        r'<a href="([^"]+)"([^>]*)>(.*?)</a>',
        replacer,
        content,
        flags=re.DOTALL,
    )

    if content != original:
        path.write_text(content)
        print(f"  Updated: {fname}")
    else:
        print(f"  No change: {fname}")

for cls_html in ["class_1.html", "class_2.html", "class_3.html",
                 "class_4.html", "class_5.html", "class_6.html"]:
    update_html_file(cls_html)

print("\nDone.")
