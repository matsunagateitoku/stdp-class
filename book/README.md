# NLP Notes Book

This folder contains a lightweight Jupyter Book scaffold for the NLP/classification material in this repo.

## What is here
- `_config.yml` — Jupyter Book configuration
- `_toc.yml` — navigation structure
- `intro.md` — overview and study guide
- `concepts/` — memory-focused concept pages

## Next steps
1. Install Jupyter Book:
   ```bash
   pip install jupyter-book
   ```
2. Add selected notebooks to this folder or update the links in the concept pages.
3. Build the site:
   ```bash
   cd book
   jupyter-book build .
   ```
4. Open `_build/html/index.html` in your browser.

## Why this structure helps
- The concept pages make the material digestible.
- Your notebooks stay as code references.
- You get a public-friendly site without forcing every notebook to be rewritten.
