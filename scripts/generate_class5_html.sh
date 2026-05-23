#!/usr/bin/env bash
set -euo pipefail

# Script: convert class_5 notebooks to HTML and create Colab wrapper pages
# Usage: run from repository root: ./scripts/generate_class5_html.sh

SRC_DIR="class_5"
OUT_DIR="notebooks/class_5"
REPO_OWNER="matsunagateitoku"
REPO_NAME="stdp-class"
BRANCH="main"

mkdir -p "$OUT_DIR"

shopt -s nullglob
for nb in "$SRC_DIR"/*.ipynb; do
  base=$(basename "$nb")
  # make a filesystem-safe name
  safe=$(echo "$base" | sed 's/ /_/g; s/[()]/_/g' | sed 's/[^A-Za-z0-9._-]/_/g')
  name="${safe%.*}"
  echo "Converting: $base -> $name.html"

  if command -v jupyter >/dev/null 2>&1; then
    jupyter nbconvert --to html "$nb" --output "$name" --output-dir "$OUT_DIR" || echo "nbconvert failed for $base"
  else
    echo "jupyter not found: skipping nbconvert for $base (will still create wrapper)"
  fi

  # URL-encode the notebook path for Colab
  relpath="$SRC_DIR/$base"
  encoded=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$relpath")
  colab_url="https://colab.research.google.com/github/$REPO_OWNER/$REPO_NAME/blob/$BRANCH/$encoded"

  wrapper="$OUT_DIR/${name}_index.html"
  cat > "$wrapper" <<HTML
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>$base</title>
  <style>body{font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;padding:1rem}</style>
</head>
<body>
  <p>
    <a href="$colab_url" target="_blank" rel="noopener">
      <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
    </a>
  </p>
  <iframe src="${name}.html" style="width:100%;height:78vh;border:0"></iframe>
</body>
</html>
HTML
done

# generate index for class_5
INDEX="$OUT_DIR/index.html"
cat > "$INDEX" <<HTML
<!doctype html>
<html>
<head><meta charset="utf-8"><title>class_5 notebooks</title></head>
<body><h1>class_5 notebooks</h1><ul>
HTML

for f in "$OUT_DIR"/*_index.html; do
  [ -f "$f" ] || continue
  fname=$(basename "$f")
  display=$(echo "$fname" | sed 's/_index.html$/.ipynb/')
  echo "  <li><a href=\"$fname\">$display</a></li>" >> "$INDEX"
done

cat >> "$INDEX" <<HTML
</ul></body></html>
HTML

echo "Done. Output in $OUT_DIR"
