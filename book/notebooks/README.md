# Book Notebooks

Drop the notebooks you want to include in the published book here.

Recommended workflow:

1. Copy or move one notebook into this folder.
2. Update the concept pages in `book/concepts/` to point to the notebook file.
3. Build the book from `book/`:
   ```bash
   cd book
   jupyter-book build --site .
   ```

If you want, I can also add a small script that scans the root repo and creates a list of candidate notebooks for the book.
