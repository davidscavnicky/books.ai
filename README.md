# booksai
V0.0.0 - smoke test


## Installation

Install from PyPI (if available) or from source.

Install the package directly from PyPI:

```bash
$ pip install booksai
```

## Usage

### Running the tests

```bash
$ pip install .[test]
```

Run `pytest` in the `tests` folder.

### Building the documentation

The documentation is built with Sphinx and lives in the `docs/` directory.

1. Install documentation dependencies:

```bash
pip install .[docs]
```

2. Ensure `pandoc` is available (used by some Sphinx extensions). With conda:

```bash
conda install pandoc
```

3. Build the docs:

```bash
cd docs
make html
```

Could run `make clean` before you run `make html`.

The generated HTML will appear in `docs/_build/html`.

### GitOps

Set hooks only if .githooks/ exists.
Create an initial tag only if missing (v0.0.0).
Make gh-pages branch.

## Examples and notebooks

There are example notebooks in `example_notebooks/` to demonstrate usage and to provide quick-start examples.

## GitHub

Set up GitHub Pages to host the documentation (public repository)
https://github.com/"your githubpage"/"Repo Name"/settings
and select the branch "gh-pages" and dir "root"

either or:

Change these settings under Settings-Action-General.
Workflow Permissions: Read and Write
Allow GitHub Actions to approve pull requests
