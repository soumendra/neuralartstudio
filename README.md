[![pypi neuralartstudio version](https://img.shields.io/pypi/v/neuralartstudio.svg)](https://pypi.python.org/pypi/neuralartstudio)
[![Conda neuralartstudio version](https://img.shields.io/conda/v/neuralartstudio/neuralartstudio.svg)](https://anaconda.org/neuralartstudio/neuralartstudio)
[![neuralartstudio python compatibility](https://img.shields.io/pypi/pyversions/neuralartstudio.svg)](https://pypi.python.org/pypi/neuralartstudio)
[![neuralartstudio license](https://img.shields.io/pypi/l/neuralartstudio.svg)](https://pypi.python.org/pypi/neuralartstudio)

[![latest release date](https://img.shields.io/github/release-date/soumendra/neuralartstudio.svg)](https://pypi.python.org/pypi/neuralartstudio)
[![latest release version](https://img.shields.io/github/release/soumendra/neuralartstudio.svg)](https://pypi.python.org/pypi/neuralartstudio)
[![issue count](https://img.shields.io/github/issues-raw/soumendra/neuralartstudio.svg)](https://pypi.python.org/pypi/neuralartstudio)
[![open pr count](https://img.shields.io/github/issues-pr-raw/soumendra/neuralartstudio.svg)](https://pypi.python.org/pypi/neuralartstudio)
[![last commit at](https://img.shields.io/github/last-commit/soumendra/neuralartstudio.svg)](https://pypi.python.org/pypi/neuralartstudio)
[![contributors count](https://img.shields.io/github/contributors/soumendra/neuralartstudio.svg)](https://pypi.python.org/pypi/neuralartstudio)

# Getting started

Latest docs at https://neuralartstudio.readthedocs.io/en/latest/

Install `neuralartstudio`

```bash
pip install neuralartstudio
```

- It is possible to create meaningful output with a cpu.
- A gpu can speed up the experiments.

## Running as a streamlit app

Create a `streamlit` app (say in a file called `main.py`)

```python
from neuralartstudio.streamlit import neural_style_transfer

neural_style_transfer(
    contentpath="assets/dancing.jpg", stylepath="assets/picasso.jpg",
)
```

**Note**: You have to provide paths to your content and style image to start with. You can replace the content image later in the app.

That's it. Now run the app with streamlit

```bash
streamlit run main.py
```

## Running from a python program

**ToDo**

## Running in a Jupyter notebook

**ToDo**
