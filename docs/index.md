# Getting started

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
