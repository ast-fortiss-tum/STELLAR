# Testing Natural Language Understanding of a Conversational In-Car Chatbot (TAROT 2026 - Hands-On) 🚗

## Overview

In this tutorial you will be able to implement your own test generator to test automatically a task-oriented conversational in-car chatbot.
Follow the instructions in the notebooks below after installing the jupyter environment in the following step:

- [`01_getting_started`](01_getting_started.ipynb): Installation guide.

- [`02_test_chatbot`](02_test_chatbot.ipynb): Apply STELLAR to test natural language understanding of a task oriented chatbot.

## Jupyter Installation

To run the Jupyter notebooks install the Jupyter environment.

Following commands are required to install the Jupyter Notebook environment (on Linux, where Python 3.11 is installed):

First create a virtual environment. You can use your preferred env manager or employ virtualenv:

```bash
python -m pip install virtualenv
python -m virtualenv venv
```

Activate virtual environment:

```bash
source venv/bin/activate
```

Then install the classical [Jupyter](https://jupyter.org/install) Notebook:

```bash
pip install notebook
```

Install the python kernel:

```bash
python -m pip install ipykernel
```

Register the virtual environment as a kernel:

```bash
python3 -m ipykernel install --user --name=venv
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Select *Kernel > Change Kernel > venv*.

