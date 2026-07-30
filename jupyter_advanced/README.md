# Tutorials


## Overview


In this folder several tutorials are provided on how to use STELLAR and apply it for a specific testing problem. The tutorials can be directly executed using the Jupyter Notebook environment to be installed in the following step.

- [`01_getting_started`](01_getting_started.ipynb): Installation guide and test experiments to verify that the installation is correct.

- [`02_navi`](02_navi.ipynb): Run experiments for the navigational use case.

- [`03_safety`](03_safety.ipynb): Run experiments for the safety use case.

- [`04_dashboard`](04_dashboard.ipynb): Start the interactive Streamlit dashboard to analyze testing results.

In case you are unable to run notebook `02_navi` and `03_safety` to obtain experimental results, we provide examples in `result_examples` directory. You can use the example outputs to visualize in the dashboard in `04_dashboard`.


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

