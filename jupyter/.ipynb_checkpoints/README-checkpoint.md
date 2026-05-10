# Tutorials

In this folder several tutorials are provided on how to use STELLAR and apply it for a specific testing problem. The tutorials can be directly executed using the Jupyter Notebook environment.

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

Run the notebooks to setup and run the tool:

- `01_getting_started`: Installation guide and mock experiments to verify that the installation is correct.

- `02_navi`: Run experiments for the navigational use case.

- `03_safety`: Run experiments for the safety use case.

- `04_dashboard`: Start the interactive Streamlit dashboard for exploring results.

In case you are unable to run notebooks 2/3 to obtain experimental results, we provide examples in `result_examples` directory. You can use them to investigate the dashboard.