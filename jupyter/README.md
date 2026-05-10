# Tutorials

In this folder several tutorials are provided of how to use STELLAR and apply if for a specific testing problem. The tutorials can be directly executed using the Jupyter Notebook environment. 

Following commands are required to install the Jupyter Notebook environment (On Linux, where Python3.11 is installed):

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

Register the virtual environment at kernel:

```bash
python3 -m ipykernel install --user --name=venv
```

Start jupyter notebook:

```bash
jupyter notebook
```

Select *Kernel > Change Kernel > venv*.

Run the notebooks starting from the first one to setup and run the tool.