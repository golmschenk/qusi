% qusi documentation master file, created by
% sphinx-quickstart on Sun Dec 17 00:13:36 2023.
% You can adapt this file completely to your liking, but it should at least
% contain the root `toctree` directive.

# Welcome to qusi's documentation!

## Installation

To install `qusi`, use

```shell
pip install qusi
```

Although not required, as is generally good practice for any development project, we highly recommend creating a separate virtual environment for each distinct project. For example, via Conda, creating a virtual environment for a project using `qusi` might look like

```
conda create -n project_using_qusi_env python=3.11
```

Then before working, be sure to activate your environment with

```shell
conda activate project_using_qusi_env
```

Then install `qusi` within this environment.

```{toctree}
:caption: 'Contents:'
:maxdepth: 2

tutorials/basic_transit_identification_with_prebuilt_components
tutorials/basic_transit_identification_dataset_construction
tutorials/crafting_standard_datasets
tutorials/crafting_injected_datasets
tutorials/advanced_crafting_datasets
reference_index
```

# Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
