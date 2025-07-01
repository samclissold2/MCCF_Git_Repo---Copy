# MCCF Repository

This repository hosts mapping utilities and translation tools developed for the Mekong Centre for Climate Finance. The code is organised under the `MCCF` Python package and includes data processing helpers for the PDP8 plan as well as scripts that integrate with Google Cloud and OpenAI APIs.

## Key folders

- `MCCF/PDP8/` – utilities for building interactive maps of Vietnam's power sector.
- `MCCF/API tools/` – helpers for OCR and translation using Google APIs and optional GPT.

A detailed description of each component is available in [docs/OVERVIEW.md](docs/OVERVIEW.md).

## Setup

Install dependencies using `pip install -r requirements.txt` or create the conda environment from `environment.yaml`.

## Usage

Mapping and API scripts are typically run directly, for example:

```bash
python MCCF/PDP8/create_map.py      # build power sector maps
python 'MCCF/API tools/translation_script.py' --use-dictionary
```

Results are written to the `MCCF/PDP8/results` directory.
