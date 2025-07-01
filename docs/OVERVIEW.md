# Repository Overview

This document summarises the code base and highlights the main modules.

## PDP8 Mapping Utilities (`MCCF/PDP8`)

These scripts generate interactive maps of Vietnam's power infrastructure using Folium and GeoPandas. Key components include:

- **`create_map.py`** – builds various map layers (projects, substations, transmission lines) and writes HTML output.
- **`map_utils.py`** – shared helpers for reading data sources, caching, and styling map features.
- **`utils/`** – extraction tools for reading geo packages, PDF tables, and solar/wind data.
- **`config.py`** – file paths for input datasets and output locations.

Running `python MCCF/PDP8/create_map.py` produces HTML maps under `MCCF/PDP8/results`.

## Translation & API Tools (`MCCF/API tools`)

Scripts in this folder assist with translating spreadsheet content and processing OCR results.

- **`translation_script.py`** – builds a custom Vietnamese translation dictionary from a CSV and can apply it to translate data.
- **`apply_translation_dict.py`** – applies a prepared dictionary to a CSV for consistent translations.
- **`translate_with_gcp_openai_v1_fixed.py`** – translates Excel sheets using Google Cloud Translate or optionally OpenAI GPT.
- **`run_vision_ocr.py`** and **`vision_api.py`** – use Google Vision API to perform OCR on PDFs stored in Cloud Storage and translate the extracted text.

These scripts depend on valid credentials for the respective APIs and are executed directly from the command line.

## Additional Files

- `Model_Translation_Dictionary.txt` – example output from the translation tools.
- `requirements.txt` and `environment.yaml` – Python dependencies.
- `setup.py` – packaging configuration exposing `mccf-map` as a console entry point.

Use these documents together with the root [README](../README.md) to understand the repository structure.
