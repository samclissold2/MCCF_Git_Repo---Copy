# apply_translation_dict.py

import pandas as pd
import json
import logging

# Setup
INPUT_CSV = "Operating Assumptions_complete.csv"
DICTIONARY_PATH = "translation_dictionary_gcp.json"
OUTPUT_CSV = "Operating Assumptions_translated_corrected.csv"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load translation dictionary
def load_translation_dict():
    with open(DICTIONARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# Apply dictionary to DataFrame
def apply_dictionary(df, translation_dict):
    def translate_cell(val):
        val_str = str(val).strip()
        return translation_dict.get(val_str, val)

    return df.applymap(translate_cell)

def main():
    logging.info("🔁 Applying translation dictionary...")
    df = pd.read_csv(INPUT_CSV)
    translation_dict = load_translation_dict()
    translated_df = apply_dictionary(df, translation_dict)
    translated_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    logging.info(f"✅ Corrected translation saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
