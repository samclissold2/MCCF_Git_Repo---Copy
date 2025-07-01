# translate_sheet.py
import pandas as pd
from googletrans import Translator
import logging
import time
import json
import re
from collections import Counter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('translation.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def extract_english_words(text):
    """Extract English words from text using regex"""
    if pd.isna(text) or text == 'nan' or text == '':
        return []
    
    # Convert to string and find English words (letters, numbers, spaces, common punctuation)
    text = str(text).strip()
    # Split on common delimiters and filter out non-English words
    words = re.findall(r'\b[a-zA-Z0-9\s\-\.\,\%\$\€\£\¥\()]+', text)
    # Clean up words
    cleaned_words = []
    for word in words:
        word = word.strip()
        if len(word) > 1 and not word.isdigit():  # Skip single characters and pure numbers
            cleaned_words.append(word)
    return cleaned_words

def translate_with_retry(translator, text, src='en', dest='vi', max_retries=3):
    """Translate text with retry logic and error handling"""
    for attempt in range(max_retries):
        try:
            logger.debug(f"Translating: {text[:50]}... (attempt {attempt + 1})")
            result = translator.translate(text, src=src, dest=dest)
            return result.text
        except Exception as e:
            logger.warning(f"Translation attempt {attempt + 1} failed for '{text[:50]}...': {e}")
            if attempt == max_retries - 1:
                logger.error(f"Failed to translate after {max_retries} attempts: {text[:50]}...")
                return text  # Return original text if all attempts fail
            time.sleep(1)  # Wait before retrying

def create_translation_dictionary():
    """Create a dictionary of English words from the CSV file"""
    logger.info("Creating translation dictionary from CSV file...")
    
    # Load CSV file
    df = pd.read_csv("Operating Assumptions_complete.csv")
    logger.info(f"CSV loaded successfully. Shape: {df.shape}")
    
    # Extract all English words from the dataframe
    all_words = []
    for col in df.columns:
        for value in df[col]:
            words = extract_english_words(value)
            all_words.extend(words)
    
    # Count word frequencies
    word_counts = Counter(all_words)
    
    # Create dictionary with Google Translate
    logger.info("Translating unique words...")
    translator = Translator()
    translation_dict = {}
    
    unique_words = list(word_counts.keys())
    for i, word in enumerate(unique_words):
        if i % 50 == 0:  # Log progress every 50 words
            logger.info(f"Translating word {i+1}/{len(unique_words)}: {word}")
        
        try:
            translated = translate_with_retry(translator, word, src='en', dest='vi')
            translation_dict[word] = {
                'vietnamese': translated,
                'frequency': word_counts[word],
                'manual_override': False
            }
            time.sleep(0.1)  # Rate limiting
        except Exception as e:
            logger.warning(f"Failed to translate '{word}': {e}")
            translation_dict[word] = {
                'vietnamese': word,  # Keep original if translation fails
                'frequency': word_counts[word],
                'manual_override': False
            }
    
    # Save dictionary to JSON file
    with open('translation_dictionary.json', 'w', encoding='utf-8') as f:
        json.dump(translation_dict, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Translation dictionary created with {len(translation_dict)} words")
    logger.info("Dictionary saved to 'translation_dictionary.json'")
    logger.info("Please review and edit the dictionary file, then run the script again with --use-dictionary flag")
    
    return translation_dict

def load_translation_dictionary():
    """Load the translation dictionary from JSON file"""
    try:
        with open('translation_dictionary.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("Translation dictionary not found. Please run the script first to create it.")
        return None

def translate_with_dictionary(text, translation_dict):
    """Translate text using the custom dictionary"""
    if pd.isna(text) or text == 'nan' or text == '':
        return text
    
    text = str(text)
    translated_text = text
    
    # Replace words in order of length (longest first to avoid partial matches)
    sorted_words = sorted(translation_dict.keys(), key=len, reverse=True)
    
    for word in sorted_words:
        if word in translated_text:
            vietnamese = translation_dict[word]['vietnamese']
            translated_text = translated_text.replace(word, vietnamese)
    
    return translated_text

def translate_csv_with_dictionary():
    """Translate the CSV using the custom dictionary"""
    logger.info("Loading translation dictionary...")
    translation_dict = load_translation_dictionary()
    if not translation_dict:
        return
    
    logger.info("Loading CSV file...")
    df = pd.read_csv("Operating Assumptions_complete.csv")
    logger.info(f"CSV loaded successfully. Shape: {df.shape}")
    
    # Translate data in each column
    logger.info("Starting translation with custom dictionary...")
    total_cells = len(df.columns) * len(df)
    current_cell = 0
    
    for col_idx, col in enumerate(df.columns):
        logger.info(f"Translating column {col_idx+1}/{len(df.columns)}: {col}")
        
        for row_idx, value in enumerate(df[col]):
            current_cell += 1
            if current_cell % 100 == 0:  # Log every 100 cells
                logger.info(f"Progress: {current_cell}/{total_cells} cells processed ({current_cell/total_cells*100:.1f}%)")
            
            translated_value = translate_with_dictionary(value, translation_dict)
            df.at[row_idx, col] = translated_value
        
        logger.info(f"Completed column {col_idx+1}/{len(df.columns)}: {col}")
    
    # Save translated data
    logger.info("Saving translated data...")
    df.to_csv("Operating Assumptions_translated.csv", index=False)
    logger.info("✅ Translation complete. File saved as 'Operating Assumptions_translated.csv'")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--use-dictionary":
        # Use existing dictionary
        translate_csv_with_dictionary()
    else:
        # Create new dictionary
        create_translation_dictionary()
