import json
from google.cloud import storage, translate_v2 as translate

# === CONFIGURATION ===
BUCKET_NAME = "vn-ppa-template"
OUTPUT_PREFIX = "vision-output/"
LOCAL_OUTPUT_FILE = "translated_text.txt"

storage_client = storage.Client()
translate_client = translate.Client()

def extract_ocr_text(bucket_name, prefix):
    print("Downloading OCR output JSON files...")
    bucket = storage_client.get_bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    full_text = ""
    for blob in blobs:
        if not blob.name.endswith(".json"):
            continue

        print(f"Reading: {blob.name}")
        data = json.loads(blob.download_as_text())
        responses = data.get("responses", [])
        for response in responses:
            annotation = response.get("fullTextAnnotation", {})
            text = annotation.get("text", "")
            if text:
                full_text += text + "\n"

    print("✅ Finished extracting OCR text")
    return full_text

def translate_text(text, source_lang='vi', target_lang='en'):
    print("Translating text...")
    CHUNK_SIZE = 4500
    chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    
    translated_chunks = []
    for i, chunk in enumerate(chunks):
        print(f" - Translating chunk {i + 1}/{len(chunks)}...")
        result = translate_client.translate(chunk, source_language=source_lang, target_language=target_lang)
        translated_chunks.append(result["translatedText"])
    
    print("✅ Translation complete")
    return "\n\n".join(translated_chunks)

def save_translation_to_file(text, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"📄 Saved translated text to: {filename}")

# === Run ===
vietnamese_text = extract_ocr_text(BUCKET_NAME, OUTPUT_PREFIX)
translated_text = translate_text(vietnamese_text)
save_translation_to_file(translated_text, LOCAL_OUTPUT_FILE)
