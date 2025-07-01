from google.cloud import vision_v1
from google.cloud import storage

# === CONFIGURATION ===
bucket_name = "vn-ppa-template"
pdf_filename = "2024_601 + 602_07-2024-TT-BCT..pdf"  # double dots!
gcs_source_uri = f"gs://{bucket_name}/{pdf_filename}"
gcs_destination_uri = f"gs://{bucket_name}/vision-output/"  # this folder will be created by GCP

def run_ocr():
    client = vision_v1.ImageAnnotatorClient()

    # Input configuration
    mime_type = "application/pdf"
    gcs_source = vision_v1.GcsSource(uri=gcs_source_uri)
    input_config = vision_v1.InputConfig(gcs_source=gcs_source, mime_type=mime_type)

    # Output configuration
    gcs_destination = vision_v1.GcsDestination(uri=gcs_destination_uri)
    output_config = vision_v1.OutputConfig(gcs_destination=gcs_destination, batch_size=1)

    feature = vision_v1.Feature(type_=vision_v1.Feature.Type.DOCUMENT_TEXT_DETECTION)

    async_request = vision_v1.AsyncAnnotateFileRequest(
        features=[feature],
        input_config=input_config,
        output_config=output_config,
    )

    print("Submitting OCR job...")
    operation = client.async_batch_annotate_files(requests=[async_request])
    operation.result(timeout=600)  # Wait up to 10 minutes

    print("OCR completed. Output written to:", gcs_destination_uri)

run_ocr()
