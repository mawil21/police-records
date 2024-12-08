import os
import json
from doctr.io import DocumentFile
from doctr.models import kie_predictor

# Paths
input_dir = "../../tst/saved_pdfs"
output_dir = "../../tst/phrase_extraction_analysis/docTR_pipeline/output"
os.makedirs(output_dir, exist_ok=True)

# Initialize the KIE model with recommended architectures
model = kie_predictor(
    det_arch='db_resnet50',
    reco_arch='crnn_vgg16_bn',
    pretrained=True
)

for root, dirs, files in os.walk(input_dir):
    for filename in files:
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(root, filename)

            # Load the PDF using DocumentFile
            doc = DocumentFile.from_pdf(pdf_path)

            # Process with the KIE model
            result = model(doc)

            pdf_output_data = {
                "file": filename,
                "pages": []
            }

            # Iterate through pages
            for i, page in enumerate(result.pages):
                # page.predictions is a dict: {class_name: [Prediction objects]}
                serializable_predictions = {}
                for class_name, pred_list in page.predictions.items():
                    serializable_preds = []
                    for pred in pred_list:
                        # Convert Prediction object into a dictionary
                        # Assuming pred.value is a string and pred.geometry is a tuple
                        serializable_preds.append({
                            "value": pred.value,
                            "geometry": list(pred.geometry)  # Convert tuple to list
                        })
                    serializable_predictions[class_name] = serializable_preds

                page_data = {
                    "page_number": i + 1,
                    "predictions": serializable_predictions
                }

                pdf_output_data["pages"].append(page_data)

            # Save JSON output
            json_filename = f"{os.path.splitext(filename)[0]}_kie.json"
            json_path = os.path.join(output_dir, json_filename)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(pdf_output_data, f, indent=4)

print("Processing complete. Results saved in:", output_dir)
