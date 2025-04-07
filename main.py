import ollama
import os
import sys
from pydantic import BaseModel, Field
from pprint import pprint
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Ollama API docs: https://github.com/ollama/ollama/blob/main/docs/api.md

class MedicalReportSummary(BaseModel):
    patient_name: str = Field(..., description="The patient's full name.")
    date_of_birth: str = Field(..., description="The patient's date of birth (YYYY-MM-DD).")
    medical_record_number: str = Field(..., description="The patient's Medical Record Number (MRN).")
    date_of_report: str = Field(..., description="The date the report was generated (YYYY-MM-DD).")
    report_summary: str = Field(..., description="A medically concise summary of the medical report.")

    def __str__(self):
        return (
            f"Patient Name: {self.patient_name}\n"
            f"Date of Birth: {self.date_of_birth}\n"
            f"MRN: {self.medical_record_number}\n"
            f"Report Date: {self.date_of_report}\n"
            f"Summary: {self.report_summary}"
        )

def get_largest_model():
    """Retrieves the largest available Ollama model."""
    try:
        models = ollama.list()['models']
        if not models:
            return None
    
        # Sort models by size (assuming size is in bytes)
        biggest_model = max(models, key=lambda model: model.get('size', 0))['model']
        logging.info(f"Using: {biggest_model}")
        return biggest_model
    except Exception as e:
        logging.error(f"Error getting models: {e}")
        return None
    
def process_file_with_ollama_pydantic(file_path, model_name, instructions):
    """
    Sends file content to Ollama, forces output to a Pydantic schema, and prints the result.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()

        schema = json.dumps(MedicalReportSummary.model_json_schema(), indent=2) #Correct way to get schema.

        prompt = f"{instructions}\n\nFile Content:\n{file_content}\n\nReturn the data in JSON format"

        response = ollama.chat(
            model=model_name,
            stream=False,
 	    format=MedicalReportSummary.model_json_schema(),
            keep_alive="10m",
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ],
        )
        logging.info("Ollama API call successful.")

        try:
            json_response = json.loads(response['message']['content'])
            summary_object = MedicalReportSummary(**json_response)
            logging.info("JSON response successfully parsed.")

            return summary_object

        except json.JSONDecodeError:
            logging.error(f"Ollama did not return valid JSON. Response:\n{response['message']['content']}")
        except Exception as e:
            logging.error(f"Error parsing JSON response: {e}")

    except FileNotFoundError:
        logging.error(f"File not found at {file_path}")
    except Exception as e:
        logging.error(f"An error occurred during file processing: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    model_name = get_largest_model()
    if not model_name:
        logging.error("No Ollama models found. Exiting.")
        sys.exit(1)

    instructions = """Extract to JSON the:
        patient's name,
        date of birth, 
        medical record number (MRN), 
        the date of the report, 
        and provide a short summary of the medical report. Respond only with valid JSON. Do not write an introduction or summary.
"""

    obj = process_file_with_ollama_pydantic(file_path, model_name, instructions)

    if obj:
        print("Parsed Medical Report Summary:")
        print(obj.model_dump_json(indent=2))
    else:
        logging.error("Failed to parse the medical report.")
