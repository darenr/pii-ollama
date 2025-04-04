import ollama
import os
import sys
from pydantic import BaseModel, Field
from pprint import pprint
import json

class MedicalReportSummary(BaseModel):
    patient_name: str = Field(..., description="The patient's full name.")
    date_of_birth: str = Field(..., description="The patient's date of birth (YYYY-MM-DD).")
    medical_record_number: str = Field(..., description="The patient's Medical Record Number (MRN).")
    date_of_report: str = Field(..., description="The date the report was generated (YYYY-MM-DD).")
    report_summary: str = Field(..., description="A concise summary of the medical report.")

    def __str__(self):
        return (
            f"Patient Name: {self.patient_name}\n"
            f"Date of Birth: {self.date_of_birth}\n"
            f"MRN: {self.medical_record_number}\n"
            f"Report Date: {self.date_of_report}\n"
            f"Summary: {self.report_summary}"
        )

def process_file_with_ollama_pydantic(file_path, model_name, instructions):
    """
    Sends file content to Ollama, forces output to a Pydantic schema, and prints the result.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()

        prompt = f"{instructions}\n\nFile Content:\n{file_content}\n\nReturn the data in JSON format that matches the following schema:\n{MedicalReportSummary.schema_json(indent=2)}"

        response = ollama.chat(
            model=model_name,
            stream=False,
            format='json',
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ],
        )

        try:
            json_response = json.loads(response['message']['content'])
            summary_object = MedicalReportSummary(**json_response)
            print(summary_object) # uses the __str__ function

        except json.JSONDecodeError:
            print(f"Error: Ollama did not return valid JSON. Response:\n{response['message']['content']}")
        except Exception as e:
            print(f"Error: {e}")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    model_name = "gemma3:12b"
    instructions = """Extract to JSON the:
        patient's name,
        date of birth, 
        medical record number (MRN), 
        the date of the report, 
        and provide a short summary of the medical report. Respond only with valid JSON. Do not write an introduction or summary.
"""

    process_file_with_ollama_pydantic(file_path, model_name, instructions)