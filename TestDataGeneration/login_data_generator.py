from sentence_transformers import SentenceTransformer
import numpy as np
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOO_API_KEY"))
model = SentenceTransformer("all-MiniLM-L6-v2")

gen_config = {
    "temperature": 0.8,
    "top_p": 1,
    "top_k": 1,
}
# Initialize the Gemini model for test data generation
Gemini = genai.GenerativeModel(model_name="gemini-2.5-flash",generation_config = gen_config)

prompt = """
You are a QA engineer. Generate test input data for a login page in JSON array format.
Each object should contain:
- email
- password
- description (e.g., "Valid credentials", "Missing email", "Password too short")
- is_valid (true or false to indicate if the input should pass or fail validation)

Include:
- Valid inputs
- Invalid formats
- Edge cases (empty fields, long passwords, SQL injection, etc.)
"""

response = Gemini.generate_content(prompt)

os.makedirs("testdata", exist_ok=True)

output_path = "testdata/login_test_data.json"
with open(output_path, "w", encoding="utf-8") as file:
    file.write(response.text)

# Print the generated test data
print("\nGenerated Login Test Data:\n")
print(response.text)
