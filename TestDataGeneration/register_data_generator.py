from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# ========== Load Environment Variables ==========
load_dotenv()
genai.configure(api_key=os.getenv("GOO_API_KEY"))
model = SentenceTransformer("all-MiniLM-L6-v2")

# ========== Configuration for Gemini ==========
gen_config = {
    "temperature": 0.7,
    "top_k": 1,
    "top_p": 1
}

# ========== Initialize Gemini Model ==========
Gemini = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config=gen_config
)

# ========== Context Documents ==========
documents = [
    "The registration page collects the following fields from the user: first name, last name, phone number, country, and email address.",
    "Phone numbers must be in international format, e.g., +1-202-555-0173.",
    "Emails should look realistic and match the first and last names when possible.",
    "The system supports users from multiple countries like USA, UK, India, Australia, Canada.",
    "Generate at least 20 unique test data entries."
]

# ========== Query for Test Data Generation ==========
query = "You are a QA engineer. Create JSON test data entries for the registration form."
scenario = query

# ========== Retrieve Relevant Context ==========
# Compute embeddings for documents and the scenario
doc_embeddings = model.encode(documents)
scenario_embedding = model.encode([scenario])[0]

# Compute cosine similarity to find the most relevant context
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarities = [cosine_similarity(scenario_embedding, emb) for emb in doc_embeddings]
top_k = 3
top_indices = np.argsort(similarities)[-top_k:][::-1]
retrieved_context = "\n".join([documents[i] for i in top_indices])

# ========== Prompt ==========
prompt = f"""
Context:
{retrieved_context}

Task:
Generate 20 unique and realistic registration test data entries in JSON format. Each entry should have:
- first_name
- last_name
- phone (with country code)
- country
- email

The output must be a JSON array, properly formatted and parseable.
"""

# ========== Generate AI Response ==========
response = Gemini.generate_content(prompt)

output_dir = "testdata"
os.makedirs(output_dir, exist_ok=True)
output_path = "testdata/register_test_data.json"
with open(output_path, "w", encoding="utf-8") as file:
    file.write(response.text)

# ========== Print Output ==========
print("\nGenerated Login Test Data:\n")
print(response.text)