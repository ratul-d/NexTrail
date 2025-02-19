import os
import re
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allows frontend to communicate with backend

# Load preprocessing objects
with open("mlb.pkl", "rb") as f:
    mlb = pickle.load(f)
with open("role_mappings.pkl", "rb") as f:
    role_to_idx, idx_to_role = pickle.load(f)
with open("job_skills.pkl", "rb") as f:
    job_skills_dict = pickle.load(f)

# Model configuration
input_size = len(mlb.classes_)
hidden_size = 128
output_size = len(idx_to_role)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize and load model
model = MLP(input_size, hidden_size, output_size)
model.load_state_dict(torch.load("skill_to_job_model.pth"))
model.eval()

def predict_roles(input_skills, threshold=2, max_suggestions=8, total_suggestions=5):
    input_vector = torch.tensor(mlb.transform([input_skills]), dtype=torch.float32)
    output = torch.sigmoid(model(input_vector)).detach().numpy()[0]

    strong_roles = []
    other_suggested_roles = set()
    skill_recommendations = {}

    for i, score in enumerate(output):
        role = idx_to_role[i]
        required_skills = job_skills_dict.get(role, set())
        matched_skills = set(input_skills) & required_skills
        missing_skills = required_skills - set(input_skills)
        if len(matched_skills) >= threshold:
            strong_roles.append(role)
        elif len(matched_skills) >= threshold - 1 and missing_skills:
            skill_recommendations[role] = list(missing_skills)[:max_suggestions]
            other_suggested_roles.add(role)

    strong_roles = strong_roles[:5]
    remaining_slots = total_suggestions - len(strong_roles)
    roadmap_suggestions = list(other_suggested_roles)[:remaining_slots]

    print("Strong Roles:")
    for role in strong_roles:
        print(f"- {role}")
    print("Roadmap Suggestions:")
    for role in roadmap_suggestions:
        print(f"- {role}")

    return strong_roles, roadmap_suggestions, skill_recommendations

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_skills = data.get("skills", [])
    if not input_skills:
        return jsonify({"error": "No skills provided"}), 400

    strong_roles, roadmap_suggestions, skill_recommendations = predict_roles(input_skills)
    return jsonify({
        "strong_roles": strong_roles,
        "roadmap_suggestions": roadmap_suggestions,
        "skill_recommendations": skill_recommendations
    })

# Endpoint to generate a roadmap using your Hugging Face LLM
@app.route('/roadmap', methods=['POST'])
def get_roadmap():
    data = request.json
    target_role = data.get("role", "")
    if not target_role:
        return jsonify({"error": "No role provided"}), 400

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    from langchain_community.llms import HuggingFaceHub
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-Small-24B-Instruct-2501",
        model_kwargs={
            "temperature": 0.7,
            "max_new_tokens": 1500
        }
    )

    prompt = (
        f"Generate a small roadmap of skills to learn to become a {target_role}. "
        "Provide only the skill headings as steps from beginner to advanced, numbered from 1. to 10., with no additional text. Ans:"
    )

    try:
        full_response = llm(prompt)
        # Clean the response if the prompt is echoed
        if full_response.startswith(prompt):
            response = full_response[len(prompt):].strip()
        else:
            response = full_response.strip()

        return jsonify({"roadmap": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- New endpoint using old project logic ---
def extract_steps(text: str):
    # Split on numbers followed by a dot (e.g., "1. ", "2. ", etc.)
    steps = re.split(r'\d+\.\s*', text)
    # Remove empty strings
    steps = [step.strip() for step in steps if step.strip()]
    return steps

@app.route('/process', methods=['POST'])
def process_text():
    data = request.get_json()
    text = data.get("text", "")
    steps = extract_steps(text)
    nodes = []
    edges = []
    for i, step in enumerate(steps):
        node_id = chr(65 + i)  # A, B, C, etc.
        nodes.append({"id": node_id, "label": step, "type": "action"})
        if i > 0:
            edges.append({"from": chr(65 + i - 1), "to": node_id})
    return jsonify({"nodes": nodes, "edges": edges})

if __name__ == '__main__':
    app.run(debug=True)
