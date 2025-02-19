import os
import re
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests for local development

# Endpoint: Generate a roadmap using the Hugging Face LLM based on the given job role
@app.route('/roadmap', methods=['POST'])
def get_roadmap():
    data = request.json
    target_role = data.get("role", "")
    if not target_role:
        return jsonify({"error": "No role provided"}), 400

    # Set your Hugging Face API token
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
        # Remove prompt echo if present
        if full_response.startswith(prompt):
            response = full_response[len(prompt):].strip()
        else:
            response = full_response.strip()
        return jsonify({"roadmap": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Utility function: Extract steps from the roadmap text based on numbering (e.g., "1. ", "2. ", etc.)
def extract_steps(text: str):
    steps = re.split(r'\d+\.\s*', text)
    steps = [step.strip() for step in steps if step.strip()]
    return steps

# Endpoint: Process the roadmap text into a set of nodes and edges for rendering a flowchart
@app.route('/process', methods=['POST'])
def process_text():
    data = request.get_json()
    text = data.get("text", "")
    steps = extract_steps(text)
    nodes = []
    edges = []
    for i, step in enumerate(steps):
        node_id = chr(65 + i)  # Convert 0->A, 1->B, etc.
        nodes.append({"id": node_id, "label": step, "type": "action"})
        if i > 0:
            edges.append({"from": chr(65 + i - 1), "to": node_id})
    return jsonify({"nodes": nodes, "edges": edges})

if __name__ == '__main__':
    app.run(debug=True)
