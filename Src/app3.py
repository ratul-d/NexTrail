from flask import Flask, render_template, jsonify, request
import requests
import random
import time

app = Flask(__name__)

# Job roles dataset
job_roles = [
    "Data Scientist", "Machine Learning Engineer", "Software Developer",
    "Cybersecurity Analyst", "Cloud Engineer", "DevOps Engineer",
    "AI Researcher", "Blockchain Developer", "Full-Stack Developer", "Data Engineer"
]

def fetch_job_trends():
    job_data = []
    for role in job_roles:
        job_data.append({
            "role": role,
            "trend": round(random.uniform(-5, 5), 2),  # Random increase/decrease
            "openings": random.randint(100, 5000)  # Simulated job count
        })
    return sorted(job_data, key=lambda x: x["trend"], reverse=True)  # Sort by highest trend

@app.route('/')
def home():
    return render_template("job.html")

@app.route('/job-trends')
def job_trends():
    return jsonify(fetch_job_trends())

@app.route('/search-job', methods=['GET'])
def search_job():
    query = request.args.get('job')
    trends = fetch_job_trends()
    job_details = next((job for job in trends if job['role'].lower() == query.lower()), None)
    return jsonify(job_details if job_details else {"error": "Job role not found"})

if __name__ == '__main__':
    app.run(debug=True)