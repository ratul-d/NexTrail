import torch
import torch.nn as nn
import pickle

# Loads saved preprocessing objects
with open("mlb.pkl", "rb") as f:
    mlb = pickle.load(f)
with open("role_mappings.pkl", "rb") as f:
    role_to_idx, idx_to_role = pickle.load(f)
with open("job_skills.pkl", "rb") as f:
    job_skills_dict = pickle.load(f)

# Load trained model
input_size = len(mlb.classes_)  # Gets input size dynamically
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

# Loads trained model
model = MLP(input_size, hidden_size, output_size)
model.load_state_dict(torch.load("skill_to_job_model.pth"))
model.eval()


# Optimized prediction function
def predict_roles_and_suggest_skills(model, input_skills, mlb, idx_to_role, job_skills_dict, threshold=2,
                                     max_suggestions=8, total_suggestions=5):
    import torch
    input_vector = torch.tensor(mlb.transform([input_skills]), dtype=torch.float32)
    output = torch.sigmoid(model(input_vector)).detach().numpy()[0]  # Apply sigmoid manually

    strong_roles = []
    other_suggested_roles = set()
    skill_recommendations = {}

    for i, score in enumerate(output):
        role = idx_to_role[i]
        required_skills = job_skills_dict.get(role, set())

        matched_skills = set(input_skills) & required_skills
        missing_skills = required_skills - set(input_skills)

        # If the role is strongly matched, it adds to strong roles
        if len(matched_skills) >= threshold:
            strong_roles.append(role)
        # If the role is close to being unlocked, suggests missing skills
        elif len(matched_skills) >= threshold - 1 and missing_skills:
            skill_recommendations[role] = list(missing_skills)[:max_suggestions]  # Suggest a few key missing skills
            other_suggested_roles.add(role)

    # Limits strong roles to at most 5
    strong_roles = strong_roles[:5]
    remaining_slots = total_suggestions - len(strong_roles)

    # Limits other suggested roles to fill remaining slots
    other_suggested_roles = list(other_suggested_roles)[:remaining_slots]

    print("Roadmap suggestions:")
    if strong_roles:

        for role in strong_roles:
            print(f"- {role}")

    if remaining_slots > 0 and other_suggested_roles:
        for role in other_suggested_roles:
            print(f"- {role}")

    return strong_roles, other_suggested_roles

# Example usage
#input_skills = ['Python','Java','C++']
#input_skills = ['Python','Machine Learning','NumPy','Pytorch']
input_skills = ['Shader Programming (HLSL/GLSL)','3D Graphics','C++']
#input_skills = ['Python','Flask','SQL']
#input_skills = ['DSA', 'REST APIs', 'PostgreSQL', 'FastAPI','Python','Java','Spring Boot']
#input_skills = ['Cloud Security', 'Threat Hunting', 'Network Security', 'Malware Analysis','Backup Solutions','Networking (TCP/IP, DNS, DHCP)']

#input_skills = ['Python','Flask','SQL','FastAPI','C++','Ruby']
predict_roles_and_suggest_skills(model, input_skills, mlb, idx_to_role, job_skills_dict)

