# ============================================================
#  Project 3: AI Recommendation Logic
#  Algorithm : Content-Based Filtering
#             (TF-IDF Vectorization + Cosine Similarity)
#  Dataset   : Tech Stack / Job Roles
#  Author    : DecodeLabs Intern | Batch 2026
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# STEP 1 — DATASET  (Job Roles as "Items")
# ============================================================
job_roles = [
    {"title": "Data Scientist",
     "skills": "python machine_learning statistics sql data_analysis numpy pandas scikit_learn deep_learning tensorflow"},
    {"title": "Machine Learning Engineer",
     "skills": "python machine_learning deep_learning tensorflow pytorch model_deployment docker kubernetes mlops"},
    {"title": "Data Analyst",
     "skills": "sql python data_analysis excel power_bi tableau statistics reporting pandas visualization"},
    {"title": "Backend Developer",
     "skills": "python java sql apis rest_api django flask nodejs databases postgresql mongodb"},
    {"title": "Frontend Developer",
     "skills": "javascript html css react angular vuejs typescript ui_design responsive_design nodejs"},
    {"title": "DevOps Engineer",
     "skills": "docker kubernetes aws linux bash ci_cd jenkins git cloud automation terraform"},
    {"title": "Cloud Architect",
     "skills": "aws azure gcp cloud docker kubernetes terraform infrastructure devops microservices linux"},
    {"title": "AI Research Scientist",
     "skills": "python deep_learning pytorch tensorflow research mathematics statistics nlp computer_vision transformers"},
    {"title": "Cybersecurity Analyst",
     "skills": "networking linux security penetration_testing firewalls encryption python bash ethical_hacking siem"},
    {"title": "Full Stack Developer",
     "skills": "javascript python react nodejs sql html css rest_api docker git postgresql mongodb"},
    {"title": "NLP Engineer",
     "skills": "python nlp transformers huggingface bert gpt text_processing deep_learning pytorch tensorflow"},
    {"title": "Business Intelligence Developer",
     "skills": "sql power_bi tableau data_warehousing etl excel reporting analytics data_modeling statistics"},
    {"title": "Robotics Engineer",
     "skills": "python cpp ros robotics control_systems computer_vision sensors automation embedded_systems linux"},
    {"title": "Blockchain Developer",
     "skills": "solidity ethereum javascript python smart_contracts web3 cryptography distributed_systems nodejs"},
    {"title": "Systems Administrator",
     "skills": "linux windows bash networking servers cloud aws azure docker automation monitoring security"},
]

df = pd.DataFrame(job_roles)

print("=" * 60)
print("  PROJECT 3 — AI TECH STACK RECOMMENDER")
print("  Content-Based Filtering | TF-IDF + Cosine Similarity")
print("=" * 60)
print(f"\n📦 Dataset: {len(df)} Job Roles loaded")

# ============================================================
# STEP 2 — TF-IDF VECTORIZATION
# ============================================================
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["skills"])

print(f"✅ TF-IDF Matrix : {tfidf_matrix.shape[0]} roles × {tfidf_matrix.shape[1]} skill terms\n")

# ============================================================
# STEP 3 — RECOMMENDATION ENGINE (4-Step Pipeline)
# ============================================================
def recommend(user_skills: list, top_n: int = 3) -> pd.DataFrame:
    # Step 1: INGESTION
    user_profile = " ".join([s.strip().lower().replace(" ", "_") for s in user_skills])

    # Step 2: SCORING
    user_vector = vectorizer.transform([user_profile])
    scores      = cosine_similarity(user_vector, tfidf_matrix).flatten()

    # Step 3: SORTING
    ranked_indices = np.argsort(scores)[::-1]

    # Step 4: FILTERING — Top-N only
    results = []
    for idx in ranked_indices[:top_n]:
        results.append({
            "Rank"       : len(results) + 1,
            "Job Role"   : df.iloc[idx]["title"],
            "Score"      : round(scores[idx], 4),
            "Match %"    : f"{round(scores[idx] * 100, 1)}%",
        })
    return pd.DataFrame(results)

# ============================================================
# STEP 4 — DEMO WITH 3 USER PROFILES
# ============================================================
demo_users = [
    {"name": "User A — Data Enthusiast",
     "skills": ["python", "machine_learning", "statistics", "sql", "deep_learning"]},
    {"name": "User B — Cloud & DevOps Fan",
     "skills": ["aws", "docker", "kubernetes", "linux", "automation"]},
    {"name": "User C — Web Developer",
     "skills": ["javascript", "react", "nodejs", "html", "css"]},
]

all_results = {}
for user in demo_users:
    print("─" * 60)
    print(f"👤 {user['name']}")
    print(f"   Skills: {', '.join(user['skills'])}")
    recs = recommend(user["skills"], top_n=3)
    all_results[user["name"]] = recs
    print(recs.to_string(index=False))
    print()

# ============================================================
# STEP 5 — LIVE PREDICTION DEMO (fixed input)
# ============================================================
print("=" * 60)
print("  🎯 LIVE PREDICTION DEMO")
print("=" * 60)
live_skills = ["python", "nlp", "transformers", "pytorch", "deep_learning"]
live_recs   = recommend(live_skills, top_n=3)
print(f"\nInput Skills : {live_skills}")
print("\n🏆 Top 3 Recommended Career Paths:\n")
print(live_recs.to_string(index=False))

# ============================================================
# STEP 6 — VISUALIZATIONS
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    "Project 3 — AI Tech Stack Recommender Dashboard\n"
    "Content-Based Filtering | TF-IDF + Cosine Similarity",
    fontsize=13, fontweight="bold"
)

palette = ["#2196F3", "#4CAF50", "#FF5722"]

for ax, (uname, recs), color in zip(axes, all_results.items(), palette):
    roles  = recs["Job Role"].tolist()[::-1]
    scores = recs["Score"].tolist()[::-1]

    bars = ax.barh(roles, scores, color=color, alpha=0.85,
                   edgecolor="black", linewidth=0.6)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Cosine Similarity Score")
    short = uname.split("—")[1].strip()
    ax.set_title(short, fontsize=11, fontweight="bold")
    ax.axvline(x=0.3, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

    for bar, sc in zip(bars, scores):
        ax.text(bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{sc:.2f}", va="center", fontsize=10, fontweight="bold")

    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("recommendation_dashboard.png", dpi=150, bbox_inches="tight")
print("\n\n📊 Dashboard saved → recommendation_dashboard.png")
print("✅ Project 3 Complete!\n")
