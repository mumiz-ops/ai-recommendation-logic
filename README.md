#  AI Recommendation Logic (Tech Stack Recommender)

> **Industrial Training**

---

## 📌 Overview

This project builds a **Content-Based Filtering** recommendation engine that maps a user's skills to the most relevant tech career paths. It uses **TF-IDF vectorization** and **Cosine Similarity** to mathematically match user profiles with job roles — the same logic powering Netflix and Amazon recommendations.

---

## 🧠 Concepts Covered

| Concept | Description |
|---|---|
| Content-Based Filtering | Match user profile to item attributes |
| TF-IDF Vectorization | Convert skills to weighted numerical vectors |
| Cosine Similarity | Measure angular alignment between vectors |
| 4-Step Pipeline | Ingestion → Scoring → Sorting → Filtering |
| Cold Start Problem | Handled via explicit user skill input |

---

## 🔁 Project Pipeline (IPO Framework)

```
INPUT                    PROCESS                    OUTPUT
─────                    ───────                    ──────
User Skills         →    TF-IDF Vectorization  →    Top-3 Job Roles
(min. 3 skills)          Cosine Similarity          with Match Score %
```

---

## 📊 Dataset

- **15 Job Roles** as recommendation items
- **87 unique skill terms** in the vocabulary
- Roles include: Data Scientist, ML Engineer, DevOps, Cloud Architect, NLP Engineer, and more

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/mumiz-ops/ai-recommendation-logic.git
cd mumiz-ops-ai-recommendation-logic

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the project
python tech_stack_recommender.py
```

---

## 📦 Requirements

```
numpy
pandas
matplotlib
scikit-learn
```

---

## 📈 Sample Results

| User Profile | #1 Match | #2 Match | #3 Match |
|---|---|---|---|
| python, ml, sql, statistics | Data Scientist (60.9%) | ML Engineer (40.0%) | AI Researcher (30.2%) |
| aws, docker, kubernetes | DevOps Engineer (59.9%) | Cloud Architect (43.9%) | Sys Admin (42.9%) |
| javascript, react, nodejs | Full Stack Dev (67.3%) | Frontend Dev (63.4%) | Blockchain Dev (22.1%) |

---

## 📂 Project Structure

```
decodelabs-ai-project3/
├── tech_stack_recommender.py    # Main script
├── recommendation_dashboard.png # Output visualization
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

---

## 📊 Output Dashboard

![Recommendation Dashboard](recommendation_dashboard.png)

---

## 👤 Author

**Mumiz Ahmad**
📧 mumizahmad30@gmail.com
