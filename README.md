# 🎓 Learning Path Predictor

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/dysr7gyyvb/badge.svg)](https://ask.thili.ai/b/dysr7gyyvb)
[![CI Tests](https://github.com/eMohan07/learning-path-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/eMohan07/learning-path-predictor/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-v1.49-FF4B4B.svg)](https://streamlit.io/)

**Learning Path Predictor** is an EdTech sequence recommendation engine designed for students. It leverages **autoregressive language modeling** and **market basket co-occurrence analysis** (`lfretail-lfmarketbasket` techniques) to recommend personalized, logical next courses based on a student's completed academic history.

---

## 📌 Table of Contents

- [Overview & Goal](#-overview--goal)
- [System Architecture](#-system-architecture)
- [Course Catalog & Academic Tracks](#-course-catalog--academic-tracks)
- [Milestones & Progress](#-milestones--progress)
- [Streamlit Interface Features](#-streamlit-interface-features)
- [Mathematical Formulation](#-mathematical-formulation)
- [Quickstart & Installation](#-quickstart--installation)
- [Running Tests](#-running-tests)
- [Project Structure](#-project-structure)
- [Stretch Goals Achieved](#-stretch-goals-achieved)

---

## 🎯 Overview & Goal

- **Goal:** Build a sequential course predictor that suggests logical, high-relevance next courses to students based on completed academic coursework.
- **Done When:** The Streamlit interface displays top $N$ relevant course recommendations for a user's input of completed courses, respecting prerequisites and quantifying model confidence.

---

## 🧠 System Architecture

The predictor integrates sequential language modeling with market basket association mining:

```
+-------------------------------------------------------------------------------+
|                             Student Input Path                                |
|             e.g., ["CS101", "PY101", "MATH101", "DS201", "STAT201"]           |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
                    +-------------------+-------------------+
                    |                                       |
                    v                                       v
    +-------------------------------+       +-------------------------------+
    |  Autoregressive Language Model|       |  Market Basket Co-Occurrence  |
    |  • Embedding lookup (32-dim)  |       |  • Pairwise co-occurrences    |
    |  • Context window (k=3)       |       |  • Conditional probabilities  |
    |  • Non-linear hidden (ReLU)   |       |  • Normalized lift adjustment |
    |  • Softmax over catalog tokens|       |    Lift = P(cand|hist)/P(cand)|
    +---------------+---------------+       +---------------+---------------+
                    |                                       |
                    +-------------------+-------------------+
                                        |
                                        v
                    +---------------------------------------+
                    | Blended Candidate Scoring (α-weighted)|
                    | P_final = (1 - α) * P_LM + α * P_Lift |
                    +-------------------+-------------------+
                                        |
                                        v
                    +---------------------------------------+
                    |       Prerequisite DAG Validation     |
                    | • Filter courses already completed    |
                    | • Validate all prerequisites are met  |
                    +-------------------+-------------------+
                                        |
                                        v
                    +---------------------------------------+
                    |     Top-N Ranked Recommendations      |
                    |     (with confidence % & rationales)  |
                    +---------------------------------------+
```

---

## 📚 Course Catalog & Academic Tracks

The system models **28 realistic academic courses** structured across four primary specialization tracks:

1. **📊 Data Science & AI:**
   `PY101` (Python) ➔ `MATH101` (Linear Algebra) ➔ `DS201` (Data Analysis) ➔ `STAT201` (Probability & Stats) ➔ `ML301` (Machine Learning) ➔ `DL401` (Deep Learning) ➔ `NLP402` (NLP) / `CV403` (Computer Vision) ➔ `MLOPS501` (MLOps).
2. **🌐 Web & Software Engineering:**
   `CS101` (Intro to CS) ➔ `CS102` (Data Structures & Algorithms) ➔ `WEB101` (Web Dev) ➔ `DB201` (Relational Databases & SQL) ➔ `BE202` (Backend Engineering) ➔ `FE203` (Modern Frontend / React) ➔ `SYS301` (System Design).
3. **☁️ Cloud & Infrastructure / DevOps:**
   `NET101` (Networking) ➔ `OS201` (Linux & Operating Systems) ➔ `CLOUD201` (Cloud Foundations) ➔ `CONT301` (Docker & Kubernetes) ➔ `IAC302` (Terraform & CI/CD) ➔ `SRE401` (Site Reliability).
4. **🔒 Cybersecurity:**
   `NET101` (Networking) ➔ `OS201` (Linux) ➔ `SEC201` (Security Foundations) ➔ `SEC301` (Network Defense) ➔ `SEC401` (Ethical Hacking) ➔ `SEC402` (Cloud & App Security).

---

## ✅ Milestones & Progress

All five project milestones have been implemented, tested, and pushed in individual commits:

- [x] **1. Generate mock student course completion data as sequences of course IDs.**  
      Programmatically generates realistic student learning trajectories respecting prerequisite DAGs, primary track focus, and complementary cross-track electives.
- [x] **2. Pretrain a small language model on these course ID sequences.**  
      Built `CourseLanguageModel`, an autoregressive neural sequence model with token embeddings, Adam optimization, cross-entropy next-token loss, and perplexity tracking.
- [x] **3. Implement a function to predict the next likely course given a partial sequence.**  
      Implemented `predict_next_courses(...)` combining causal sequence likelihood and association lift, outputting top $N$ recommendations with confidence percentages, prerequisite checks, and pedagogical rationales.
- [x] **4. Create a Streamlit interface for students to input their completed courses and view recommendations.**  
      Built a multi-tab web application in `app/main.py` with 1-click academic presets, dynamic multiselect, confidence progress bars, and bookmarking to persistent storage.
- [x] **5. Analyze which course "itemsets" the model has learned.**  
      Applied bitset Apriori mining (`thilimarketbasket.mine`) to the model's generated sample distributions to discover frequent course itemsets and association rules with Support, Confidence, and Lift metrics.

---

## 🖥️ Streamlit Interface Features

The Streamlit application (`app/main.py`) provides an interactive interface:

1. **Academic Presets & Flexible Search:**
   - 1-click presets for common academic profiles: *Data Science*, *Web Dev*, *Cloud/DevOps*, *Cybersecurity*.
   - Searchable multi-select input for entering any custom combination of completed coursework.
2. **Interactive Recommendations:**
   - Displays ranked course recommendations (#1, #2, #3, ...) with course code, title, track badge, and level.
   - Interactive recommendation score progress bar displaying relative match strength.
   - Detailed pedagogical rationale explaining why each course is the logical next step.
3. **Market Basket Analytics Tab:**
   - Discovers recurring course itemsets learned by the language model.
   - Association rule table with Support, Confidence, and Lift metrics.
4. **Curriculum Bookmarking:**
   - Students can bookmark recommended courses, stored persistently via `Store` in `data/bookmarks.json`.

---

## 📐 Mathematical Formulation

### 1. Autoregressive Language Modeling Loss
For a course sequence $S = (c_1, c_2, \dots, c_T)$, the causal next-token cross-entropy loss is:

$$\mathcal{L}_{LM} = - \frac{1}{T} \sum_{t=1}^{T} \log P(c_t \mid c_{t-k}, \dots, c_{t-1})$$

Where $k$ is the context window size ($k=3$) and predictions are computed via learned token embeddings and hidden ReLU projections:

$$z_t = W_o \cdot \text{ReLU}(W_h \cdot \mathbf{e}_{context} + b_h) + b_o$$

$$P(c_t \mid context) = \text{softmax}(z_t)$$

### 2. Market Basket Co-Occurrence Lift
To reward courses that specifically co-occur with a student's prior courses rather than just generally popular courses, we compute normalized association lift:

$$\text{Lift}(c \mid \text{history}) = \frac{1}{|\text{history}|} \sum_{h \in \text{history}} \frac{P(c \mid h)}{P(c)} = \frac{1}{|\text{history}|} \sum_{h \in \text{history}} \frac{\text{Count}(c \cap h) \cdot N}{\text{Count}(h) \cdot \text{Count}(c)}$$

The final candidate recommendation score balances sequential order and co-occurrence:

$$P_{\text{blended}}(c) = (1 - \alpha) \cdot P_{LM}(c \mid context) + \alpha \cdot \text{NormLift}(c)$$

---

## 🚀 Quickstart & Installation

### Requirements
- Python 3.12 or higher
- Git

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/eMohan07/learning-path-predictor.git
cd learning-path-predictor

# 2. Create and activate a virtual environment
python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Optional environment file
cp .env.example .env
```

### Launch the Streamlit Interface

```bash
streamlit run app/main.py
```

Open your browser to `http://localhost:8501`.

---

## 🧪 Running Tests

The test suite thoroughly covers all milestones and plumbing:

```bash
pytest
```

Output:
```text
============================= test session starts =============================
collected 8 items

tests/test_core.py .....                                                 [ 62%]
tests/test_smoke.py ...                                                  [100%]

============================== 8 passed in 4.30s ==============================
```

---

## 📂 Project Structure

```
learning-path-predictor/
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI workflow (Python 3.12 test runner)
├── app/
│   ├── __init__.py
│   ├── core.py                # Core engine: CourseLanguageModel, sequence generation,
│   │                          # prediction scoring, and itemset mining (Milestones 1-5)
│   ├── llm.py                 # Multi-provider LLM client (Ollama/Groq/Gemini)
│   ├── main.py                # Streamlit user interface with tabs and presets
│   └── storage.py             # Atomic file-backed key-value store for bookmarks
├── tests/
│   ├── test_core.py           # Comprehensive tests for Milestones 1 through 5
│   └── test_smoke.py          # Initial smoke & environment plumbing tests
├── data/                      # Persistent storage directory
├── pytest.ini                 # Pytest configuration
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

---

## ⭐ Stretch Goals Achieved

- [x] **Confidence Scores:** Visualized normalized recommendation scores and raw LM probabilities via interactive progress bars.
- [x] **Course Bookmarking:** Enabled saving and viewing planned learning path bookmarks persistently in local storage.
- [x] **Learned Itemset Analysis:** Extracted frequent course bundles and association rules using bitset Apriori with support, confidence, and lift metrics.
