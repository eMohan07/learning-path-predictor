"""Core implementation for Learning Path Predictor.
One stub per milestone; built in order.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Course Catalog & Curricula Definition
# -----------------------------------------------------------------------------

COURSE_CATALOG: Dict[str, Dict[str, Any]] = {
    # Computer Science Fundamentals
    "CS101": {
        "title": "Introduction to Computer Science",
        "track": "Computer Science",
        "level": 1,
        "prerequisites": [],
        "description": "Foundational computing concepts, algorithms, and computational problem solving.",
    },
    "CS102": {
        "title": "Data Structures & Algorithms",
        "track": "Computer Science",
        "level": 2,
        "prerequisites": ["CS101"],
        "description": "Arrays, trees, graphs, sorting, and algorithmic complexity analysis.",
    },
    "CS103": {
        "title": "Computer Systems Architecture",
        "track": "Computer Science",
        "level": 2,
        "prerequisites": ["CS101"],
        "description": "Hardware-software interface, memory hierarchy, and processor architecture.",
    },

    # Data Science & AI Track
    "PY101": {
        "title": "Python Programming Fundamentals",
        "track": "Data Science & AI",
        "level": 1,
        "prerequisites": [],
        "description": "Core syntax, data structures, modular programming, and Python standard libraries.",
    },
    "MATH101": {
        "title": "Linear Algebra & Discrete Math",
        "track": "Data Science & AI",
        "level": 1,
        "prerequisites": [],
        "description": "Vectors, matrices, eigenvalues, combinatorics, and discrete mathematical structures.",
    },
    "DS201": {
        "title": "Data Analysis & Visualization",
        "track": "Data Science & AI",
        "level": 2,
        "prerequisites": ["PY101"],
        "description": "Tabular data manipulation with Pandas, exploratory data analysis, and visualization.",
    },
    "STAT201": {
        "title": "Applied Probability & Statistics",
        "track": "Data Science & AI",
        "level": 2,
        "prerequisites": ["MATH101"],
        "description": "Hypothesis testing, probability distributions, regression, and Bayesian inference.",
    },
    "ML301": {
        "title": "Machine Learning Foundations",
        "track": "Data Science & AI",
        "level": 3,
        "prerequisites": ["DS201", "STAT201"],
        "description": "Supervised and unsupervised learning, decision trees, ensemble methods, and evaluation.",
    },
    "DL401": {
        "title": "Deep Learning & Neural Networks",
        "track": "Data Science & AI",
        "level": 4,
        "prerequisites": ["ML301"],
        "description": "Feedforward networks, CNNs, RNNs, backpropagation, and PyTorch/TensorFlow.",
    },
    "NLP402": {
        "title": "Natural Language Processing",
        "track": "Data Science & AI",
        "level": 4,
        "prerequisites": ["DL401"],
        "description": "Word embeddings, sequence models, transformer architectures, and attention mechanisms.",
    },
    "CV403": {
        "title": "Computer Vision",
        "track": "Data Science & AI",
        "level": 4,
        "prerequisites": ["DL401"],
        "description": "Image classification, object detection, segmentation, and convolutional neural nets.",
    },
    "MLOPS501": {
        "title": "MLOps & Production AI",
        "track": "Data Science & AI",
        "level": 4,
        "prerequisites": ["ML301"],
        "description": "Model deployment, experiment tracking, pipeline orchestration, and monitoring.",
    },

    # Web & Software Engineering Track
    "WEB101": {
        "title": "Web Development Fundamentals",
        "track": "Software Engineering",
        "level": 1,
        "prerequisites": [],
        "description": "HTML5, CSS3, modern JavaScript, DOM manipulation, and responsive web design.",
    },
    "DB201": {
        "title": "Database Systems & SQL",
        "track": "Software Engineering",
        "level": 2,
        "prerequisites": ["CS101"],
        "description": "Relational schema design, SQL querying, indexing, transactions, and normalization.",
    },
    "BE202": {
        "title": "Backend Engineering & REST APIs",
        "track": "Software Engineering",
        "level": 2,
        "prerequisites": ["CS101", "DB201"],
        "description": "Server-side architecture, RESTful API design, authentication, and ORM integration.",
    },
    "FE203": {
        "title": "Modern Frontend Frameworks",
        "track": "Software Engineering",
        "level": 3,
        "prerequisites": ["WEB101"],
        "description": "Component-based architecture, state management, SPA routing, and React/Vue.",
    },
    "SYS301": {
        "title": "System Design & Architecture",
        "track": "Software Engineering",
        "level": 3,
        "prerequisites": ["BE202", "CS102"],
        "description": "Distributed systems, scalability, caching, load balancing, and microservices.",
    },
    "TEST302": {
        "title": "Software Testing & QA",
        "track": "Software Engineering",
        "level": 3,
        "prerequisites": ["BE202"],
        "description": "Unit testing, integration testing, CI testing pipelines, and TDD methodologies.",
    },

    # Cloud & DevOps Track
    "NET101": {
        "title": "Computer Networking Fundamentals",
        "track": "Cloud & Infrastructure",
        "level": 1,
        "prerequisites": [],
        "description": "TCP/IP stack, DNS, routing protocols, HTTP/HTTPS, and network diagnostics.",
    },
    "OS201": {
        "title": "Operating Systems & Linux",
        "track": "Cloud & Infrastructure",
        "level": 2,
        "prerequisites": ["CS101"],
        "description": "Process management, virtual memory, concurrency, shell scripting, and Linux administration.",
    },
    "CLOUD201": {
        "title": "Cloud Computing Foundations",
        "track": "Cloud & Infrastructure",
        "level": 2,
        "prerequisites": ["NET101", "OS201"],
        "description": "Cloud primitives, compute instances, object storage, IAM, and VPC networking.",
    },
    "CONT301": {
        "title": "Containers & Kubernetes",
        "track": "Cloud & Infrastructure",
        "level": 3,
        "prerequisites": ["CLOUD201", "OS201"],
        "description": "Docker containerization, container runtimes, Kubernetes pods, services, and deployments.",
    },
    "IAC302": {
        "title": "Infrastructure as Code & CI/CD",
        "track": "Cloud & Infrastructure",
        "level": 3,
        "prerequisites": ["CLOUD201"],
        "description": "Terraform, Ansible, declarative configuration, and automated delivery pipelines.",
    },
    "SRE401": {
        "title": "Site Reliability Engineering",
        "track": "Cloud & Infrastructure",
        "level": 4,
        "prerequisites": ["CONT301", "SYS301"],
        "description": "Service Level Objectives (SLOs), observability, distributed tracing, and resilience.",
    },

    # Cybersecurity Track
    "SEC201": {
        "title": "Cybersecurity Fundamentals",
        "track": "Cybersecurity",
        "level": 2,
        "prerequisites": ["NET101"],
        "description": "Security principles, cryptography, symmetric/asymmetric encryption, and threat modeling.",
    },
    "SEC301": {
        "title": "Network Security & Defense",
        "track": "Cybersecurity",
        "level": 3,
        "prerequisites": ["SEC201", "NET101"],
        "description": "Firewalls, IDS/IPS, VPNs, packet inspection, and secure network design.",
    },
    "SEC401": {
        "title": "Ethical Hacking & Penetration Testing",
        "track": "Cybersecurity",
        "level": 4,
        "prerequisites": ["SEC301", "OS201"],
        "description": "Vulnerability scanning, exploitation methodologies, privilege escalation, and red teaming.",
    },
    "SEC402": {
        "title": "Cloud & Application Security",
        "track": "Cybersecurity",
        "level": 4,
        "prerequisites": ["SEC201", "CLOUD201"],
        "description": "OWASP Top 10, DevSecOps, cloud posture management, and zero trust architecture.",
    },
}

TRACK_COURSES = {
    "Data Science & AI": ["CS101", "PY101", "MATH101", "DS201", "STAT201", "ML301", "DL401", "NLP402", "CV403", "MLOPS501"],
    "Software Engineering": ["CS101", "CS102", "WEB101", "DB201", "BE202", "FE203", "SYS301", "TEST302"],
    "Cloud & Infrastructure": ["CS101", "NET101", "OS201", "CLOUD201", "CONT301", "IAC302", "SRE401"],
    "Cybersecurity": ["CS101", "NET101", "OS201", "SEC201", "SEC301", "SEC401", "SEC402"],
}


def get_course_catalog() -> Dict[str, Dict[str, Any]]:
    """Return the global course catalog dictionary."""
    return COURSE_CATALOG


# --- Milestone 1: Generate mock student course completion data as sequences of course IDs.
# Hint: Use Python to programmatically create lists of sequential course IDs, simulating student enrollment histories.

def generate_student_sequences(
    n_students: int = 1200,
    min_length: int = 3,
    max_length: int = 8,
    seed: int = 42,
) -> List[List[str]]:
    """Programmatically generate lists of sequential course IDs simulating student enrollment histories.

    Each student follows a primary track with high affinity, may take relevant electives
    from complementary tracks, and respects foundational prerequisite flows.
    """
    rng = random.Random(seed)
    tracks = list(TRACK_COURSES.keys())
    track_weights = [0.35, 0.30, 0.20, 0.15]

    sequences: List[List[str]] = []

    for _ in range(n_students):
        primary_track = rng.choices(tracks, weights=track_weights, k=1)[0]
        target_length = rng.randint(min_length, max_length)

        completed: List[str] = []
        available_pool = set(COURSE_CATALOG.keys())

        # Select introductory course
        intro_options = [c for c in TRACK_COURSES[primary_track] if COURSE_CATALOG[c]["level"] == 1]
        first_course = rng.choice(intro_options)
        completed.append(first_course)
        available_pool.discard(first_course)

        # Progressively choose subsequent courses
        while len(completed) < target_length and available_pool:
            candidates: List[Tuple[str, float]] = []

            for cid in available_pool:
                meta = COURSE_CATALOG[cid]
                # Check prerequisites: are all prerequisites already completed?
                prereqs = meta["prerequisites"]
                prereqs_met = all(p in completed for p in prereqs)
                if not prereqs_met:
                    continue

                weight = 1.0

                # Strong weight boost for primary track
                if cid in TRACK_COURSES[primary_track]:
                    weight += 5.0

                # Moderate weight boost for natural sequential progression (level + 1)
                last_level = COURSE_CATALOG[completed[-1]]["level"]
                if meta["level"] == last_level + 1:
                    weight += 3.0
                elif meta["level"] == last_level:
                    weight += 1.5

                # Cross-track synergy boosts
                if primary_track == "Data Science & AI" and cid in {"DB201", "CLOUD201"}:
                    weight += 2.0
                elif primary_track == "Software Engineering" and cid in {"CLOUD201", "NET101", "DS201"}:
                    weight += 2.0
                elif primary_track == "Cybersecurity" and cid in {"CLOUD201", "SYS301"}:
                    weight += 2.0
                elif primary_track == "Cloud & Infrastructure" and cid in {"SYS301", "SEC201", "DB201"}:
                    weight += 2.0

                candidates.append((cid, weight))

            if not candidates:
                break

            total_weight = sum(w for _, w in candidates)
            probs = [w / total_weight for _, w in candidates]
            chosen = rng.choices([c for c, _ in candidates], weights=probs, k=1)[0]

            completed.append(chosen)
            available_pool.discard(chosen)

        sequences.append(completed)

    return sequences


def milestone_1(
    n_students: int = 1200,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate mock student course completion data as sequences of course IDs.

    Returns:
        dict containing 'sequences', 'catalog', and dataset 'summary'.
    """
    sequences = generate_student_sequences(n_students=n_students, seed=seed)
    seq_lengths = [len(s) for s in sequences]
    unique_courses = sorted({cid for s in sequences for cid in s})

    summary = {
        "n_students": len(sequences),
        "total_enrollments": sum(seq_lengths),
        "mean_sequence_length": float(np.mean(seq_lengths)),
        "min_sequence_length": min(seq_lengths),
        "max_sequence_length": max(seq_lengths),
        "unique_courses": len(unique_courses),
        "catalog_size": len(COURSE_CATALOG),
    }

    return {
        "sequences": sequences,
        "catalog": COURSE_CATALOG,
        "summary": summary,
    }


# --- Milestone 2: Pretrain a small language model on these course ID sequences.
# Hint: Apply the `lfretail-lfmarketbasket` techniques to fine-tune a small language model on your generated course ID sequences.
def milestone_2():  # TODO(milestone 2)
    raise NotImplementedError("TODO(milestone 2)")


# --- Milestone 3: Implement a function to predict the next likely course given a partial sequence.
# Hint: Implement a function that uses your fine-tuned language model to predict the next likely course ID given a partial sequence of completed courses.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Create a Streamlit interface for students to input their completed courses and view recommendations.
# Hint: Build a Streamlit application with input fields for students to enter their completed courses and a section to display the model's recommendations.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Analyze which course "itemsets" the model has learned.
# Hint: Inspect the model's learned representations or attention patterns to identify frequently co-occurring course 'itemsets,' similar to analyzing market basket data.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
