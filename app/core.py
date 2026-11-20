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


# -----------------------------------------------------------------------------
# Milestone 2: Pretrain a Small Language Model on Course Sequences
# -----------------------------------------------------------------------------

class CourseLanguageModel:
    """A lightweight autoregressive language model for sequential course path prediction.

    Pretrained on student course ID sequences using next-token cross-entropy loss.
    Incorporates token embeddings, a context projection layer, softmax output distribution,
    and Item-to-Item co-occurrence matrices inspired by market basket analysis.
    """

    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"

    def __init__(
        self,
        catalog: Optional[Dict[str, Dict[str, Any]]] = None,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        context_window: int = 3,
        seed: int = 42,
    ):
        self.catalog = catalog or COURSE_CATALOG
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.context_window = context_window
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Build vocabulary
        self.special_tokens = [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        self.course_tokens = sorted(self.catalog.keys())
        self.vocab = self.special_tokens + self.course_tokens
        self.token_to_id = {tok: idx for idx, tok in enumerate(self.vocab)}
        self.id_to_token = {idx: tok for idx, tok in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

        self.pad_id = self.token_to_id[self.PAD_TOKEN]
        self.bos_id = self.token_to_id[self.BOS_TOKEN]
        self.eos_id = self.token_to_id[self.EOS_TOKEN]

        # Weights initialization
        self._init_weights()

        # Co-occurrence & Item-To-Item tracking (Market Basket techniques)
        self.co_occurrence = np.zeros((self.vocab_size, self.vocab_size), dtype=np.float64)
        self.item_counts = np.zeros(self.vocab_size, dtype=np.float64)
        self.total_baskets = 0

    def _init_weights(self) -> None:
        """Initialize neural network weights using He/Xavier normal distribution."""
        bound_e = np.sqrt(1.0 / self.embedding_dim)
        bound_h = np.sqrt(2.0 / (self.context_window * self.embedding_dim))
        bound_o = np.sqrt(2.0 / self.hidden_dim)

        self.W_e = self.rng.normal(0.0, bound_e, (self.vocab_size, self.embedding_dim))
        self.W_h = self.rng.normal(0.0, bound_h, (self.context_window * self.embedding_dim, self.hidden_dim))
        self.b_h = np.zeros(self.hidden_dim, dtype=np.float64)
        self.W_o = self.rng.normal(0.0, bound_o, (self.hidden_dim, self.vocab_size))
        self.b_o = np.zeros(self.vocab_size, dtype=np.float64)

    def _prepare_training_data(self, sequences: List[List[str]]) -> Tuple[np.ndarray, np.ndarray]:
        """Convert sequences of course IDs into (context_window, target) integer arrays."""
        contexts: List[List[int]] = []
        targets: List[int] = []

        for seq in sequences:
            token_ids = [self.token_to_id[c] for c in seq if c in self.token_to_id]
            if not token_ids:
                continue

            full_seq = [self.bos_id] + token_ids + [self.eos_id]

            for i in range(1, len(full_seq)):
                target = full_seq[i]
                history = full_seq[max(0, i - self.context_window):i]
                # Pad to the left if history is shorter than context_window
                padded = [self.pad_id] * (self.context_window - len(history)) + history
                contexts.append(padded)
                targets.append(target)

        return np.array(contexts, dtype=np.int32), np.array(targets, dtype=np.int32)

    def _update_co_occurrences(self, sequences: List[List[str]]) -> None:
        """Update pairwise co-occurrence frequencies across course histories."""
        self.total_baskets += len(sequences)
        for seq in sequences:
            ids = [self.token_to_id[c] for c in seq if c in self.token_to_id]
            unique_ids = list(set(ids))
            for i in unique_ids:
                self.item_counts[i] += 1
                for j in unique_ids:
                    self.co_occurrence[i, j] += 1

    def forward(self, contexts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass through the embedding and hidden layers to logits and softmax probabilities."""
        batch_size = contexts.shape[0]
        # Embeddings lookup: (batch, context_window, embedding_dim) -> (batch, context_window * embedding_dim)
        emb = self.W_e[contexts].reshape(batch_size, self.context_window * self.embedding_dim)
        # Hidden layer with ReLU
        z_h = emb @ self.W_h + self.b_h
        a_h = np.maximum(0.0, z_h)
        # Output logits
        logits = a_h @ self.W_o + self.b_o
        # Stable softmax
        logits_max = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return emb, a_h, probs

    def fit(
        self,
        sequences: List[List[str]],
        epochs: int = 15,
        batch_size: int = 64,
        lr: float = 0.01,
        log_callback: Optional[Any] = None,
    ) -> List[Dict[str, float]]:
        """Pretrain the language model on course sequences using Adam optimization.

        Returns a list of dicts with training history per epoch.
        """
        self._update_co_occurrences(sequences)
        contexts, targets = self._prepare_training_data(sequences)
        n_samples = len(contexts)

        if n_samples == 0:
            return []

        # Adam optimizer state
        m_We, v_We = np.zeros_like(self.W_e), np.zeros_like(self.W_e)
        m_Wh, v_Wh = np.zeros_like(self.W_h), np.zeros_like(self.W_h)
        m_bh, v_bh = np.zeros_like(self.b_h), np.zeros_like(self.b_h)
        m_Wo, v_Wo = np.zeros_like(self.W_o), np.zeros_like(self.W_o)
        m_bo, v_bo = np.zeros_like(self.b_o), np.zeros_like(self.b_o)

        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0
        history: List[Dict[str, float]] = []

        for epoch in range(epochs):
            indices = np.arange(n_samples)
            self.rng.shuffle(indices)

            epoch_losses: List[float] = []

            for start in range(0, n_samples, batch_size):
                t += 1
                end = min(start + batch_size, n_samples)
                b_idx = indices[start:end]
                b_ctx = contexts[b_idx]
                b_tar = targets[b_idx]
                curr_b = len(b_idx)

                # Forward pass
                emb, a_h, probs = self.forward(b_ctx)

                # Cross-entropy loss
                target_probs = probs[np.arange(curr_b), b_tar]
                loss = -np.mean(np.log(np.maximum(target_probs, 1e-12)))
                epoch_losses.append(float(loss))

                # Backward pass
                d_logits = probs.copy()
                d_logits[np.arange(curr_b), b_tar] -= 1.0
                d_logits /= curr_b

                d_Wo = a_h.T @ d_logits
                d_bo = np.sum(d_logits, axis=0)

                d_ah = d_logits @ self.W_o.T
                d_zh = d_ah * (a_h > 0.0)

                d_Wh = emb.T @ d_zh
                d_bh = np.sum(d_zh, axis=0)

                d_emb = (d_zh @ self.W_h.T).reshape(curr_b, self.context_window, self.embedding_dim)

                # Gradient update helper with Adam
                def adam_update(param, grad, m, v):
                    m[:] = beta1 * m + (1.0 - beta1) * grad
                    v[:] = beta2 * v + (1.0 - beta2) * (grad ** 2)
                    m_hat = m / (1.0 - beta1 ** t)
                    v_hat = v / (1.0 - beta2 ** t)
                    param -= lr * m_hat / (np.sqrt(v_hat) + eps)

                adam_update(self.W_o, d_Wo, m_Wo, v_Wo)
                adam_update(self.b_o, d_bo, m_bo, v_bo)
                adam_update(self.W_h, d_Wh, m_Wh, v_Wh)
                adam_update(self.b_h, d_bh, m_bh, v_bh)

                # Accumulate and update embedding gradients
                d_We = np.zeros_like(self.W_e)
                for pos in range(self.context_window):
                    np.add.at(d_We, b_ctx[:, pos], d_emb[:, pos, :])
                adam_update(self.W_e, d_We, m_We, v_We)

            mean_loss = float(np.mean(epoch_losses))
            rec = {"epoch": epoch + 1, "loss": mean_loss, "perplexity": float(np.exp(min(mean_loss, 20.0)))}
            history.append(rec)
            if log_callback:
                log_callback(rec)

        return history

    def compute_perplexity(self, sequences: List[List[str]]) -> float:
        """Compute the validation perplexity on unseen course sequences."""
        contexts, targets = self._prepare_training_data(sequences)
        if len(contexts) == 0:
            return 1.0

        _, _, probs = self.forward(contexts)
        target_probs = probs[np.arange(len(targets)), targets]
        cross_entropy = -np.mean(np.log(np.maximum(target_probs, 1e-12)))
        return float(np.exp(min(cross_entropy, 20.0)))

    def predict_next_probabilities(
        self,
        completed_courses: List[str],
        alpha_lift: float = 0.3,
    ) -> Dict[str, float]:
        """Predict probability distribution over next courses given completed history.

        Blends autoregressive sequence likelihood with market basket lift adjustment.
        """
        # Tokenize completed history
        valid_ids = [self.token_to_id[c] for c in completed_courses if c in self.token_to_id]
        if not valid_ids:
            history = [self.bos_id]
        else:
            history = valid_ids

        # Construct padded context window
        ctx = history[max(0, len(history) - self.context_window):]
        padded_ctx = [self.pad_id] * (self.context_window - len(ctx)) + ctx
        ctx_array = np.array([padded_ctx], dtype=np.int32)

        # Forward pass to obtain raw LM probability
        _, _, probs = self.forward(ctx_array)
        raw_probs = probs[0].copy()

        # Item-to-item lift boost (from completed courses)
        lift_scores = np.zeros(self.vocab_size)
        if valid_ids and self.total_baskets > 0:
            for past_id in valid_ids:
                co_counts = self.co_occurrence[past_id]
                past_count = max(self.item_counts[past_id], 1.0)
                # P(candidate | past)
                cond_prob = co_counts / past_count
                lift_scores += cond_prob
            lift_scores /= len(valid_ids)

        # Blend autoregressive LM probability with co-occurrence lift
        blended = (1.0 - alpha_lift) * raw_probs + alpha_lift * lift_scores

        # Filter out special tokens and courses already taken
        taken_set = set(completed_courses)
        course_probs: Dict[str, float] = {}

        for cid in self.course_tokens:
            if cid in taken_set:
                continue
            idx = self.token_to_id[cid]
            course_probs[cid] = float(blended[idx])

        # Normalize remaining probabilities
        total = sum(course_probs.values())
        if total > 0:
            course_probs = {cid: p / total for cid, p in course_probs.items()}
        else:
            uniform = 1.0 / max(1, len(course_probs))
            course_probs = {cid: uniform for cid in course_probs}

        return course_probs


def milestone_2(
    sequences: Optional[List[List[str]]] = None,
    epochs: int = 15,
    seed: int = 42,
) -> Dict[str, Any]:
    """Pretrain a small language model on student course ID sequences.

    Returns:
        dict containing 'model', 'history', 'initial_loss', 'final_loss', and 'perplexity'.
    """
    if sequences is None:
        sequences = generate_student_sequences(n_students=800, seed=seed)

    split = int(0.8 * len(sequences))
    train_seqs = sequences[:split]
    val_seqs = sequences[split:]

    model = CourseLanguageModel(catalog=COURSE_CATALOG, embedding_dim=32, hidden_dim=64, context_window=3, seed=seed)
    history = model.fit(train_seqs, epochs=epochs, lr=0.01)
    val_perplexity = model.compute_perplexity(val_seqs)

    return {
        "model": model,
        "history": history,
        "initial_loss": history[0]["loss"] if history else 0.0,
        "final_loss": history[-1]["loss"] if history else 0.0,
        "val_perplexity": val_perplexity,
        "vocab_size": model.vocab_size,
    }


# -----------------------------------------------------------------------------
# Milestone 3: Next Course Prediction for Partial Sequences
# -----------------------------------------------------------------------------

_DEFAULT_MODEL: Optional[CourseLanguageModel] = None


def get_default_model(force_retrain: bool = False) -> CourseLanguageModel:
    """Retrieve or initialize the default pretrained CourseLanguageModel."""
    global _DEFAULT_MODEL
    if _DEFAULT_MODEL is None or force_retrain:
        sequences = generate_student_sequences(n_students=1000, seed=42)
        model = CourseLanguageModel(
            catalog=COURSE_CATALOG,
            embedding_dim=32,
            hidden_dim=64,
            context_window=3,
            seed=42,
        )
        model.fit(sequences, epochs=15, lr=0.01)
        _DEFAULT_MODEL = model
    return _DEFAULT_MODEL


def predict_next_courses(
    completed_courses: List[str],
    top_n: int = 5,
    model: Optional[CourseLanguageModel] = None,
    enforce_prerequisites: bool = True,
    alpha_lift: float = 0.3,
) -> List[Dict[str, Any]]:
    """Predict top N next likely courses given a student's partial sequence of completed courses.

    Args:
        completed_courses: List of course IDs the student has finished.
        top_n: Number of recommendations to return.
        model: Pretrained CourseLanguageModel (defaults to cached global model).
        enforce_prerequisites: Only recommend courses whose prerequisites are satisfied.
        alpha_lift: Market basket lift weight vs sequential LM prediction.

    Returns:
        List of course recommendations with IDs, titles, tracks, probabilities, and rationales.
    """
    if model is None:
        model = get_default_model()

    probs = model.predict_next_probabilities(completed_courses, alpha_lift=alpha_lift)

    candidates: List[Dict[str, Any]] = []
    completed_set = set(completed_courses)

    for cid, prob in probs.items():
        if cid in completed_set:
            continue

        meta = COURSE_CATALOG.get(cid, {})
        prereqs = meta.get("prerequisites", [])
        prereqs_met = all(p in completed_set for p in prereqs)

        if enforce_prerequisites and not prereqs_met:
            continue

        track = meta.get("track", "General")
        level = meta.get("level", 1)

        if prereqs_met and prereqs:
            prereq_str = ", ".join(prereqs)
            rationale = f"Builds directly upon completed prerequisite(s): {prereq_str} in {track}."
        elif not prereqs:
            rationale = f"Foundational level {level} course introducing key concepts in {track}."
        else:
            rationale = f"Recommended step in the {track} specialization track."

        candidates.append({
            "course_id": cid,
            "title": meta.get("title", cid),
            "track": track,
            "level": level,
            "probability": float(prob),
            "confidence_pct": round(float(prob) * 100.0, 1),
            "prerequisites": prereqs,
            "prerequisites_met": prereqs_met,
            "description": meta.get("description", ""),
            "rationale": rationale,
        })

    candidates.sort(key=lambda x: x["probability"], reverse=True)
    top_candidates = candidates[:top_n]

    total_top_p = sum(c["probability"] for c in top_candidates)
    if total_top_p > 0:
        for c in top_candidates:
            c["relative_score"] = round((c["probability"] / total_top_p) * 100.0, 1)
    else:
        for c in top_candidates:
            c["relative_score"] = round(100.0 / max(1, len(top_candidates)), 1)

    return top_candidates


def milestone_3(
    test_sequences: Optional[List[List[str]]] = None,
    model: Optional[CourseLanguageModel] = None,
) -> Dict[str, Any]:
    """Implement a function to predict the next likely course given a partial sequence.

    Evaluates predictions across canonical student tracks and returns recommendations.
    """
    if model is None:
        model = get_default_model()

    if test_sequences is None:
        test_sequences = [
            ["CS101", "PY101"],                              # Data Science start
            ["CS101", "PY101", "MATH101", "DS201", "STAT201"],# Full prep -> ML301
            ["CS101", "WEB101"],                             # Web dev track
            ["CS101", "NET101"],                             # Networking / Infrastructure
        ]

    evaluations: List[Dict[str, Any]] = []
    for seq in test_sequences:
        recs = predict_next_courses(seq, top_n=5, model=model)
        evaluations.append({
            "input_sequence": seq,
            "recommendations": recs,
            "top_course_id": recs[0]["course_id"] if recs else None,
            "top_course_title": recs[0]["title"] if recs else None,
        })

    return {
        "model_vocab_size": model.vocab_size,
        "evaluations": evaluations,
    }


# --- Milestone 4: Create a Streamlit interface for students to input their completed courses and view recommendations.
# Hint: Build a Streamlit application with input fields for students to enter their completed courses and a section to display the model's recommendations.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Analyze which course "itemsets" the model has learned.
# Hint: Inspect the model's learned representations or attention patterns to identify frequently co-occurring course 'itemsets,' similar to analyzing market basket data.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
