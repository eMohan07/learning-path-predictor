"""Streamlit entry point: `streamlit run app/main.py`
Learning Path Predictor - AI Course Recommendation Engine
"""
from __future__ import annotations

import streamlit as st
import pandas as pd

from app.core import (
    COURSE_CATALOG,
    TRACK_COURSES,
    get_default_model,
    predict_next_courses,
)
from app.storage import Store

# Page configuration
st.set_page_config(
    page_title="Learning Path Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize persistence store
store = Store("data/bookmarks.json")

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "completed_courses" not in st.session_state:
    st.session_state["completed_courses"] = ["CS101", "PY101"]

if "bookmarks" not in st.session_state:
    st.session_state["bookmarks"] = store.get("bookmarks", [])

# -----------------------------------------------------------------------------
# Sidebar Configuration & Settings
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Recommendation Settings")
    st.markdown("Customize model prediction parameters.")

    top_n = st.slider("Number of Recommendations (Top N)", min_value=1, max_value=8, value=5)
    enforce_prereqs = st.checkbox("Enforce Course Prerequisites", value=True, help="Only recommend courses whose prerequisites you have completed.")
    alpha_lift = st.slider(
        "Association Lift Weight (α)",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="0.0 = Pure Sequential Autoregressive LM; 1.0 = Pure Market Basket Co-occurrence Lift.",
    )

    st.divider()
    st.subheader("📌 Quick Presets")
    st.caption("Click to load common student academic profiles:")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if st.button("📊 Data Science", use_container_width=True):
            st.session_state["completed_courses"] = ["CS101", "PY101", "MATH101", "DS201", "STAT201"]
            st.rerun()
        if col_p1.button("🌐 Web Dev", use_container_width=True):
            st.session_state["completed_courses"] = ["CS101", "WEB101", "DB201"]
            st.rerun()

    with col_p2:
        if st.button("☁️ Cloud/DevOps", use_container_width=True):
            st.session_state["completed_courses"] = ["CS101", "NET101", "OS201"]
            st.rerun()
        if col_p2.button("🔒 Cybersecurity", use_container_width=True):
            st.session_state["completed_courses"] = ["CS101", "NET101", "SEC201"]
            st.rerun()

    if st.button("🧹 Clear All", use_container_width=True):
        st.session_state["completed_courses"] = []
        st.rerun()

    st.divider()
    st.caption("Learning Path Predictor v1.0 • Powered by Causal LM & Market Basket Analysis")

# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.title("🎓 Learning Path Predictor")
st.markdown(
    "Personalized course recommendations for students based on academic history. "
    "Our model predicts your logical next learning steps using **autoregressive sequence modeling** "
    "and **market basket co-occurrence patterns**."
)

# Header Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Available Courses", len(COURSE_CATALOG))
m2.metric("Specialization Tracks", len(TRACK_COURSES))
m3.metric("Courses Completed", len(st.session_state["completed_courses"]))
m4.metric("Saved Bookmarks", len(st.session_state["bookmarks"]))

st.divider()

# -----------------------------------------------------------------------------
# Step 1: Input Completed Courses
# -----------------------------------------------------------------------------
st.subheader("1. Enter Your Completed Studies")

# Format options for multiselect
all_course_ids = sorted(COURSE_CATALOG.keys())
course_options = {
    cid: f"{cid}: {COURSE_CATALOG[cid]['title']} [{COURSE_CATALOG[cid]['track']}]"
    for cid in all_course_ids
}

selected = st.multiselect(
    "Select all courses you have finished so far (or choose a preset in the sidebar):",
    options=all_course_ids,
    default=[c for c in st.session_state["completed_courses"] if c in COURSE_CATALOG],
    format_func=lambda cid: course_options[cid],
    placeholder="Search by course code, name, or track...",
)

# Keep session state in sync
st.session_state["completed_courses"] = selected

# Show current trajectory tags
if selected:
    st.markdown("**Your Current Academic Path:**")
    path_cols = st.columns(min(len(selected), 6))
    for i, cid in enumerate(selected):
        col = path_cols[i % len(path_cols)]
        meta = COURSE_CATALOG.get(cid, {})
        col.info(f"**Step {i+1}**: {cid}\n\n*{meta.get('title', '')}*")
else:
    st.warning("👉 No completed courses selected yet. Select courses above or choose a preset in the sidebar to generate recommendations!")

st.divider()

# -----------------------------------------------------------------------------
# Step 2: Next Course Recommendations
# -----------------------------------------------------------------------------
st.subheader(f"2. Top {top_n} Recommended Next Steps")

if not selected:
    st.info("Select at least one introductory course (e.g., CS101, PY101, NET101, or WEB101) to view recommendations.")
else:
    with st.spinner("Analyzing sequence and calculating next-step probabilities..."):
        model = get_default_model()
        recommendations = predict_next_courses(
            completed_courses=selected,
            top_n=top_n,
            model=model,
            enforce_prerequisites=enforce_prereqs,
            alpha_lift=alpha_lift,
        )

    if not recommendations:
        st.warning("No candidate courses found matching prerequisite requirements. Try unchecking 'Enforce Course Prerequisites' in the sidebar.")
    else:
        st.success(f"Generated {len(recommendations)} optimal next-step course recommendation(s) for your academic profile!")

        for rank, rec in enumerate(recommendations, start=1):
            with st.container():
                c_head, c_score, c_action = st.columns([3, 1.5, 1])
                with c_head:
                    st.markdown(f"### #{rank} `{rec['course_id']}`: {rec['title']}")
                    st.caption(f"📁 Track: **{rec['track']}** | 🎚️ Level: **Level {rec['level']}**")
                    st.write(rec["description"])
                    st.markdown(f"💡 *Why this course*: {rec['rationale']}")

                with c_score:
                    st.markdown("**Recommendation Score**")
                    st.progress(min(1.0, rec["relative_score"] / 100.0))
                    st.write(f"Relative Match: **{rec['relative_score']:.1f}%**")
                    st.caption(f"Raw LM probability: {rec['confidence_pct']:.2f}%")

                with c_action:
                    st.write("")
                    st.write("")
                    b_key = f"bm_{rec['course_id']}"
                    is_bookmarked = any(b.get("course_id") == rec["course_id"] for b in st.session_state["bookmarks"])
                    if is_bookmarked:
                        st.button("Saved ⭐", key=b_key, disabled=True)
                    else:
                        if st.button("Bookmark 🔖", key=b_key):
                            new_bm = {
                                "course_id": rec["course_id"],
                                "title": rec["title"],
                                "track": rec["track"],
                                "score": rec["relative_score"],
                            }
                            st.session_state["bookmarks"].append(new_bm)
                            store.put("bookmarks", st.session_state["bookmarks"])
                            st.rerun()

                st.divider()

# -----------------------------------------------------------------------------
# Step 3: Bookmarks & Curriculum Plan
# -----------------------------------------------------------------------------
if st.session_state["bookmarks"]:
    st.subheader("3. 🔖 Your Saved Learning Path Bookmarks")
    bm_df = pd.DataFrame(st.session_state["bookmarks"])
    st.dataframe(bm_df, use_container_width=True)

    if st.button("Clear All Bookmarks"):
        st.session_state["bookmarks"] = []
        store.put("bookmarks", [])
        st.rerun()
