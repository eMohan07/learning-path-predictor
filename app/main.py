"""Streamlit entry point: `streamlit run app/main.py`
Learning Path Predictor - AI Course Recommendation Engine
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.core import (
    COURSE_CATALOG,
    TRACK_COURSES,
    analyze_learned_itemsets,
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
        value=0.35,
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
        if st.button("🌐 Web Dev", use_container_width=True):
            st.session_state["completed_courses"] = ["CS101", "WEB101", "DB201"]
            st.rerun()

    with col_p2:
        if st.button("☁️ Cloud/DevOps", use_container_width=True):
            st.session_state["completed_courses"] = ["CS101", "NET101", "OS201"]
            st.rerun()
        if st.button("🔒 Cybersecurity", use_container_width=True):
            st.session_state["completed_courses"] = ["CS101", "NET101", "SEC201"]
            st.rerun()

    if st.button("🧹 Clear All", use_container_width=True):
        st.session_state["completed_courses"] = []
        st.rerun()

    st.divider()
    st.caption("Learning Path Predictor • Powered by Causal LM & Market Basket Analysis")

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
# Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🎯 Course Recommendations",
    "📊 Learned Itemsets & Market Basket Analysis",
    "🔖 Saved Bookmarks",
])

with tab1:
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

with tab2:
    st.subheader("📊 Learned Course Itemsets & Association Rules")
    st.markdown(
        "By applying **Market Basket Analysis (bitset Apriori)** to course sequences generated by the language model, "
        "we discover recurring course combinations ('itemsets') and strong association rules that the model learned."
    )

    with st.spinner("Mining frequent itemsets and association rules from language model representations..."):
        model = get_default_model()
        analysis = analyze_learned_itemsets(model=model, n_samples=400, min_support=0.03, min_confidence=0.30)

    col_stat1, col_stat2, col_stat3 = st.columns(3)
    col_stat1.metric("Sampled Paths Analyzed", analysis["n_samples"])
    col_stat2.metric("Frequent Course Itemsets", analysis["total_itemsets_found"])
    col_stat3.metric("Association Rules Discovered", analysis["total_rules_found"])

    st.markdown("### 🔥 Frequent Course Itemsets")
    st.caption("Courses that consistently co-occur within student learning trajectories:")

    if analysis["frequent_itemsets"]:
        itemset_rows = [
            {
                "Course Bundle": " + ".join(item["items"]),
                "Bundle Size": item["size"],
                "Dominant Track": item["track"],
                "Support (%)": f"{item['support_pct']}%",
            }
            for item in analysis["frequent_itemsets"][:15]
        ]
        st.dataframe(pd.DataFrame(itemset_rows), use_container_width=True)

    st.markdown("### 🔗 Curricular Association Rules")
    st.caption("Rules of the form *'Students who complete Antecedent also complete Consequent'*:")

    if analysis["association_rules"]:
        rule_rows = [
            {
                "If Student Completed (Antecedent)": rule["antecedent_str"],
                "Likely Also Completes (Consequent)": rule["consequent_str"],
                "Support": f"{rule['support']:.3f}",
                "Confidence": f"{rule['confidence'] * 100:.1f}%",
                "Lift Score": f"{rule['lift']:.2f}x",
            }
            for rule in analysis["association_rules"][:15]
        ]
        st.dataframe(pd.DataFrame(rule_rows), use_container_width=True)
        st.info("💡 **Lift > 1.0** indicates that taking the antecedent course significantly increases the likelihood of enrolling in the consequent course compared to baseline enrollment rates.")

with tab3:
    st.subheader("🔖 Your Saved Learning Path Bookmarks")
    if st.session_state["bookmarks"]:
        bm_df = pd.DataFrame(st.session_state["bookmarks"])
        st.dataframe(bm_df, use_container_width=True)

        if st.button("Clear All Bookmarks", key="clear_bm_tab"):
            st.session_state["bookmarks"] = []
            store.put("bookmarks", [])
            st.rerun()
    else:
        st.info("No courses bookmarked yet! Click 'Bookmark 🔖' next to any recommendation in Tab 1 to save it.")
