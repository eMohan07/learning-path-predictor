# Learning Path Predictor

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/dysr7gyyvb/badge.svg)](https://ask.thili.ai/b/dysr7gyyvb)

This project, 'Learning Path Predictor,' is an EdTech solution designed for students. It leverages language model techniques to recommend personalized next courses or study materials based on a student's completed academic history, optimizing their learning journey.

## What you are building

**Goal.** You will build a Learning Path Predictor that recommends the next most relevant courses to students based on their completed studies.

**Done when.** The project is complete when the Streamlit interface successfully displays top N relevant course recommendations for a user's input of completed courses, proving the model's ability to suggest logical next steps.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Generate mock student course completion data as sequences of course IDs.**  
      Hint: Use Python to programmatically create lists of sequential course IDs, simulating student enrollment histories.
- [x] **2. Pretrain a small language model on these course ID sequences.**  
      Hint: Apply the `lfretail-lfmarketbasket` techniques to fine-tune a small language model on your generated course ID sequences.
- [x] **3. Implement a function to predict the next likely course given a partial sequence.**  
      Hint: Implement a function that uses your fine-tuned language model to predict the next likely course ID given a partial sequence of completed courses.
- [x] **4. Create a Streamlit interface for students to input their completed courses and view recommendations.**  
      Hint: Build a Streamlit application with input fields for students to enter their completed courses and a section to display the model's recommendations.
- [x] **5. Analyze which course "itemsets" the model has learned.**  
      Hint: Inspect the model's learned representations or attention patterns to identify frequently co-occurring course 'itemsets,' similar to analyzing market basket data.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
streamlit run app/main.py
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Display the confidence score or probability alongside each recommended course.
- Allow students to 'bookmark' or save recommended learning paths within the Streamlit application.
- Visualize the strong associations or 'itemsets' between courses that the model has learned, perhaps as a network graph.
