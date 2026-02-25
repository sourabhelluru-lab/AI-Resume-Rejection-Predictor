import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="AI Resume Builder", layout="centered")

st.title("📄 AI Resume Selection Predictor")
st.write("Paste your Resume and Job Description below to analyze your selection probability.")

# -------------------------------
# Input Fields
# -------------------------------
resume_text = st.text_area("Paste Resume Here")
job_description = st.text_area("Paste Job Description Here")

if st.button("Analyze"):

    if resume_text.strip() == "" or job_description.strip() == "":
        st.error("Please paste both Resume and Job Description.")
    else:

        # -------------------------------
        # TF-IDF Similarity
        # -------------------------------
        documents = [resume_text, job_description]
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(documents)

        similarity_score = cosine_similarity(
            tfidf_matrix[0:1], tfidf_matrix[1:2]
        )
        match_percentage = similarity_score[0][0] * 100

        # -------------------------------
        # Keyword Analysis
        # -------------------------------
        feature_names = vectorizer.get_feature_names_out()

        resume_vector = tfidf_matrix[0].toarray()[0]
        job_vector = tfidf_matrix[1].toarray()[0]

        matched_keywords = []
        missing_keywords = []

        for i in range(len(feature_names)):
            if job_vector[i] > 0 and resume_vector[i] > 0:
                matched_keywords.append(feature_names[i])
            elif job_vector[i] > 0 and resume_vector[i] == 0:
                missing_keywords.append(feature_names[i])

        # -------------------------------
        # Demo ML Model Training
        # -------------------------------
        data = {
            "resume": [
                "Python SQL Machine Learning Data Analysis",
                "Java Spring Boot Backend Development",
                "HTML CSS React Frontend Developer",
                "Python Flask Machine Learning AWS",
                "C++ Embedded Systems Microcontrollers",
                "Graphic Designer Photoshop Illustrator",
                "Marketing Sales Communication Skills",
                "Civil Engineering Construction AutoCAD"
            ],
            "job_description": [
                "Looking for Python Machine Learning Engineer",
                "Backend Developer with Java experience",
                "Frontend Developer with React skills",
                "Cloud Engineer with AWS and Python",
                "Embedded Engineer with C++ knowledge",
                "Python Data Analyst with SQL skills",
                "Software Developer with Machine Learning",
                "Backend Developer with Flask experience"
            ],
            "status": [1, 1, 1, 1, 1, 0, 0, 0]
        }

        df = pd.DataFrame(data)
        df["combined"] = df["resume"] + " " + df["job_description"]

        ml_vectorizer = TfidfVectorizer(stop_words='english')
        X = ml_vectorizer.fit_transform(df["combined"])
        y = df["status"]

        model = LogisticRegression()
        model.fit(X, y)

        # -------------------------------
        # Predict User Input
        # -------------------------------
        new_input = resume_text + " " + job_description
        new_vector = ml_vectorizer.transform([new_input])

        probability = model.predict_proba(new_vector)

        selection_probability = probability[0][1] * 100
        rejection_probability = probability[0][0] * 100

        # -------------------------------
        # Display Results
        # -------------------------------
        st.subheader("📊 Analysis Results")

        st.metric("Resume Match Score", f"{match_percentage:.2f}%")
        st.metric("Selection Probability", f"{selection_probability:.2f}%")
        st.metric("Rejection Probability", f"{rejection_probability:.2f}%")

        if selection_probability > 70:
            st.success("✅ Strong Profile Match")
        elif selection_probability > 40:
            st.warning("⚠ Moderate Match - Improve Resume")
        else:
            st.error("❌ High Rejection Risk")

        # -------------------------------
        # Keyword Display
        # -------------------------------
        st.subheader("🔎 Keyword Analysis")

        st.write("### Matched Keywords")
        st.write(matched_keywords if matched_keywords else "None")

        st.write("### Missing Keywords")
        st.write(missing_keywords if missing_keywords else "None")