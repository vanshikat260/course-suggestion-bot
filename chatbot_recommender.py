import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np
import os

data = [
    [100, 150, 'Frontend', 'Frontend', 1],
    [100, 250, 'AI', 'Frontend', 0],
    [300, 200, 'Backend', 'Backend', 1],
    [150, 170, 'AI', 'AI', 1],
    [50, 200, 'Backend', 'Frontend', 0],
    [450, 250, 'AI', 'AI', 1],
    [250, 180, 'Backend', 'AI', 0],
    [200, 150, 'Frontend', 'AI', 0],
    [400, 150, 'Frontend', 'Frontend', 1],
    [50, 120, 'Frontend', 'Backend', 0],
    [300, 250, 'AI', 'AI', 1],
    [180, 180, 'Backend', 'Frontend', 0],
    [350, 200, 'Backend', 'Backend', 1],
    [220, 160, 'Frontend', 'Frontend', 1],
    [160, 150, 'AI', 'Backend', 0],
    [500, 250, 'AI', 'AI', 1],
    [100, 100, 'Frontend', 'AI', 0],
    [330, 200, 'Backend', 'AI', 1],
    [270, 250, 'AI', 'AI', 1],
    [90, 120, 'Frontend', 'Frontend', 1]
]

df = pd.DataFrame(data, columns=["user_xp", "course_xp", "course_cat", "user_pref_cat", "interested"])

le = LabelEncoder()
df["course_cat_enc"] = le.fit_transform(df["course_cat"])
df["user_pref_cat_enc"] = le.transform(df["user_pref_cat"])


X = df[["user_xp", "course_xp", "course_cat_enc", "user_pref_cat_enc"]]
y = df["interested"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)


joblib.dump(model, "recommender_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("Model and encoder trained and saved!")

course_catalog = [
    {"title": "React Basics", "category": "Frontend", "xp": 150},
    {"title": "Advanced CSS", "category": "Frontend", "xp": 120},
    {"title": "Node.js for Beginners", "category": "Backend", "xp": 180},
    {"title": "REST APIs with Express", "category": "Backend", "xp": 200},
    {"title": "Intro to Machine Learning", "category": "AI", "xp": 170},
    {"title": "Deep Learning with Python", "category": "AI", "xp": 250},
    {"title": "AI for Beginners", "category": "AI", "xp": 160},
    {"title": "Build a Portfolio Website", "category": "Frontend", "xp": 130},
    {"title": "Database Design Fundamentals", "category": "Backend", "xp": 190},
    {"title": "Computer Vision Projects", "category": "AI", "xp": 240},
]

def predict_interest(user_xp, course_xp, course_cat, user_pref_cat):
    model = joblib.load("recommender_model.pkl")
    le = joblib.load("label_encoder.pkl")

    try:
        course_cat_encoded = le.transform([course_cat])[0]
        user_pref_cat_encoded = le.transform([user_pref_cat])[0]
        features = pd.DataFrame([{
        "user_xp": user_xp,
        "course_xp": course_xp,
        "course_cat_enc": course_cat_encoded,
        "user_pref_cat_enc": user_pref_cat_encoded
}])
        prediction = model.predict(features)[0]
        return prediction
    except Exception as e:
        print(f"Prediction error: {e}")
        return 0


def recommend_courses(completed_titles, user_pref_cat):
    total_xp = sum(course["xp"] for course in course_catalog if course["title"] in completed_titles)
    recommended = []

    for course in course_catalog:
        if course["title"] not in completed_titles:
            interested = predict_interest(total_xp, course["xp"], course["category"], user_pref_cat)
            if interested == 1:
                recommended.append(course["title"])
    return recommended


if __name__ == "__main__":
    print("\n Welcome to the AI Course Recommender Chatbot!\n")
    name = input(" What is your name? ")
    interest = input("💡 What category are you interested in (AI / Frontend / Backend)? ").strip()

    print("\nCourses you've completed (type titles from the list or press enter to skip):")
    for course in course_catalog:
        print(" -", course["title"])
    
    completed_input = input("\n Enter completed course titles separated by commas: ").split(",")
    completed_titles = [title.strip() for title in completed_input if title.strip()]

    print(f"\n Recommending courses for {name} based on your interest in '{interest}'...\n")
    recommended = recommend_courses(completed_titles, interest)

    if recommended:
        print("Recommended courses for you:")
        for course in recommended:
            print(" -", course)
    else:
        print("No strong recommendations found. Try completing more courses or changing your interest.")
