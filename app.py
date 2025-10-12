import pickle
import streamlit as st
import numpy as np
file_name = "minmax.pkl"  # Replace with your trained model filename
with open(file_name, 'rb') as f:
    scaler = pickle.load(f)
# Load the trained model
file_name = "best_grade_class_model.pkl"  # Replace with your trained model filename
with open(file_name, 'rb') as f:
    model = pickle.load(f)

# Title of the app
st.title("Student Grade Classification App")
 
# Input fields
age = st.number_input("Enter Age:", min_value=15, max_value=18)
gender = st.selectbox("Select Gender (0 for Male, 1 for Female):", [0, 1])
ethnicity = st.selectbox("Select Ethnicity (0: Caucasian, 1: African American, 2: Asian, 3: Other):", [0, 1, 2, 3])
parental_education = st.selectbox("Parental Education Level (0: None, 1: High School, 2: Some College, 3: Bachelor's, 4: Higher):", [0, 1, 2, 3, 4])
study_time_weekly = st.number_input("Weekly Study Time (hours):", min_value=0, max_value=20)
absences = st.number_input("Number of Absences:", min_value=0, max_value=30)
tutoring = st.selectbox("Tutoring Status (0: No, 1: Yes):", [0, 1])
parental_support = st.slider("Parental Support (0: None, 4: Very High):", 0, 4, step=1)
extracurricular = st.selectbox("Participation in Extracurricular Activities (0: No, 1: Yes):", [0, 1])
sports = st.selectbox("Participation in Sports (0: No, 1: Yes):", [0, 1])
music = st.selectbox("Participation in Music (0: No, 1: Yes):", [0, 1])
volunteering = st.selectbox("Participation in Volunteering (0: No, 1: Yes):", [0, 1])

# Academic Performance

# Predict Grade Class
if st.button("Predict Grade Class"):
   if (gender == 0 and ethnicity == 0 and parental_education == 0 and 
        study_time_weekly == 0 and absences == 0 and tutoring == 0 and 
        parental_support == 0 and extracurricular == 0 and 
        sports == 0 and music == 0 and volunteering == 0):
        
        st.error("Invalid input! Please provide meaningful values for prediction.")

   else:

    # Ensure input matches the model's training features
    input_features = np.array([[age, gender, ethnicity, parental_education, study_time_weekly,
                                 absences, tutoring, parental_support, extracurricular,
                                 sports, music, volunteering]])
    prediction = model.predict(input_features)

    # Map numerical prediction to grade label
    grade_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}
    predicted_grade = grade_mapping[prediction[0]]

    proba = model.predict_proba(input_features)[0]          # length 5
    labels = ['A', 'B', 'C', 'D', 'F']
    import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
    proba_df = pd.DataFrame({'Grade': labels, 'Probability': proba})

    fig, ax = plt.subplots()
    sns.barplot(x='Grade', y='Probability', data=proba_df, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title('Predicted Probability for Each Grade Class')
    ax.bar_label(ax.containers[0], fmt='%.2f')
    st.pyplot(fig)

    st.success(f"The predicted grade class is: {predicted_grade}")


