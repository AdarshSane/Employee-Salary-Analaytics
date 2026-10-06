import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Salary Intelligence", page_icon=":material/analytics:", layout="wide")

df = pd.read_csv("data/Salary Data.csv")
df = df.dropna(how="all").drop_duplicates().dropna()
df = df[df["Salary"] >= 10000]
model = joblib.load("salary_prediction_pipeline.pkl")

if "predicted" not in st.session_state:
    st.session_state.predicted = False
if "prediction" not in st.session_state:
    st.session_state.prediction = None

st.title(":material/payments: Salary Intelligence Dashboard")
st.write("An interactive machine learning application for salary prediction, benchmarking, and salary trend analysis.")

tab1, tab2, tab3 = st.tabs([":material/analytics: Salary Prediction", "📊 Salary Analytics", ":material/psychology: Model Insights"])

with tab1:
    st.header("Employee Information")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 70, 30, 1)
        gender = st.selectbox("Gender", sorted(df["Gender"].unique()))
        education = st.selectbox("Education Level", sorted(df["Education Level"].unique()))
    with col2:
        experience = st.slider("Years of Experience", 0.0, 50.0, 5.0, 0.5)
        job_title = st.selectbox("Job Title", sorted(df["Job Title"].unique()))

    if st.button("💰 Predict Salary", use_container_width=True):
        if experience > age - 18:
            st.error("Years of experience cannot exceed the person's likely working age.")
        else:
            input_data = pd.DataFrame({"Age":[age],"Gender":[gender],"Education Level":[education],"Job Title":[job_title],"Years of Experience":[experience]})
            prediction = model.predict(input_data)[0]
            st.session_state.predicted = True
            st.session_state.prediction = prediction
            st.session_state.prediction_age = age
            st.session_state.prediction_gender = gender
            st.session_state.prediction_education = education
            st.session_state.prediction_job = job_title
            st.session_state.prediction_experience = experience

    if st.session_state.predicted:
        prediction = st.session_state.prediction
        saved_job = st.session_state.prediction_job
        saved_education = st.session_state.prediction_education
        st.success(f"Predicted Salary: ${prediction:,.2f}")
        st.subheader("📊 Salary Benchmark")
        overall_median = df["Salary"].median()
        job_average = df[df["Job Title"] == saved_job]["Salary"].mean()
        education_average = df[df["Education Level"] == saved_education]["Salary"].mean()
        percentile = (df["Salary"] < prediction).mean() * 100
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Predicted Salary", f"${prediction:,.0f}")
        with col2: st.metric("Dataset Median", f"${overall_median:,.0f}")
        with col3: st.metric("Job Average", f"${job_average:,.0f}")
        with col4: st.metric("Estimated Percentile", f"{percentile:.0f}th")
                # Compare prediction with dataset median
        difference_percent = (
            (prediction - overall_median) / overall_median
        ) * 100

        if prediction >= overall_median:
            st.info(
                f"📈 The predicted salary is "
                f"{difference_percent:.1f}% above the dataset median."
            )
        else:
            st.info(
                f"📉 The predicted salary is "
                f"{abs(difference_percent):.1f}% below the dataset median."
            )

        st.subheader("🔮 What-If Analysis")
        st.write("See how the predicted salary changes when the employee's years of experience changes.")
        what_if_experience = st.slider("What if the employee had this much experience?", 0.0, 30.0, float(st.session_state.prediction_experience), 0.5)
        what_if_data = pd.DataFrame({"Age":[st.session_state.prediction_age],"Gender":[st.session_state.prediction_gender],"Education Level":[st.session_state.prediction_education],"Job Title":[st.session_state.prediction_job],"Years of Experience":[what_if_experience]})
        what_if_prediction = model.predict(what_if_data)[0]
        difference = what_if_prediction - prediction
        st.metric("What-If Predicted Salary", f"${what_if_prediction:,.0f}", f"${difference:,.0f} compared with original")
        experience_range = np.arange(0, 31, 1)
        what_if_df = pd.DataFrame({"Age":[st.session_state.prediction_age]*len(experience_range),"Gender":[st.session_state.prediction_gender]*len(experience_range),"Education Level":[st.session_state.prediction_education]*len(experience_range),"Job Title":[st.session_state.prediction_job]*len(experience_range),"Years of Experience":experience_range})
        salary_predictions = model.predict(what_if_df)
        chart_df = pd.DataFrame({"Years of Experience":experience_range,"Predicted Salary":salary_predictions})
        fig = px.line(chart_df, x="Years of Experience", y="Predicted Salary", markers=True, title="Predicted Salary vs Years of Experience")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("📊 Salary Analytics")
    st.subheader("Salary Distribution")
    fig = px.histogram(df, x="Salary", nbins=30, title="Distribution of Salaries")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Experience vs Salary")
    fig = px.scatter(df, x="Years of Experience", y="Salary", title="Salary vs Years of Experience", hover_data=["Job Title", "Education Level"])
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Average Salary by Education Level")
    education_salary = df.groupby("Education Level")["Salary"].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(education_salary, x="Education Level", y="Salary", title="Average Salary by Education Level")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("💼 Top Job Titles by Average Salary")
    job_salary = df.groupby("Job Title")["Salary"].mean().sort_values(ascending=False).head(15).sort_values().reset_index()
    fig = px.bar(job_salary, x="Salary", y="Job Title", orientation="h", title="Top 15 Job Titles by Average Salary")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("🧠 Model Insights")
    grouped_importance = pd.read_csv("grouped_feature_importance.csv", index_col=0)
    grouped_importance.columns = ["Importance"]
    st.subheader("What Drives Salary Predictions?")
    importance_for_chart = grouped_importance.sort_values("Importance")
    fig = px.bar(importance_for_chart, x="Importance", y=importance_for_chart.index, orientation="h", title="Feature Importance")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Feature importance shows the relative contribution of each original input feature to the Random Forest model's predictions.")
