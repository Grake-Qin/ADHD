import streamlit as st
import pandas as pd
import joblib
import pickle
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
model_path = "best_random_forest_model.pkl"
best_svm_model = joblib.load(model_path)

scaler_path = "scaler.pkl"
scaler = joblib.load(scaler_path)

explainer_path = "explainer.pkl"
with open(explainer_path, 'rb') as f:
    explainer = pickle.load(f)

shap_values_path = "shap_values.pkl"
with open(shap_values_path, 'rb') as f:
    shap_values = pickle.load(f)
# 定义特征名称
feature_names = [
    "Gender", "Age", "Handedness", "ADHD Measure", "ADHD Index",
    "Inattentive", "Hyper/Impulsive", "IQ Measure", "Verbal IQ", "Performance IQ", "Full4 IQ"
]

prompts = {
    "Gender": "Gender (enter 1 for male, enter 0 for female)",
    "Age": "Age",
    "Handedness": "Handedness：enter 1 for right, enter 0 for left",
    "ADHD Measure": "ADHD Measure：enter 1 for ADHD Rating Scale IV (ADHD-RS), enter 2 for Conners’ Parent Rating Scale-Revised, Long version (CPRS-LV)，enter 2 for Connors’ Rating Scale-3rd Edition)",
    "ADHD Index": "ADHD Index",
    "Inattentive": "Inattentive",
    "Hyper/Impulsive": "Hyper/Impulsive",
    "IQ Measure": "IQ Measure：enter 1 for，enter 1 for Wechsler Intelligence Scale for Children, Fourth Edition (WISC-IV)，enter 2 for Wechsler Abbreviated Scale of Intelligence (WASI)，enter 3 for Wechsler Intelligence Scale for Chinese Children-Revised (WISCC-R)",
    "Verbal IQ": "Verbal IQ",
    "Performance IQ": "Performance IQ",
    "Full4 IQ": "Full4 IQ"
}

# Streamlit应用标题
st.title("Children's ADHD prediction model trained based on the ADHD-200 dataset")

# 创建输入框供用户输入特征值
user_input = []
for feature in feature_names:
    value = st.text_input(prompts[feature], value="", placeholder="Please enter a value")
    user_input.append(value)

# 检查所有输入是否为空
if st.button("Predict"):
    if "" in user_input:
        st.write("Please enter values for all features.")
    else:
        # 将用户输入转换为DataFrame
        input_df = pd.DataFrame([user_input], columns=feature_names)

        # 转换数据类型
        input_df = input_df.astype(float)

        # 标准化用户输入
        input_scaled = scaler.transform(input_df)

        # 进行预测
        probability = best_svm_model.predict_proba(input_scaled)[0, 1]
        st.write(
            f"Based on the above features, the probability that this child has ADHD is: **{probability * 100:.2f}%**")

        # 计算SHAP值
        shap_value_sample = explainer.shap_values(input_scaled)

        # 绘制SHAP决策图
        st.subheader("SHAP explainer")
        fig, ax = plt.subplots()
        shap.decision_plot(explainer.expected_value[1], shap_value_sample[1], input_df)
        plt.title('Decision Plot')
        st.pyplot(fig)
