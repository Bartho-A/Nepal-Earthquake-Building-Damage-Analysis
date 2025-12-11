import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

from category_encoders import OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier


# ----------------------------
# Load data
# ----------------------------

@st.cache_data
def load_data():
    df = pd.read_pickle(
        "nepal_buildings_clean.pkl",
        compression="gzip", 
    )
    X = df.drop(columns="severe_damage")
    y = df["severe_damage"]
    return df, X, y



@st.cache_resource
def train_models(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_lr = make_pipeline(
        OneHotEncoder(use_cat_names=True),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    model_lr.fit(X_train, y_train)

    final_model_dt = make_pipeline(
        OrdinalEncoder(),
        DecisionTreeClassifier(max_depth=14, random_state=42),
    )
    final_model_dt.fit(X_train, y_train)

    return X_train, X_val, y_train, y_val, model_lr, final_model_dt


# ----------------------------
# Layout
# ----------------------------

st.set_page_config(
    page_title="Nepal Earthquake Damage",
    page_icon="🌏",
    layout="wide",
)

st.title("🌏 Nepal Earthquake Building Damage Analysis")

df, X, y = load_data()
X_train, X_val, y_train, y_val, model_lr, final_model_dt = train_models(X, y)

tab_eda, tab_model = st.tabs(["EDA", "Model"])


# ----------------------------
# EDA tab
# ----------------------------

with tab_eda:
    st.subheader("Class Balance – Severe Damage")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(4, 3))
        y.value_counts(normalize=True).plot(
            kind="bar",
            ax=ax,
        )
        ax.set_xlabel("Severe Damage")
        ax.set_ylabel("Relative Frequency")
        ax.set_title("Class Balance (All Data)")
        st.pyplot(fig)

        st.write("Value counts:")
        st.write(y.value_counts())

    with col2:
        st.markdown(
            "- Class 0: buildings with damage grade ≤ 3\n"
            "- Class 1: buildings with damage grade > 3"
        )

    st.markdown("---")
    st.subheader("Plinth Area vs Severe Damage")

    if "plinth_area_sq_ft" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x="severe_damage", y="plinth_area_sq_ft", data=df, ax=ax)
        ax.set_xlabel("Severe Damage")
        ax.set_ylabel("Plinth Area [sq. ft.]")
        ax.set_title("Plinth Area vs Building Damage")
        st.pyplot(fig)
    else:
        st.info("Column 'plinth_area_sq_ft' not found in the dataset.")

    st.markdown("---")
    st.subheader("Severe Damage Rate by Roof Type")

    if "roof_type" in df.columns:
        roof_pivot = pd.pivot_table(
            df, index="roof_type", values="severe_damage", aggfunc="mean"
        ).sort_values("severe_damage")

        st.write("Table: severe damage rate by roof type")
        st.dataframe(roof_pivot)

        fig, ax = plt.subplots(figsize=(6, 4))
        roof_pivot["severe_damage"].plot(kind="barh", ax=ax)
        ax.set_xlabel("Severe Damage Rate")
        ax.set_ylabel("Roof Type")
        ax.set_title("Severe Damage Rate by Roof Type")
        st.pyplot(fig)
    else:
        st.info("Column 'roof_type' not found in the dataset.")


# ----------------------------
# Model tab
# ----------------------------

with tab_model:
    st.subheader("Model Performance – Logistic Regression vs Decision Tree")

    # Baseline
    acc_baseline = y_train.value_counts(normalize=True).max()

    # Metrics function
    def metrics_for(model, X_tr, y_tr, X_te, y_te, name):
        y_tr_pred = model.predict(X_tr)
        y_te_pred = model.predict(X_te)
        return {
            "Model": name,
            "Train Accuracy": accuracy_score(y_tr, y_tr_pred),
            "Val Accuracy": accuracy_score(y_te, y_te_pred),
            "Val Precision (Severe)": precision_score(y_te, y_te_pred, pos_label=1),
            "Val Recall (Severe)": recall_score(y_te, y_te_pred, pos_label=1),
            "Val F1 (Severe)": f1_score(y_te, y_te_pred, pos_label=1),
        }

    rows = []
    rows.append(metrics_for(model_lr, X_train, y_train, X_val, y_val, "Logistic Regression"))
    rows.append(
        metrics_for(
            final_model_dt,
            X_train,
            y_train,
            X_val,
            y_val,
            "Decision Tree (max_depth=14)",
        )
    )
    metrics_df = pd.DataFrame(rows)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Baseline Accuracy", f"{acc_baseline:.2f}")
    with col2:
        st.write("")

    st.write("Validation metrics by model:")
    # Identify numeric columns
    num_cols = metrics_df.select_dtypes(include=["float", "int"]).columns

    st.dataframe(
    metrics_df.style.format(
        {col: "{:.2f}" for col in num_cols}
    )
    )


    st.markdown("---")
    st.subheader("Confusion Matrices (Validation)")

    col1, col2 = st.columns(2)

    with col1:
        st.caption("Logistic Regression")
        y_val_pred_lr = model_lr.predict(X_val)
        cm_lr = confusion_matrix(y_val, y_val_pred_lr)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm_lr, display_labels=["No severe", "Severe"]).plot(ax=ax)
        ax.set_title("Logistic Regression – Confusion Matrix")
        st.pyplot(fig)

    with col2:
        st.caption("Decision Tree (max_depth=14)")
        y_val_pred_dt = final_model_dt.predict(X_val)
        cm_dt = confusion_matrix(y_val, y_val_pred_dt)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm_dt, display_labels=["No severe", "Severe"]).plot(ax=ax)
        ax.set_title("Decision Tree – Confusion Matrix")
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Decision Tree Feature Importance")

    # Gini feature importance (using training columns)
    features = X_train.columns
    importances = final_model_dt.named_steps["decisiontreeclassifier"].feature_importances_
    feat_imp = pd.Series(importances, index=features)

    feat_imp_sorted = feat_imp.sort_values(ascending=False)
    feat_imp_top = feat_imp_sorted.head(15)

    st.write("Top 15 features driving severe damage:")
    st.dataframe(
        feat_imp_top.reset_index().rename(
            columns={"index": "Feature", 0: "Gini Importance"}
        )
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    feat_imp_top.sort_values().plot(kind="barh", ax=ax)
    ax.set_xlabel("Gini Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Top 15 Features – Decision Tree")
    plt.tight_layout()
    st.pyplot(fig)


st.markdown(
    "<p style='text-align:center; font-size:12px;'>"
    "Built with Streamlit & scikit-learn – Nepal Earthquake Building Damage</p>",
    unsafe_allow_html=True,
)
