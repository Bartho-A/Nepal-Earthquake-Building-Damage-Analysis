# Nepal Earthquake Building Damage Analysis

Interactive Streamlit dashboard analyzing building vulnerability to severe damage from the 2015 Gorkha earthquake in Nepal.

## 📊 **Dashboard Features**

### **EDA Tab**
- **Class balance**: Distribution of severe vs non-severe damage (binary target from damage_grade > 3)
- **Plinth area analysis**: Boxplot showing building size vs damage severity
- **Roof type vulnerability**: Pivot table + bar chart of severe damage rates by roof_type

### **Model Tab** 
- **Model comparison**: Logistic Regression vs Decision Tree (max_depth=14)
  - Train/validation accuracy, precision/recall/F1 for severe class
  - Confusion matrices for both models
- **Feature importance**: Top 15 predictors from decision tree (Gini importance)
  - Highlights structural factors driving severe damage risk

### **Key Findings**
- Decision Tree (depth=14): 82% train | 82% val accuracy
- Logistic Regression: Balanced class weighting
- Top features: Roof type, building height, building age, plinth area
  
## **Tech Stack**
- Streamlit (interactive dashboard)
- scikit-learn (LogisticRegression, DecisionTreeClassifier)
- category_encoders (OneHotEncoder, OrdinalEncoder)
- pandas, matplotlib, seaborn (EDA & viz)
- Data: Nepal Earthquake dataset

## **Quick Start**
**Clone & install**
git clone https://github.com/Bartho_A/Nepal-Earthquake-Building-Damage-Analysis.git cd Nepal-Earthquake-Building-Damage-Analysis pip install -r requirements.txt

## **Project Structure**
- streamlit_app.py
- nepal_buildings_clean.pkl
- requirements.txt
- README.md

## **License**
MIT © 2025 Bartho Aobe
Credits: WQU
