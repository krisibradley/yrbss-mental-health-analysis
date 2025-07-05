# 🔬 Methodology: YRBSS Mental Health Analysis

## Overview
This document provides a comprehensive methodology for analyzing the 2023 Youth Risk Behavior Surveillance System (YRBSS) data to predict persistent sadness and hopelessness among high school students.

## 📊 Data Source & Collection

### Survey Design
- **Survey Name:** Youth Risk Behavior Surveillance System (YRBSS) 2023
- **Administered by:** Centers for Disease Control and Prevention (CDC)
- **Survey Period:** Spring 2023 (typically March-May)
- **Target Population:** Students in grades 9-12 in public and private schools

### Sampling Methodology
- **Design:** Three-stage cluster sampling design
- **Stage 1:** Primary sampling units (counties/groups of counties)
- **Stage 2:** Schools within selected areas
- **Stage 3:** Classes within selected schools
- **Weighting:** Applied to adjust for nonresponse and oversampling

### Sample Characteristics
- **Total Sample:** 20,103 students
- **Valid Q26 Responses:** 19,863 students (98.8% response rate)
- **Geographic Coverage:** Nationally representative
- **School Types:** Public and private schools included

## 🎯 Research Question & Objectives

### Primary Research Question
What factors predict persistent sadness and hopelessness among U.S. high school students?

### Specific Objectives
1. **Identify** the prevalence of persistent sadness/hopelessness (Q26)
2. **Determine** the strongest predictors from 100+ behavioral variables
3. **Develop** machine learning models for early identification
4. **Analyze** risk and protective factors separately
5. **Evaluate** demographic variations in mental health outcomes
6. **Create** actionable insights for intervention programs

## 🔍 Variable Selection & Data Processing

### Target Variable
- **Q26:** "During the past 12 months, did you ever feel so sad or hopeless almost every day for two weeks or more in a row that you stopped doing some usual activities?"
- **Coding:** 1 = Yes, 2 = No
- **Recoding for Analysis:** 1 = Yes (persistent sadness), 0 = No (resilient)
- **Clinical Significance:** Aligns with DSM-5 criteria for major depressive episodes

### Predictor Variables

#### Inclusion Criteria
- **Original survey questions only** (Q1-Q107, demographic variables)
- **Exclude derived variables** (QN*, qn* prefixes) to prevent correlation inflation
- **Exclude structural variables** (stratum, psu, weight) used for survey design
- **Minimum sample size:** 500+ valid responses per variable

#### Variable Categories Analyzed
1. **Demographics:** Age, sex, grade, race/ethnicity (5 variables)
2. **Mental Health:** Suicide ideation, attempts, plans (4 variables)
3. **Substance Use:** Alcohol, tobacco, drugs, prescription misuse (15 variables)
4. **Violence & Safety:** Fighting, bullying, dating violence (10 variables)
5. **Sexual Behavior:** Activity, contraception, contacts (8 variables)
6. **Physical Health:** Weight, exercise, sleep, nutrition (12 variables)
7. **Social Environment:** School, family, community factors (8 variables)
8. **Risk Behaviors:** Driving, weapon carrying, risky activities (6 variables)

#### Data Quality Filters
- **Missing Data Threshold:** Variables with >50% missing excluded
- **Correlation Threshold:** |r| > 0.05 for meaningful relationships
- **Sample Size Requirement:** Minimum 500 valid pairs for correlation calculation

## 🧮 Statistical Analysis Methods

### Descriptive Statistics
- **Frequency distributions** for all categorical variables
- **Cross-tabulations** between Q26 and key predictors
- **Prevalence calculations** with 95% confidence intervals
- **Demographic breakdowns** by subgroups

### Correlation Analysis
- **Method:** Pearson product-moment correlation
- **Interpretation:** 
  - |r| > 0.5: Very strong relationship
  - |r| 0.3-0.5: Strong relationship
  - |r| 0.2-0.3: Moderate relationship
  - |r| 0.1-0.2: Weak relationship
  - |r| < 0.1: Very weak/no relationship

### Variable Naming Convention
- **Original codes** (Q84) mapped to **meaningful names** (Current_mental_health)
- **Systematic naming** for interpretability
- **Consistent formatting** across all analyses

## 🤖 Machine Learning Pipeline

### Data Preprocessing

#### Missing Data Handling
```python
# Strategy: Median imputation for numeric variables
for col in X.columns:
    if X[col].isnull().any():
        X[col].fillna(X[col].median(), inplace=True)
```

#### Feature Selection
- **Top 15 predictors** based on correlation strength
- **Gender (Q2)** included due to known mental health differences
- **Domain knowledge** applied to ensure relevant variables

#### Class Imbalance Handling
- **Problem:** 40.8% positive class (persistent sadness) vs 59.2% negative class
- **Solution:** SMOTE (Synthetic Minority Oversampling Technique)
- **Rationale:** Prevents model bias toward majority class

### Model Selection & Training

#### Algorithms Evaluated
1. **Logistic Regression**
   - Baseline linear model
   - Interpretable coefficients
   - Handles binary classification naturally

2. **Random Forest**
   - Ensemble method with feature importance
   - Handles non-linear relationships
   - Robust to outliers

3. **Gradient Boosting**
   - Sequential learning algorithm
   - High predictive accuracy
   - Handles complex interactions

#### Model Configuration
```python
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}
```

### Validation Strategy

#### Train/Test Split
- **Split ratio:** 80% training, 20% testing
- **Stratification:** Maintains class distribution in both sets
- **Random state:** Fixed for reproducibility

#### Cross-Validation
- **Method:** 5-fold stratified cross-validation
- **Metric:** AUC-ROC (Area Under Receiver Operating Characteristic)
- **Rationale:** AUC is robust to class imbalance

#### Performance Metrics
- **Primary:** AUC-ROC (discrimination ability)
- **Secondary:** Precision, recall, F1-score
- **Interpretation:** AUC > 0.8 considered excellent

## 📊 Statistical Interpretation

### Correlation Interpretation
- **Positive correlation with Q26:** Protective factor (higher values = less sadness)
- **Negative correlation with Q26:** Risk factor (higher values = more sadness)
- **Note:** Q26 coded as 1=Yes(sad), 2=No(not sad), so interpretation is reversed

### Feature Importance
- **Random Forest:** Gini importance scores
- **Interpretation:** Higher values = more predictive power
- **Ranking:** Top 15 features displayed with names

### Model Comparison
- **ROC Curves:** Visual comparison of model performance
- **AUC Values:** Quantitative discrimination ability
- **Cross-validation:** Estimates of generalization performance

## 🔄 Reproducibility & Quality Assurance

### Code Standards
- **Version Control:** Git repository with clear commit messages
- **Documentation:** Comprehensive docstrings and comments
- **Error Handling:** Graceful handling of missing data and edge cases
- **Modularity:** Functions for each analysis step

### Reproducibility Measures
- **Random Seeds:** Fixed for all stochastic processes
- **Environment:** requirements.txt with specific versions
- **Data Provenance:** Clear documentation of data source
- **Analysis Pipeline:** Step-by-step methodology documentation

### Quality Checks
- **Data Validation:** Checking for impossible values
- **Statistical Assumptions:** Verifying correlation assumptions
- **Model Diagnostics:** Checking for overfitting
- **Results Verification:** Manual spot-checks of calculations

## ⚠️ Limitations & Considerations

### Data Limitations
- **Cross-sectional design:** Cannot establish causation
- **Self-reported data:** Potential for response bias
- **Survey timing:** Single snapshot in time
- **Missing data:** Some variables have substantial missingness

### Methodological Limitations
- **Correlation vs. Causation:** Associations do not imply causation
- **Model Complexity:** May not capture all relevant interactions
- **Generalizability:** Results specific to 2023 survey year
- **Clinical Validation:** Models not validated against clinical diagnoses

### Ethical Considerations
- **Privacy:** Individual-level data not stored or shared
- **Stigma:** Results should inform support, not labeling
- **Intervention:** Predictions should guide help, not punishment
- **Professional Support:** Mental health requires qualified professionals

## 📚 Statistical Software & Tools

### Primary Tools
- **Python 3.8+:** Main programming language
- **pandas:** Data manipulation and analysis
- **scikit-learn:** Machine learning algorithms
- **matplotlib/seaborn:** Data visualization
- **numpy:** Numerical computing

### Analysis Environment
- **Jupyter Notebook:** Interactive development
- **Git:** Version control
- **GitHub:** Code sharing and collaboration

## 🎯 Future Methodological Enhancements

### Potential Improvements
1. **Longitudinal Analysis:** Multiple survey years
2. **Causal Inference:** Instrumental variables, propensity matching
3. **Deep Learning:** Neural networks for complex patterns
4. **Geographic Analysis:** State and regional variations
5. **Intervention Studies:** Randomized controlled trials

### Advanced Techniques
- **Survival Analysis:** Time to mental health crisis
- **Multilevel Modeling:** School and district effects
- **Bayesian Methods:** Uncertainty quantification
- **Ensemble Methods:** Combining multiple model types

## 📖 References & Standards

### Methodological Standards
- **STROBE Guidelines:** Strengthening the Reporting of Observational Studies
- **FAIR Principles:** Findable, Accessible, Interoperable, Reusable data
- **APA Style:** Statistical reporting standards

### Statistical References
- **Cohen, J. (1988):** Statistical Power Analysis for the Behavioral Sciences
- **Hosmer & Lemeshow (2000):** Applied Logistic Regression
- **Hastie et al. (2009):** The Elements of Statistical Learning

---

*This methodology follows best practices for reproducible research in public health and data science.*