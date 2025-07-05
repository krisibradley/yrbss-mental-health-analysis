# 📊 Data Dictionary: YRBSS 2023 Variables

## Overview
This document provides detailed explanations of all variables used in the YRBSS mental health analysis, including question text, response options, and coding schemes.

## 🎯 Target Variable

| Variable Code | Variable Name | Question Text | Response Options | Notes |
|---------------|---------------|---------------|------------------|-------|
| **Q26** | **Persistent_sadness_hopelessness** | During the past 12 months, did you ever feel so sad or hopeless almost every day for two weeks or more in a row that you stopped doing some usual activities? | 1=Yes, 2=No | **Primary outcome variable**. Aligns with DSM-5 criteria for major depressive episodes |

## 👥 Demographic Variables

| Variable Code | Variable Name | Question Text | Response Options | Notes |
|---------------|---------------|---------------|------------------|-------|
| **Q1** | **Age** | How old are you? | 1=12 years or younger, 2=13, 3=14, 4=15, 5=16, 6=17, 7=18 or older | Age categories |
| **Q2** | **Sex** | What is your sex? | 1=Female, 2=Male | Biological sex |
| **Q3** | **Grade** | In what grade are you? | 1=9th, 2=10th, 3=11th, 4=12th, 5=Ungraded/other | School grade level |
| **Q4** | **Hispanic_Latino** | Are you Hispanic or Latino? | 1=Yes, 2=No | Ethnicity identifier |
| **Q5** | **Race** | What is your race? (Select one or more) | A=American Indian/Alaska Native, B=Asian, C=Black/African American, D=Native Hawaiian/Pacific Islander, E=White | Multiple selection allowed |
| **raceeth** | **Race_ethnicity_combined** | Combined race/ethnicity variable | 1=American Indian/Alaska Native, 2=Asian, 3=Black/African American, 4=Native Hawaiian/Pacific Islander, 5=White, 6=Hispanic/Latino, 7=Multiple-Hispanic, 8=Multiple-Non-Hispanic | CDC-created composite variable |

## 🧠 Mental Health Variables

| Variable Code | Variable Name | Question Text | Response Options | Notes |
|---------------|---------------|---------------|------------------|-------|
| **Q27** | **Considered_suicide** | During the past 12 months, did you ever seriously consider attempting suicide? | 1=Yes, 2=No | Suicide ideation |
| **Q28** | **Made_suicide_plan** | During the past 12 months, did you make a plan about how you would attempt suicide? | 1=Yes, 2=No | Suicide planning |
| **Q29** | **Attempted_suicide** | During the past 12 months, how many times did you actually attempt suicide? | 1=0 times, 2=1 time, 3=2-3 times, 4=4-5 times, 5=6+ times | Suicide attempts |
| **Q30** | **Injurious_suicide_attempt** | If you attempted suicide during the past 12 months, did any attempt result in injury, poisoning, or overdose that had to be treated by a doctor or nurse? | 1=Did not attempt, 2=Yes, 3=No | Medical treatment for suicide attempt |
| **Q84** | **Current_mental_health** | During the past 30 days, how often was your mental health not good? | 1=Never, 2=Rarely, 3=Sometimes, 4=Most of the time, 5=Always | Current mental health status |
| **Q106** | **Concentration_difficulty** | Because of a physical, mental, or emotional problem, do you have serious difficulty concentrating, remembering, or making decisions? | 1=Yes, 2=No | Cognitive difficulties |

## 🏫 School & Social Environment

| Variable Code | Variable Name | Question Text | Response Options | Notes |
|---------------|---------------|---------------|------------------|-------|
| **Q23** | **Treated_badly_race_ethnicity** | During your life, how often have you felt that you were treated badly or unfairly in school because of your race or ethnicity? | 1=Never, 2=Rarely, 3=Sometimes, 4=Most of the time, 5=Always | Racial discrimination |
| **Q24** | **Bullying_at_school** | During the past 12 months, have you ever been bullied on school property? | 1=Yes, 2=No | School bullying |
| **Q25** | **Electronic_bullying** | During the past 12 months, have you ever been electronically bullied? | 1=Yes, 2=No | Cyberbullying |
| **Q87** | **Academic_grades** | During the past 12 months, how would you describe your grades in school? | 1=Mostly A's, 2=Mostly B's, 3=Mostly C's, 4=Mostly D's, 5=Mostly F's, 6=None of these, 7=Not sure | Academic performance |
| **Q103** | **School_connectedness** | Do you agree or disagree that you feel close to people at your school? | 1=Strongly agree, 2=Agree, 3=Not sure, 4=Disagree, 5=Strongly disagree | School belonging |
| **Q105** | **Unfair_school_discipline** | During the past 12 months, have you been unfairly disciplined at school? | 1=Yes, 2=No | School discipline |

## 👨‍👩‍👧‍👦 Family Environment

| Variable Code | Variable Name | Question Text | Response Options | Notes |
|---------------|---------------|---------------|------------------|-------|
| **Q89** | **Parental_verbal_abuse** | During your life, how often has a parent or other adult in your home insulted you or put you down? | 1=Never, 2=Rarely, 3=Sometimes, 4=Most of the time, 5=Always | Emotional abuse |
| **Q90** | **Parental_physical_abuse** | During your life, how often has a parent or other adult in your home hit, beat, kicked, or physically hurt you in any way? | 1=Never, 2=Rarely, 3=Sometimes, 4=Most of the time, 5=Always | Physical abuse |
| **Q91** | **Parental_domestic_violence** | During your life, how often have your parents or other adults in your home slapped, hit, kicked, punched, or beat each other up? | 1=Never, 2=Rarely, 3=Sometimes, 4=Most of the time, 5=Always | Domestic violence exposure |
| **Q99** | **Adult_care_basic_needs** | During your life, how often has there been an adult in your household who tried hard to make sure your basic needs were met? | 1=Never, 2=Rarely, 3=Sometimes, 4=Most of the time, 5=Always | Parental care |
| **Q100** | **Parent_substance_abuse** | Have you ever lived with a parent or guardian who was having a problem with alcohol or drug use? | 1=Yes, 2=No | Parental substance abuse |
| **Q101** | **Parent_mental_illness** | Have you ever lived with a parent or guardian who had severe depression, anxiety, or another mental illness, or was suicidal? | 1=Yes, 2=No | Parental mental illness |
| **Q102** | **Parent_incarceration** | Have you ever been separated from a parent or guardian because they went to jail, prison, or a detention center? | 1=Yes, 2=No | Parental incarceration |
| **Q104** | **Parental_monitoring** | How often do your parents or other adults in your family know where you are going or with whom you will be? | 1=Never, 2=Rarely, 3=Sometimes, 4=Most of the time, 5=Always | Parental supervision |

## 🚗 Safety & Risk Behaviors

| Variable Code | Variable Name | Question Text | Response Options | Notes |
|---------------|---------------|---------------|------------------|-------|
| **Q8** | **Seatbelt_use** | How often do you wear a seat belt when riding in a car driven by someone else? | 1=Never, 2=Rarely, 3=Sometimes, 4=Most of the time, 5=Always | Safety behavior |
| **Q9** | **Ride_with_drinking_driver** | During the past 30 days, how many times did you ride in a car or other vehicle driven by someone who had been drinking alcohol? | 1=0 times, 2=1 time, 3=2-3 times, 4=4-5 times, 5=6+ times | Risky transportation |
| **Q12** | **Weapon_at_school** | During the past 30 days, on how many days did you carry a weapon such as a gun, knife, or club on school property? | 1=0 days, 2=1 day, 3=2-3 days, 4=4-5 days, 5=6+ days | Weapon carrying |
| **Q13** | **Gun_carrying** | During the past 12 months, on how many days did you carry a gun? | 1=0 days, 2=1 day, 3=2-3 days, 4=4-5 days, 5=6+ days | Gun carrying |
| **Q14** | **Safety_concerns_school** | During the past 30 days, on how many days did you not go to school because you felt you would be unsafe at school or on your way to or from school? | 1=0 days, 2=1 day, 3=2-3 days, 4=4-5 days, 5=6+ days | School safety concerns |
| **Q16** | **Physical_fighting** | During the past 12 months, how many times were you in a physical fight? | 1=0 times, 2=1 time, 3=2-3 times, 4=4-5 times, 5=6-7 times, 6=8-9 times, 7=10-11 times, 8=12+ times | Fighting behavior |
| **Q18** | **Saw_violence_neighborhood** | Have you ever seen someone get physically attacked, beaten, stabbed, or shot in your neighborhood? | 1=Yes, 2=No | Violence exposure |

## 💊 Substance Use

| Variable Code | Variable Name | Question Text | Response Options | Notes |
|---------------|---------------|---------------|------------------|-------|
| **Q33** | **Current_cigarette_use** | During the past 30 days, on how many days did you smoke cigarettes? | 1=0 days, 2=1-2 days, 3=3-5 days, 4=6-9 days, 5=10-19 days, 6=20-29 days, 7=All 30 days | Cigarette use |
| **Q36** | **Current_electronic_vapor_use** | During the past 30 days, on how many days did you use an electronic vapor product? | 1=0 days, 2=1-2 days, 3=3-5 days, 4=6-9 days, 5=10-19 days, 6=20-29 days, 7=All 30 days | E-cigarette use |
| **Q42** | **Current_alcohol_use** | During the past 30 days, on how many days did you have at least one drink of alcohol? | 1=0 days, 2=1-2 days, 3=3-5 days, 4=6-9 days, 5=10-19 days, 6=20-29 days, 7=All 30 days | Alcohol use |
| **Q43** | **Binge_drinking** | During the past 30 days, on how many days did you have 4 or more drinks in a row (females) or 5 or more drinks in a row (males)? | 1=0 days, 2=1 day, 3=2 days, 4=3-5 days, 5=6-9 days, 6=10-19 days, 7=20+ days | Binge drinking |
| **Q48** | **Current_marijuana_use** | During the past 30 days, how many times did you use marijuana? | 1=0 times, 2=1-2 times, 3=3-9 times, 4=10-19 times, 5=20-39 times, 6=40+ times | Marijuana use |
| **Q49** | **Prescription_pain_medicine_misuse** | During your life, how many times have you taken prescription pain medicine without a doctor's prescription? | 1=0 times, 2=1-2 times, 3=3-9 times, 4=10-19 times, 5=20-39 times, 6=40+ times | Prescription drug misuse |

## 🏃‍♂️ Physical Health & Nutrition

| Variable Code | Variable Name | Question Text | Response Options | Notes |
|---------------|---------------|---------------|------------------|-------|
| **Q6** | **Height** | How tall are you without your shoes on? | Free text (converted to meters) | Height measurement |
| **Q7** | **Weight** | How much do you weigh without your shoes on? | Free text (converted to kilograms) | Weight measurement |
| **Q66** | **Weight_perception** | How do you describe your weight? | 1=Very underweight, 2=Slightly underweight, 3=About right, 4=Slightly overweight, 5=Very overweight | Weight perception |
| **Q75** | **Breakfast_frequency** | During the past 7 days, on how many days did you eat breakfast? | 1=0 days, 2=1 day, 3=2 days, 4=3 days, 5=4 days, 6=5 days, 7=6 days, 8=7 days | Breakfast eating |
| **Q76** | **Physical_activity_60min** | During the past 7 days, on how many days were you physically active for at least 60 minutes? | 1=0 days, 2=1 day, 3=2 days, 4=3 days, 5=4 days, 6=5 days, 7=6 days, 8=7 days | Physical activity |
| **Q85** | **Sleep_hours** | On an average school night, how many hours of sleep do you get? | 1=4 or less, 2=5 hours, 3=6 hours, 4=7 hours, 5=8 hours, 6=9 hours, 7=10+ hours | Sleep duration |
| **Q86** | **Housing_stability** | During the past 30 days, where did you usually sleep? | 1=Parent's/guardian's home, 2=Friend's/family's home, 3=Shelter, 4=Motel/hotel, 5=Car/park/public place, 6=No usual place, 7=Somewhere else | Housing situation |
| **bmipct** | **BMI_percentile** | Body Mass Index percentile for age and sex | Continuous variable (0-100) | CDC growth chart percentile |

## 💻 Technology & Social Media

| Variable Code | Variable Name | Question Text | Response Options | Notes |
|---------------|---------------|---------------|------------------|-------|
| **Q80** | **Social_media_use** | How often do you use social media? | 1=Do not use, 2=Few times/month, 3=About once/week, 4=Few times/week, 5=About once/day, 6=Several times/day, 7=About once/hour, 8=More than once/hour | Social media frequency |

## 🔄 Data Processing Notes

### Missing Data Codes
- **Blank/Missing:** No response provided
- **System Missing:** Question not applicable (skip pattern)
- **Edited to Missing:** Response failed quality checks

### Variable Transformations
- **Height/Weight:** Converted to metric units (meters/kilograms)
- **BMI:** Calculated as weight(kg)/height(m)²
- **Race/Ethnicity:** Multiple race responses combined into single variable
- **Binary Recoding:** For analysis, Q26 recoded as 1=Yes(persistent sadness), 0=No

### Quality Filters Applied
- **Minimum Sample Size:** 500+ valid responses per variable
- **Correlation Threshold:** |r| > 0.05 for meaningful relationships
- **Biologically Implausible Values:** Height/weight outside CDC ranges set to missing

### Variables Excluded from Analysis
- **QN Variables:** Derived dichotomous variables (e.g., QN26, QN27)
- **qn Variables:** Supplemental calculated variables (e.g., qnfrcig, qnobese)
- **Survey Design:** stratum, psu, weight (used for survey weighting only)

## 📊 Variable Categories Summary

| Category | Count | Examples |
|----------|--------|----------|
| **Demographics** | 5 | Age, sex, grade, race/ethnicity |
| **Mental Health** | 6 | Suicide ideation, depression, concentration |
| **Family Environment** | 8 | Parental abuse, substance use, monitoring |
| **School Environment** | 6 | Bullying, connectedness, grades |
| **Substance Use** | 15 | Alcohol, tobacco, marijuana, prescription drugs |
| **Safety & Violence** | 10 | Fighting, weapons, neighborhood violence |
| **Physical Health** | 12 | BMI, nutrition, sleep, physical activity |
| **Sexual Behavior** | 8 | Activity, contraception, consent |
| **Technology** | 1 | Social media use |

## 🎯 Key Predictor Variables

Based on correlation analysis, the strongest predictors of persistent sadness (Q26) are:

### Top Risk Factors (Negative Correlations)
1. **Q84 - Current_mental_health** (r = -0.519)
2. **Q89 - Parental_verbal_abuse** (r = -0.412)
3. **Q30 - Injurious_suicide_attempt** (r = -0.314)
4. **Q29 - Attempted_suicide** (r = -0.278)
5. **Q63 - Sex_of_sexual_contacts** (r = -0.274)

### Top Protective Factors (Positive Correlations)
1. **Q106 - Concentration_difficulty** (r = 0.493)
2. **Q27 - Considered_suicide** (r = 0.491)
3. **Q28 - Made_suicide_plan** (r = 0.431)
4. **Q101 - Parent_mental_illness** (r = 0.341)
5. **Q35 - Ever_electronic_vapor_use** (r = 0.278)

*Note: Interpretation depends on variable coding. See individual variable descriptions for proper interpretation.*

---

*This data dictionary is based on the 2023 YRBSS Data User's Guide from the CDC. For complete survey methodology and additional variables, refer to the official CDC documentation.*