# Business Understanding

## Project Overview
 Analysis of Vaccination Patterns from the National 2009 H1N1 Flu Survey
 
## Business problem
A vaccine for the H1N1 flu virus became publicly available in October 2009. In late 2009 and early 2010, the United States conducted the National 2009 H1N1 Flu Survey. This phone survey asked respondents whether they had received the H1N1 and seasonal flu vaccines, in conjunction with questions about themselves. These additional questions covered their social, economic, and demographic background, opinions on risks of illness and vaccine effectiveness, and behaviors towards mitigating transmission. A better understanding of how these characteristics are associated with personal vaccination patterns can provide guidance for future public health efforts.
 
## Project objectives:
## Main Objective  
To analyze the demographic characteristics of respondents, including age, education, income, employment, and household composition.  

## Specific Objectives  
1. **Age Distribution** – Examine the age group distribution among respondents.  
  - **Feature Used:** age_group  

2. **Educational Attainment** – Analyze the levels of education across different respondents.  
  - **Feature Used:** education  

3. **Income and Employment Status** – Assess variations in income levels and employment status.  
  - **Features Used:** income_poverty, employment_status, employment_industry, employment_occupation  

4. **Household Composition** – Investigate household structure based on marital status, homeownership, and number of adults/children.  
  - **Features Used:** marital_status, rent_or_own, household_adults, household_children  

5. **Geographic Demographics** – Identify demographic variations across different regions.  
  - **Features Used:** hhs_geo_region, census_msa  


# Data Understanding 

## Data collection
The data for this competition comes from the National 2009 H1N1 Flu Survey (NHFS).

In their own words:

>The National 2009 H1N1 Flu Survey (NHFS) was sponsored by the National Center for Immunization and Respiratory Diseases (NCIRD) and conducted jointly by NCIRD and the National Center for Health Statistics (NCHS), Centers for Disease Control and Prevention (CDC). The NHFS was a list-assisted random-digit-dialing telephone survey of households, designed to monitor influenza immunization coverage in the 2009-10 season.

The target population for the NHFS was all persons 6 months or older living in the United States at the time of the interview. Data from the NHFS were used to produce timely estimates of vaccination coverage rates for both the monovalent pH1N1 and trivalent seasonal influenza vaccines.

The NHFS was conducted between October 2009 and June 2010. It was one-time survey designed specifically to monitor vaccination during the 2009-2010 flu season in response to the 2009 H1N1 pandemic. The CDC has other ongoing programs for annual phone surveys that continue to monitor seasonal flu vaccination.


# Conclusion

 
- **Logistic Regression**: 0.63 Accuracy
- **Decision Tree Classifier**: 0.59 Accuracy
- **Random Forest**: 0.60 Accuracy
- **Support Vector Machine (SVM)**: 0.62 Accuracy
- **Naive Bayes**: 0.62 Accuracy
Logistic Regression achieved the highest accuracy at **0.63**, making it the best-performing model. Decision Tree Classifier had the lowest accuracy at **0.59**. Random Forest, Naive Bayes and SVM performed similarly with accuracies of **0.60**, **0.62** and **0.62**, respectively.
