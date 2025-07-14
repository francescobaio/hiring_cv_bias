<div align="center">
<h1>Hiring CV bias</h1>

[Francesco Baiocchi](https://github.com/francescobaio), [Leonardo Petrilli](https://github.com/leonardopetrilli)
</div>

## Overview

LLMs can be subject to cultural biases. One such domain is recruiting, where LLMs are used to extract information from candidates’ CVs and then to select the most suitable profiles. These biases may favour or penalize applicants based on language or cultural cues, risking the exclusion of qualified talent.

The goal of this study is to evaluate the **potential biases** in a proprietary CV parser, which automatically extracts skills from anonymized, raw resumes.

https://github.com/user-attachments/assets/347d2a38-2b2a-4be4-899c-cd495bbbd5a7


> [!NOTE]
> Because the CV data, although anonymized is proprietary, it cannot be shared to reproduce the experiments using in the hiring_cv_bias module.


To explore the work in detail, visit the page here: [link to the page](#)


## Repository Structure

```text

── hiring_cv_bias
│   ├── bias_detection
│   │   ├── fuzzy
│   │   │   ├── matcher.py  # perfoms matching between our extracted and parser skills 
│   │   │   ├── parser.py  # performs exact matching on CVs with a list of jobs titles 
│   │   │   └── utils.py # job filtering  
│   │   └── rule_based
│   │       ├── app
│   │       │   └── fn_app.py # visualization app for extraction pipeline 
│   │       ├── evaluation
│   │       │   ├── compare_parser.py # computes bias detection metrics for each group 
│   │       │   └── metrics.py  
│   │       ├── extractors.py # extract and apply regex patterns 
│   │       ├── patterns.py # define patterns for exact matching 
│   │       └── utils.py
│   ├── cleaning
│   │   ├── common.py 
│   │   └── raw_cv.py # cleaning of corrupted CVs 
│   ├── config.py
│   ├── exploration
│   │   ├── gender_analysis.py # computes bias_strenght metric for each skill type   
│   │   ├── disparity.py # computes Gini-Index metric for each skill type 
│   │   ├── visualize.py # plotting functions for visualizing distribution
│   │   └── utils.py 
│   ├── hard_soft_skills_labelling
│   │   ├── CoT.log # full CoT of the model used 
│   │   ├── hard_soft_skill_labelling.ipynb 
│   │   └── utils.py # automates the labeling of extracted Professional_Skill entries as Hard, Soft, or Unknown
│   ├── translation
│   │   └── translate.py # script for translating CVs in English 
│   └── utils.py
│
├── notebooks
│   ├── data_cleaning.ipynb  
│   ├── data_exploration.ipynb
│   ├── distributions_analysis.ipynb
│   └── bias_detection.ipynb

```
