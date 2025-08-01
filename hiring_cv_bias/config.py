import os
from pathlib import Path

DATA_DIR = str(Path(__file__).parent.parent).replace(os.sep, "/") + "/data/"
CV_DIR = DATA_DIR + "Adecco_Dataset_Rev_match_parsed_cvs"
FUZZY_DATA_DIR = DATA_DIR + "fuzzy_data/"
PARSED_DATA_PATH = CV_DIR + "/Candidate_CVs_extracted_data.csv"
CANDIDATE_CVS_PATH = CV_DIR + "/Candidate_CVs.csv"
REVERSE_MATCHING_PATH = CV_DIR + "/ReverseMatching.xlsx"
CANDIDATE_CVS_TRANSLATED_PATH = CV_DIR + "/Candidate_CVs_translated.csv"
JOBS_PATH = FUZZY_DATA_DIR + "occupations_en.csv"
EXTRACTED_JOBS = FUZZY_DATA_DIR + "Parsed_Jobs.csv"
HARD_SOFT_SKILLS = DATA_DIR + "hard_soft_skills.csv"

JOB_LINKS = [
    "https://www.zippia.com/baby-sitter-jobs/demographics/",
    "https://www.zippia.com/construction-worker-jobs/demographics/",
    "https://www.zippia.com/secretary-jobs/demographics/",
    "https://www.zippia.com/maintenance-technician-jobs/demographics/",
    "https://www.zippia.com/electrician-jobs/demographics/",
    "https://www.zippia.com/chauffeur-jobs/demographics/",
    "https://www.zippia.com/motor-vehicle-assembler-jobs/demographics/",
    "https://www.zippia.com/delivery-driver-jobs/demographics/",
    "https://www.zippia.com/car-mechanic-jobs/demographics/",
    "https://www.zippia.com/room-attendant-jobs/demographics/",
    "https://www.zippia.com/plumber-jobs/demographics/",
    "https://www.zippia.com/bricklayer-jobs/demographics/",
    "https://www.zippia.com/warehouse-employee-jobs/demographics/",
    "https://www.zippia.com/painter-jobs/demographics/",
    "https://www.zippia.com/carpenter-jobs/demographics/",
]

CV_CLEANED_DIR = DATA_DIR + "Adecco_Dataset_cleaned"
CANDIDATE_CVS_TRANSLATED_CLEANED_PATH = CV_CLEANED_DIR + "/CV_translated_cleaned.csv"
CLEANED_SKILLS = CV_CLEANED_DIR + "/Skills_cleaned.csv"
CLEANED_REVERSE_MATCHING_PATH = (
    CV_CLEANED_DIR + "/reversed_skills_matching_candidate.csv"
)
DRIVING_LICENSE_FALSE_NEGATIVES_PATH = (
    CV_CLEANED_DIR + "/driving_license/false_negatives.csv"
)
LANGUAGE_SKILL_FALSE_NEGATIVES_PATH = (
    CV_CLEANED_DIR + "/language_skill/false_negatives.csv"
)
JOB_TITLE_FALSE_NEGATIVES_PATH = CV_CLEANED_DIR + "/job_title/false_negatives.csv"
