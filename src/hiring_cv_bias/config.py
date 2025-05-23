import os
from pathlib import Path

DATA_DIR = str(Path(__file__).parent.parent.parent).replace(os.sep, "/") + "/data/"
CV_DIR = DATA_DIR + "Adecco_Dataset_Rev_match_parsed_cvs"

PARSED_DATA_PATH = CV_DIR + "/Candidate_CVs_extracted_data.csv"
CANDIDATE_CVS_PATH = CV_DIR + "/Candidate_CVs.csv"
REVERSE_MATCHING_PATH = CV_DIR + "/ReverseMatching.xlsx"
CANDIDATE_CVS_TRANSLATED_PATH = CV_DIR + "/Candidate_CVs_translated.csv"
JOBS_PATH = DATA_DIR + "occupations_en.csv"

JOB_LINKS = [
    "https://www.zippia.com/baby-sitter-jobs/demographics/",
    "https://www.zippia.com/construction-worker-jobs/demographics/",
    "https://www.zippia.com/secretary-jobs/demographics/",
    "https://www.zippia.com/maintenance-technician-jobs/demographics/",
    "https://www.zippia.com/electrician-jobs/demographics/",
    "https://www.zippia.com/delivery-driver-jobs/demographics/",
    "https://www.zippia.com/motor-vehicle-assembler-jobs/demographics/",
    "https://www.zippia.com/chauffeur-jobs/demographics/",
    "https://www.zippia.com/warehouse-employee-jobs/demographics/",
    "https://www.zippia.com/car-mechanic-jobs/demographics/",
    "https://www.zippia.com/plumber-jobs/demographics/",
    "https://www.zippia.com/bricklayer-jobs/demographics/",
    "https://www.zippia.com/carpenter-jobs/demographics/",
    "https://www.zippia.com/forklift-operator-jobs/demographics/",
]

CV_CLEANED_DIR = DATA_DIR + "Adecco_Dataset_cleaned"
CANDIDATE_CVS_TRANSLATED_CLEANED_PATH = CV_CLEANED_DIR + "CV_translated_cleaned.csv"
CLEANED_SKILLS = CV_CLEANED_DIR + "/Skills_cleaned.csv"