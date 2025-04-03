from pathlib import Path
import os


DATA_DIR = str(Path(__file__).parent.parent.parent).replace(os.sep,'/') +'/data/'
CV_DIR = DATA_DIR+'Adecco_Dataset_Rev_match_parsed_cvs/'


PARSED_DATA_PATH = (
    CV_DIR+"Candidate_CVs_extracted_data.csv"
)
CANDIDATE_CVS_PATH = CV_DIR+"Candidate_CVs.csv"
REVERSE_MATCHING_PATH = (
    CV_DIR+"ReverseMatching.xlsx"
)

CVS_TRANSLATED_PATH = (
    CV_DIR+ "Candidate_CVs_translated.csv"
)


JOB_TITLES = {
    # Tecnologia e Informatica
    "Software Developer", "Data Scientist", "Data Analyst", "Web Developer", "Front-end Developer",
    "Back-end Developer", "Full Stack Developer", "DevOps Engineer", "Cloud Architect", 
    "Machine Learning Engineer", "Artificial Intelligence Specialist", "System Administrator", 
    "Database Administrator", "Cybersecurity Analyst", "IT Support Specialist", "Network Engineer", 
    "Mobile App Developer", "UI/UX Designer", "QA Engineer", "IT Project Manager", "Business Intelligence Analyst",

    # Marketing e Comunicazione
    "Digital Marketing Specialist", "SEO Specialist", "Content Strategist", "Social Media Manager", 
    "Brand Manager", "Public Relations Specialist", "Marketing Analyst", "Copywriter", "Graphic Designer", 
    "Email Marketing Manager", "Event Planner", "Marketing Director", "Creative Director", "PPC Specialist",
    "Product Marketing Manager", "Market Research Analyst", "Growth Hacker",

    # Finanza e Contabilità
    "Financial Analyst", "Accountant", "Auditor", "Tax Consultant", "Finance Manager", 
    "Investment Analyst", "Portfolio Manager", "Risk Manager", "Credit Analyst", "Controller", 
    "Bookkeeper", "Chief Financial Officer", "Investment Banker", "Corporate Treasurer", "Financial Planner",

    # Gestione e Amministrazione
    "Project Manager", "Operations Manager", "Business Analyst", "Supply Chain Manager", 
    "Human Resources Manager", "Recruitment Specialist", "Executive Assistant", "Office Manager", 
    "Facilities Manager", "Procurement Specialist", "Customer Service Manager", "Business Development Manager",
    "Operations Director", "Product Manager", "General Manager", "Management Consultant",

    # Vendite e Servizi
    "Sales Manager", "Account Executive", "Sales Associate", "Sales Representative", "Account Manager", 
    "Inside Sales Representative", "Retail Manager", "Customer Success Manager", "B2B Sales Specialist", 
    "Sales Operations Manager", "Field Sales Representative", "Regional Sales Manager", "Sales Director",

    # Ricerca e Sviluppo
    "Research Scientist", "Biochemist", "Biomedical Engineer", "Clinical Research Associate", "Product Manager",
    "Innovation Manager", "R&D Engineer", "Pharmaceutical Researcher", "Environmental Scientist", 
    "Chemist", "Laboratory Technician", "Agricultural Scientist", "Clinical Researcher", 
    "Product Development Specialist", "Medical Researcher",

    # Legale e Compliance
    "Lawyer", "Legal Counsel", "Paralegal", "Compliance Officer", "Contract Manager", "Corporate Lawyer", 
    "Intellectual Property Specialist", "Legal Assistant", "Litigation Attorney", 
    "Criminal Defense Attorney", "Mergers & Acquisitions Specialist", "Family Law Attorney", 
    "Real Estate Attorney",

    # Educazione e Formazione
    "Teacher", "Lecturer", "Educational Consultant", "Curriculum Developer", "Instructional Designer", 
    "Research Fellow", "School Counselor", "Principal", "Teaching Assistant", "Special Education Teacher", 
    "Professor", "University Lecturer", "Corporate Trainer", "Educational Administrator", "Tutor",

    # Arte e Design
    "Art Director", "Graphic Designer", "Illustrator", "Animator", "Architect", "Interior Designer", 
    "Fashion Designer", "Photographer", "Video Editor", "Product Designer", "Multimedia Artist", 
    "Visual Designer", "Web Designer", "3D Modeler", "Creative Director", "Exhibit Designer", 
    "Fashion Stylist", "Set Designer",

    # Sanità e Medicina
    "Doctor", "Nurse", "Pharmacist", "Medical Technician", "Physical Therapist", "Surgeon", 
    "Healthcare Administrator", "Dentist", "Optometrist", "Radiologist", "Occupational Therapist", 
    "Medical Researcher", "Healthcare Consultant", "Dietitian", "Veterinarian", "Chiropractor", 
    "Pediatrician", "Psychiatrist", "Clinical Psychologist", "Speech Therapist",

    # Logistica e Trasporti
    "Logistics Manager", "Supply Chain Coordinator", "Warehouse Manager", "Fleet Manager", 
    "Shipping Coordinator", "Transport Planner", "Route Planner", "Logistics Analyst", 
    "Import/Export Specialist", "Inventory Manager", "Freight Forwarder", "Truck Driver", 
    "Delivery Driver", "Logistics Engineer",

    # Turismo e Ospitalità
    "Hotel Manager", "Restaurant Manager", "Tourism Director", "Concierge", "Travel Agent", 
    "Event Coordinator", "Housekeeper", "Waiter", "Chef", "Bartender", "Tour Guide", 
    "Spa Manager", "Hospitality Manager", "Food & Beverage Director", "Hotel Receptionist", 
    "Event Planner",

    # Energia e Ambiente
    "Energy Consultant", "Environmental Engineer", "Sustainability Manager", "Renewable Energy Specialist", 
    "Energy Auditor", "Water Resources Manager", "Environmental Consultant", "Geologist", 
    "Climate Change Analyst", "Environmental Scientist", "Solar Panel Technician", "Wind Energy Technician", 
    "Marine Biologist",

    # Tecnologia dell'Informazione e Intelligenza Artificiale
    "AI Research Scientist", "Robotics Engineer", "Blockchain Developer", "Quantum Computing Researcher", 
    "AI Engineer", "Big Data Engineer", "Automation Engineer", "Chatbot Developer", "Data Engineer", 
    "IoT Specialist", "AI Software Developer", "AI Business Consultant", "Ethical Hacker", 
    "Game Developer", "Virtual Reality Developer",

    # Altri Ruoli
    "Retail Manager", "Customer Service Representative", "Administrative Assistant", "Security Guard", 
    "Waitress", "Cashier", "Bank Teller", "Bartender", "Receptionist", "Maintenance Worker", 
    "Janitor", "Factory Worker", "Construction Worker", "Electrician", "Plumber", "Carpenter", 
    "Painter", "Driver", "Cleaner", "Laborer", "Security Analyst", "Event Planner"
}
