import os
EXPERIMENT_DIR = os.path.join(os.path.dirname(__file__), '..', 'experiments')
TAKEDA_LOG_DIR = os.path.join("/dbfs", "FileStore", "logs",)
AE_ALL_DIM = 12
EVENT_SHIFT = 0
TYPE_VAR_DICT = {"var-serum":
                 ["Immunoglobulin A (g/L)",
                  "Immunoglobulin G (g/L)",
                  "SPEP Kappa Light Chain, Free (mg/L)",
                  "SPEP Lambda Light Chain, Free (mg/L)",
                  "SPEP Monoclonal Protein (g/L)"],
                 "var-chem":
                     ['Albumin (g/L)',
                         'Alkaline Phosphatase (U/L)',
                         'Alanine Aminotransferase (U/L)',
                         'Aspartate Aminotransferase (U/L)',
                         'Blood Urea Nitrogen (mmol/L)',
                         'Calcium (mmol/L)',
                         'Chloride (mmol/L)',
                         'Carbon Dioxide (mmol/L)',
                         'Creatinine (umol/L)',
                         'Glucose (mmol/L)',
                         'Hemoglobin (g/L)',
                         'Potassium (mmol/L)'],
                     "var-all": []
                 }
