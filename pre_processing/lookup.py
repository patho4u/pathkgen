# Split rules
SPLIT_RULES = {

    "Cervical Squamous Cell Carcinoma and Endocervical Adenocarcinoma": [
        ("Cervical Squamous Cell Carcinoma", [
            r"squamous",
            r"scc",
            r"keratinizing",
            r"non[- ]keratinizing"
        ]),
        ("Endocervical Adenocarcinoma", [
            r"adenocarcinoma",
            r"endocervical",
            r"glandular"
        ])
    ],

    "Pheochromocytoma and Paraganglioma": [
        ("Pheochromocytoma", [
            r"pheochromocytoma",
            r"adrenal medulla"
        ]),
        ("Paraganglioma", [
            r"paraganglioma",
            r"extra[- ]adrenal"
        ])
    ]
}

# FINAL CLASS NAME REMAPPING (31 → desired names)
FINAL_CLASS_REMAP = {

    # Split classes
    "Cervical Squamous Cell Carcinoma": "Cervical Squamous Cell Carcinoma",
    "Endocervical Adenocarcinoma": "Endocervical Adenocarcinoma",
    "Pheochromocytoma": "Pheochromocytoma",
    "Paraganglioma": "Paraganglioma",

    # Examples of unchanged TCGA projects (add all 29)
    "Breast Invasive Carcinoma": "Breast Invasive Carcinoma",
    "Lung Adenocarcinoma": "Lung Adenocarcinoma",
    "Lung Squamous Cell Carcinoma": "Lung Squamous Cell Carcinoma",
    "Kidney Renal Clear Cell Carcinoma": "Kidney Renal Clear Cell Carcinoma",
    "Kidney Renal Papillary Cell Carcinoma": "Kidney Renal Papillary Cell Carcinoma",
    "Kidney Chromophobe": "Kidney Chromophobe",
    "Prostate Adenocarcinoma": "Prostate Adenocarcinoma",
    "Bladder Urothelial Carcinoma": "Bladder Urothelial Carcinoma",
    "Colon Adenocarcinoma": "Colon Adenocarcinoma",
    "Stomach Adenocarcinoma": "Stomach Adenocarcinoma",
    "Liver Hepatocellular Carcinoma": "Liver Hepatocellular Carcinoma",
    "Ovarian Serous Cystadenocarcinoma": "Ovarian Serous Cystadenocarcinoma",
    "Uterine Corpus Endometrial Carcinoma": "Uterine Corpus Endometrial Carcinoma",
    "Sarcoma": "Sarcoma",
    "Skin Cutaneous Melanoma": "Skin Cutaneous Melanoma",
    "Brain Lower Grade Glioma": "Brain Lower Grade Glioma",
    "Thyroid Carcinoma": "Thyroid Carcinoma",
    "Esophageal Carcinoma": "Esophageal Carcinoma",
    "Mesothelioma": "Mesothelioma",
    "Uveal Melanoma": "Uveal Melanoma",
    "Cholangiocarcinoma": "Cholangiocarcinoma",
    "Rectum Adenocarcinoma": "Rectum Adenocarcinoma",
    "Adrenocortical Carcinoma": "Adrenocortical Carcinoma",
    "Thymoma": "Thymoma",
    "Uterine Carcinosarcoma": "Uterine Carcinosarcoma",
    "Lymphoid Neoplasm Diffuse Large B-cell Lymphoma": "Lymphoid Neoplasm Diffuse Large B-cell Lymphoma",
    "Head and Neck Squamous Cell Carcinoma": "Head and Neck Squamous Cell Carcinoma",
    "Testicular Germ Cell Tumors": "Testicular Germ Cell Tumors"
}

# UMLS CUI mapping for each final class name
HIST_TO_CUI = {
    # Split classes
    "Cervical Squamous Cell Carcinoma": "C0279671",  # Cervical Squamous Cell Carcinoma
    "Endocervical Adenocarcinoma":      "C0279672",  # Endocervical Adenocarcinoma
    "Pheochromocytoma":                "C0031511",  # Pheochromocytoma
    "Paraganglioma":                   "C0030421",  # Paraganglioma

    # TCGA project classes
    "Breast Invasive Carcinoma":        "C0853879",  # Breast Invasive Carcinoma
    "Lung Adenocarcinoma":              "C0152013",  # Lung Adenocarcinoma
    "Lung Squamous Cell Carcinoma":     "C0149782",  # Lung Squamous Cell Carcinoma
    "Kidney Renal Clear Cell Carcinoma": "C0279702",  # Kidney Clear Cell RCC
    "Kidney Renal Papillary Cell Carcinoma": "C1306837",  # Kidney Papillary RCC
    "Kidney Chromophobe":              "C1266042",  # Kidney Chromophobe
    "Prostate Adenocarcinoma":         "C0007112",  # Prostate Adenocarcinoma
    "Bladder Urothelial Carcinoma":    "C2145472",  # Bladder Urothelial Carcinoma
    "Colon Adenocarcinoma":            "C0338106",  # Colon Adenocarcinoma
    "Stomach Adenocarcinoma":          "C0278701",  # Stomach Adenocarcinoma
    "Hepatocellular Carcinoma":        "C2239176",  # Hepatocellular Carcinoma
    "Ovarian Serous Cystadenocarcinoma":        "C0206701",  # Ovarian Serous Cystadenocarcinoma
    "Uterine Corpus Endometrial Carcinoma":           "C0476089",  # Uterine Corpus Endometrial Carcinoma
    "Sarcoma":                         "C1261473",  # Sarcoma
    "Skin Cutaneous Melanoma":         "C0151779",  # Skin Cutaneous Melanoma
    "Brain Lower Grade Glioma":         "C0017638",  # Brain Lower Grade Glioma
    "Thyroid Carcinoma":               "C0549473",  # Thyroid Carcinoma
    "Esophageal Carcinoma":            "C0152018",  # Esophageal Carcinoma
    "Mesothelioma":                    "C0025500",  # Mesothelioma
    "Uveal Melanoma":                  "C0220633",  # Uveal Melanoma
    "Cholangiocarcinoma":              "C0206698",  # Cholangiocarcinoma
    "Rectum Adenocarcinoma":           "C0007113",  # Rectum Adenocarcinoma
    "Adrenocortical Carcinoma":        "C0206686",  # Adrenocortical Carcinoma
    "Thymoma":                         "C0040100",  # Thymoma
    "Uterine Carcinosarcoma":          "C0280630",  # Uterine Carcinosarcoma
    "Diffuse Large B-cell Lymphoma":   "C0079744",  # Diffuse Large B-cell Lymphoma
    "Head and Neck Squamous Cell Carcinoma": "C1168401",  # Head and Neck Squamous Cell Carcinoma
    "Testicular Germ Cell Tumors":     "C1336708"
}