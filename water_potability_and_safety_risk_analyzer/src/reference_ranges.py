"""
reference_ranges.py

Reference "safe" ranges for each water quality parameter.

These are approximate acceptable ranges commonly cited for drinking water
(loosely based on WHO / BIS guidance and the value ranges used by the
Kaggle "Water Quality and Potability" dataset documentation). They are
used only to FLAG individual parameters as unsafe/borderline for
explanation purposes -- the actual potability prediction is made by the
trained ML model, not by these rules.

NOTE (responsible use): these ranges are simplified for an educational/
student project and must not be used as the sole basis for a real
drinking-water safety decision. Always confirm with a certified water
testing lab / local health authority.
"""

# Each entry: (min_safe, max_safe, unit, short description)
SAFE_RANGES = {
    "ph": (6.5, 8.5, "", "Acceptable pH range for drinking water"),
    "Hardness": (0, 300, "mg/L", "Soft-to-moderately-hard water is preferred"),
    "Solids": (0, 500, "ppm", "Total Dissolved Solids (TDS), WHO desirable limit"),
    "Chloramines": (0, 4, "ppm", "Max allowed disinfectant residual"),
    "Sulfate": (0, 250, "mg/L", "Upper limit before taste/health issues"),
    "Conductivity": (0, 400, "\u03bcS/cm", "Typical acceptable conductivity"),
    "Organic_carbon": (0, 2, "ppm", "US EPA guidance for treated water (TOC)"),
    "Trihalomethanes": (0, 80, "\u03bcg/L", "US EPA maximum contaminant level"),
    "Turbidity": (0, 5, "NTU", "WHO recommended maximum turbidity"),
}

FEATURE_COLUMNS = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity",
]

TARGET_COLUMN = "Potability"
