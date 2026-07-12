"""
utils.py

Helper functions for:
- Flagging individual water quality parameters as safe / borderline / unsafe
- Building a simple, human-readable "safety risk" explanation for a sample
"""

from reference_ranges import SAFE_RANGES


def flag_parameters(sample: dict) -> list:
    """
    Given a dict of {parameter_name: value}, compare each value against
    SAFE_RANGES and return a list of flag dicts describing whether each
    parameter is within the safe range.

    Returns a list of dicts:
        {
            "parameter": str,
            "value": float,
            "safe_range": (min, max),
            "unit": str,
            "status": "Safe" | "Unsafe",
            "note": str
        }
    """
    flags = []
    for param, (low, high, unit, desc) in SAFE_RANGES.items():
        value = sample.get(param)
        if value is None:
            continue
        is_safe = low <= value <= high
        flags.append({
            "parameter": param,
            "value": round(float(value), 3),
            "safe_range": (low, high),
            "unit": unit,
            "status": "Safe" if is_safe else "Unsafe",
            "note": desc,
        })
    return flags


def risk_score_from_flags(flags: list) -> dict:
    """
    Convert a list of parameter flags into an overall risk score (0-100,
    higher = riskier) and a risk band.
    """
    total = len(flags)
    unsafe = sum(1 for f in flags if f["status"] == "Unsafe")
    score = round((unsafe / total) * 100, 1) if total else 0.0

    if score == 0:
        band = "Low Risk"
    elif score <= 30:
        band = "Moderate Risk"
    elif score <= 60:
        band = "High Risk"
    else:
        band = "Very High Risk"

    return {
        "unsafe_parameter_count": unsafe,
        "total_parameters_checked": total,
        "risk_score": score,
        "risk_band": band,
    }


def build_recommendation(model_prediction: int, risk_info: dict, flags: list) -> str:
    """
    Combine the ML model's prediction with the rule-based parameter
    risk score to produce a short, plain-English recommendation.
    """
    unsafe_params = [f["parameter"] for f in flags if f["status"] == "Unsafe"]

    if model_prediction == 1 and risk_info["risk_band"] == "Low Risk":
        return (
            "This sample is predicted POTABLE and no individual parameters "
            "breach safe limits. Water appears safe to drink based on the "
            "tested parameters."
        )

    if model_prediction == 1 and unsafe_params:
        return (
            "The model predicts this sample is POTABLE overall, but "
            f"{', '.join(unsafe_params)} fall outside recommended safe "
            "ranges. Consider verifying with a lab test before use."
        )

    if model_prediction == 0 and not unsafe_params:
        return (
            "The model predicts this sample is NOT POTABLE, even though no "
            "single parameter clearly breaches the reference ranges used "
            "here. This may be due to a combination effect across "
            "parameters -- recommend a professional lab test."
        )

    return (
        "This sample is predicted NOT POTABLE. "
        f"Parameters of concern: {', '.join(unsafe_params) if unsafe_params else 'combination of factors'}. "
        "Do not consume without treatment; recommend retesting and "
        "consulting a water treatment / public health authority."
    )
