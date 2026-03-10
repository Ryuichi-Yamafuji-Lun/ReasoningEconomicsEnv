"""Tag questions by difficulty bucket: gsm8k, math_l1_l2, math_l3, math_l4_l5."""

# MetaMathQA type field: GSM_*, MATH_*. MATH dataset has level 1-5.


def classify_question(item: dict) -> str:
    """Classify a MetaMathQA or MATH dataset row into one difficulty bucket.

    Returns one of: 'gsm8k', 'math_l1_l2', 'math_l3', 'math_l4_l5'.
    """
    raw_type = (item.get("type") or "").upper()
    level = item.get("level")

    if "GSM" in raw_type:
        return "gsm8k"

    if "MATH" in raw_type or level is not None:
        if level is not None:
            try:
                l = int(level)
                if l <= 2:
                    return "math_l1_l2"
                if l == 3:
                    return "math_l3"
                return "math_l4_l5"
            except (TypeError, ValueError):
                pass
        return "math_l3"

    return "math_l3"
