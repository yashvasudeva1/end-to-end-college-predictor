def classify_tier(user_rank, closing_rank):
    ratio = user_rank / closing_rank
    if ratio <= 0.8:
        return "SAFE"
    elif ratio <= 1.0:
        return "MODERATE"
    elif ratio <= 1.1:
        return "REACH"
    else:
        return "UNLIKELY"

def predicted_chance(user_rank, predicted_rank):
    ratio = user_rank / predicted_rank
    score = 1.3 - ratio
    score = max(0, min(score, 1))
    return round(score * 100, 2)
