def localize(latitude: float) -> str:
    if latitude > 44.5:
        return "NORTH"
    if latitude < 40:
        return "SOUTH"
    return "CENTER"
