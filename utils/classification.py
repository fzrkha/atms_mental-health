import pandas as pd

from utils.preprocessing import preprocessing_data


LEXICONS = {
    "mild": [
        "capek", "sedih", "bingung", "lelah", "nangis",
        "takut", "kecewa", "stres", "stress", "pusing",
        "overthinking", "jahat", "cemas",
        "lemes", "lemas", "marah", "exhausted",
        "gila", "pahit", "panik"
    ],

    "severe": [
        "depresi", "trauma", "mati", "menyerah",
        "putus asa", "hancur", "hilang", "ga kuat",
        "gagal", "sendiri", "burnout", "muntah",
        "insomnia", "mentally exhausted",
        "gerd", "ptsd", "anxiety",
        "panic attack"
    ],

    "amplifier": [
        "banget", "parah", "sangat", "terus",
        "sekarang", "lagi", "darurat", "tolong",
        "berat", "pendam", "mendem", "mendam",
        "memendam", "nahan", "tahan",
        "tekan", "tertekan", "teken"
    ],

    "self_harm": [
        "bunuh diri",
        "pengen mati",
        "ga sanggup",
        "capek hidup",
        "hidup ga ada arti",
        "udah selesai",
        "pengen hilang"
    ]
}


def get_auto_label(text):

    if not isinstance(text, str):
        return pd.Series(["Curhat Ringan", 0])

    text = text.lower()

    mild_count = sum(
        1 for w in LEXICONS["mild"]
        if w in text
    )

    severe_count = sum(
        1 for w in LEXICONS["severe"]
        if w in text
    )

    amplifier_count = sum(
        1 for w in LEXICONS["amplifier"]
        if w in text
    )

    self_harm_count = sum(
        1 for w in LEXICONS["self_harm"]
        if w in text
    )

    score = (
        mild_count * 1
        + severe_count * 3
        + self_harm_count * 5
    )

    if severe_count > 0:
        score += amplifier_count * 1.5

    label = (
        "Butuh Pertolongan Segera"
        if (
            severe_count >= 2
            or self_harm_count >= 1
            or score >= 5
        )
        else "Curhat Ringan"
    )

    return pd.Series([label, score])


def classification_data():

    df = preprocessing_data().copy()

    df[["auto_label", "distress_score"]] = (
        df["clean_text"].apply(get_auto_label)
    )

    return df