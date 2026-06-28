import re
import string
import streamlit as st

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from utils.load_data import load_data


factory = StemmerFactory()
stemmer = factory.create_stemmer()


slang_dict = {
    "gak":"tidak","ga":"tidak","nggak":"tidak","ngga":"tidak",
    "tdk":"tidak","tak":"tidak","tidakk":"tidak",

    "yg":"yang","utk":"untuk","dr":"dari","dgn":"dengan",
    "krn":"karena","krna":"karena","gim":"game",

    "bgt":"banget","bgtt":"banget","bngt":"banget","bangett":"banget",

    "skrg":"sekarang","skrng":"sekarang",
    "kmrn":"kemarin","bsk":"besok",

    "gw":"saya","gua":"saya","gue":"saya","aq":"saya",

    "lu":"kamu","loe":"kamu","lo":"kamu",

    "udah":"sudah","udh":"sudah","sdh":"sudah",
    "belom":"belum","blm":"belum",

    "kayak":"seperti","kaya":"seperti","kyk":"seperti",

    "klo":"kalau","kalo":"kalau","kl":"kalau",

    "gimana":"bagaimana","gmana":"bagaimana",

    "kenapa":"mengapa","knp":"mengapa",

    "pdhl":"padahal","padhl":"padahal",

    "sm":"sama","sma":"sama","ama":"sama",

    "nih":"ini","ni":"ini",

    "tuh":"itu","tu":"itu",

    "bener":"benar","bnr":"benar","bener2":"benar",

    "gpp":"gapapa","ywdh":"yaudah",

    "org":"orang","mrk":"mereka",

    "amp":"",
    "&amp;":"",

    "ajg":"anjing",
    "anjg":"anjing",
    "bgst":"bangsat",
    "bngsat":"bangsat",

    "selfdeclare":"self declare",
    "awalselfdeclare":"awal self declare"
}


def clean_text(text):

    if not isinstance(text, str):
        return ""

    text = text.lower().strip()

    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"&amp;", " ", text)

    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\d+", " ", text)

    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    for p in string.punctuation:
        text = text.replace(p, " ")

    text = re.sub(r"\s+", " ", text).strip()

    words = [
        slang_dict.get(word, word)
        for word in text.split()
    ]

    words = [
        stemmer.stem(word)
        for word in words
    ]

    return " ".join(words)


@st.cache_data
def preprocessing_data():

    df = load_data().copy()

    df["clean_text"] = (
        df["full_text"]
        .astype(str)
        .apply(clean_text)
    )

    df = df.dropna(subset=["clean_text"])

    df = df[df["clean_text"].str.strip() != ""]

    return df