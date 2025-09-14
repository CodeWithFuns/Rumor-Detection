import re
import nltk
from nltk.corpus import stopwords

# best-effort downloads (won't crash if offline after first success)
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception:
    pass

_stop = set(stopwords.words('english')) if stopwords.words('english') else set()

def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def simple_tokenize(text: str):
    # super light tokenizer to keep deps small
    return [t for t in clean_text(text).split() if t and t not in _stop]
