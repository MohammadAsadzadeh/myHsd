import re

class Preprocessor():
    def __init__(self, normalizer, punctuation):
        self.normalizer = normalizer
        self.punctuation = punctuation
    
    def __call__(self, text : str, method= "rule_based"):
        if method == "rule_based":
            text = self.normalizer.normalize(text)
            text = re.sub("http[^\s]+", "", text)
            text = re.sub("["+ self.punctuation + "]", " ", text)
            text = re.sub("\s+", " ", text)
            text = re.sub("^([\sa-zA-Z0-9_]+\s)+", "", text)
            return text
        elif method == "bert_based":
            text = text.lower()
            text = re.sub("#","",text)
            text = re.sub("@[^\s]+", "@USERNAME",text)
            text = re.sub("http[^\s]+", "", text)
            text = re.sub("\s+", " ", text)
            text = re.sub(r'([ا-ی])\1{2,}', r"\1", text)
            text = re.sub(r'\.', "", text)
            text = re.sub(r"([^\s\?\!\)\.,:\{\}]+)([\?\!\)\.,:\{\}]+)", r"\1 \2", text)
            text = re.sub(r"([\(\.,:\{\}]+)([^\s\?\!\)\.,:\{\}]+)", r"\1 \2", text)
            return " ".join([t for t in re.split("[\s\+\/\-_ـ]",text) if t not in  ["",".", ",", "..", ":", "(", "(", "{", "}","_"]])
        else:
            return text