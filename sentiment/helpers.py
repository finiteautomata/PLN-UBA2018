import re
from nltk.tokenize import TweetTokenizer

urls = r'(?:https?\://t.co/[\w]+)'

def mytokenize(text, remove_hashtags=True):
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
    tokens = tokenizer.tokenize(text)

    tokens = [tk for tk in tokens if tk[0] != "#"]
    tokens = [tk for tk in tokens if not re.match(urls, tk)]
    tokens = [re.sub(r'(.)\1\1+', r'\1\1', tk) for tk in tokens]
    return tokens
