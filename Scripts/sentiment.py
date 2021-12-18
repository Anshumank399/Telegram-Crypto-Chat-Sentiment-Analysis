"""
@author: Anshuman Dey Kirty
Sentiment Analysis on crypto group chats from Telegram.
"""


import json
import nltk
import emoji
import string
import pandas as pd
from tqdm import tqdm
from nltk.corpus import stopwords
import plotly.express as px
from textblob import TextBlob
import plotly.graph_objects as go
from plotly.subplots import make_subplots

nltk.download("stopwords")
nltk.download("words")


def clean_text(text: str):
    """
    Function to clean text. Delete Punctuations and Stopwords.

    """
    punct = set(string.punctuation)
    stop_words = set(stopwords.words("english"))
    text = "".join([ch for ch in text if ch not in punct])
    text = " ".join([word for word in text.split() if word not in stop_words])
    text = TextBlob(text)
    text = " ".join([w.lemmatize() for w in text.words])
    return text


def emoji_to_text(emoji_str: str):
    """
    Function to convert emoji to text.

    """
    emoji_str = emoji.demojize(emoji_str)
    emoji_str = emoji_str.replace(":", " ")
    text = " ".join(emoji_str.split("_"))
    text = " ".join(text.split())
    return text


def is_statement_english(string: str):
    """
    Function to check if a statement is in English.
    """
    words = set(nltk.corpus.words.words())
    english_string = " ".join(
        w for w in nltk.wordpunct_tokenize(string) if w in words)
    if len(english_string.split()) >= 0.5 * len(string.split()):
        return True
    else:
        return False


def get_sentiments(text: str):
    """
    Function to get sentiment score. 

    """
    result = TextBlob(text)
    polarity = result.sentiment.polarity
    return polarity


# Load Data
f = open("Data/Raw/result.json")
tlg_data_json = json.load(f)

# Flatten the json file using normalize in pandas
tlg_message = pd.json_normalize(tlg_data_json["messages"])

tlg_message["text"] = tlg_message["text"].astype(str)
tlg_message["text"] = tlg_message["text"].str.lower()
tqdm.pandas()
tlg_message["text"] = tlg_message["text"].progress_map(
    lambda x: emoji_to_text(x))
tlg_message["text"] = tlg_message["text"].progress_map(lambda x: clean_text(x))
shib_doge_message = tlg_message[tlg_message["text"].str.contains("shib|doge")]

shib_doge_message = shib_doge_message[
    shib_doge_message["text"].progress_map(lambda x: is_statement_english(x))
]
shib_doge_message["Score"] = shib_doge_message["text"].progress_map(
    lambda x: get_sentiments(x)
)

shib_doge_message["date"] = pd.to_datetime(shib_doge_message["date"])
df = pd.DataFrame(
    shib_doge_message.groupby(
        [shib_doge_message["date"].dt.date], as_index=True, group_keys=True
    ).mean()["Score"]
)
df1 = pd.DataFrame(
    shib_doge_message.groupby(
        [shib_doge_message["date"].dt.date], as_index=True, group_keys=True
    ).count()["text"]
)
df.reset_index(inplace=True)
df1.reset_index(inplace=True)


fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
    go.Scatter(x=df["date"], y=df["Score"], name="Score"), secondary_y=True,
)
fig.add_trace(
    go.Bar(
        x=df1["date"],
        y=df1["text"],
        name="Count",
        text=round(df1["text"], 3),
        textposition="auto",
    )
)
fig.update_layout(
    title_text="Count of SHIB/DOGE related chats along with Sentiment Score."
)
fig.update_xaxes(title_text="Date")
fig.update_yaxes(title_text="<b>Count</b>", secondary_y=False)
fig.update_yaxes(title_text="<b>Score</b>", secondary_y=True)
fig.show()
