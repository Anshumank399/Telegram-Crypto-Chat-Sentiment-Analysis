# Telegram-Crypto-Chat-Sentiment-Analysis
Analyse sentiment of telegram group chat.

## How to run the code?  
1. Download the source code from the [git repo](https://github.com/Anshumank399/Telegram-Crypto-Chat-Sentiment-Analysis). 
2. Install requirements.txt using command: 
```
pip install -r requirements.txt
```
3. Either use the _Script/sentiment.py_ file or _Sentiment Analysis Telegram Chat.ipynb_ notebook.

## More about the method used.
1. Telegram chats of a crypto group were downloaded from telegram and exported in json format. 
2. The chat is converted into a pandas table by flattening the json. 
3. As the text contains a lot of emojis (chat data), we eliminated the emojis by text so that we continue to capture their emotions. 
4. Clean the text using some basic method of changing text to lower case, applying lemmatization, removing the punctuations and stop words.
5. Filter the chats that ccontained "Shib" or "Doge".
6. Filter the texts that were not in english. (Maching bag of words and having a threshold for the sentence.)
7. Apply sentiment analysis for each line and get the score. 
8. Plot the sentiment score and the count of chats each day. 


## Findings !
1. The chats about the doge and shib crypto coins was really high 8th and 10th May, 2021.
2. High score on the graph represents +ve sentiment. 
3. 2nd May had high +ve chats on the the given crypto coins but the number of chats were much fewer than normal.
4. Chat sentiment and Count Graph ![Analyse](https://github.com/Anshumank399/Telegram-Crypto-Chat-Sentiment-Analysis/blob/main/Sentiment%20and%20Chat%20Count%20Plot.png) 

## Future work or Things that could be improved or New Methods.
1. Try filtering the non english data using TFIDF on stop words.
2. Built a personalized sentiment scoring model for Crypto chats. Example give +ve weights to BUY and -ve to SELL.
