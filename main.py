import csv
import re
import random
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud


class SentimentAnalysis:

    def __init__(self):
        self.tweets = []
        self.tweetText = []

    def DownloadData(self):

        searchTerm = input("Enter Keyword/Tag to search about: ")
        NoOfTerms = int(input("Enter how many tweets to search: "))

        # load tweets dataset
        with open('tweets.csv', encoding='latin-1') as file:
            reader = list(csv.reader(file))
            random.shuffle(reader)

            for row in reader:
                tweet = row[5]
                if searchTerm.lower() in tweet.lower():
                    self.tweets.append(tweet)
                if len(self.tweets) >= NoOfTerms:
                    break

        print("Fetched", len(self.tweets), "tweets from dataset.\n")

        polarity = 0
        positive = 0
        wpositive = 0
        spositive = 0
        negative = 0
        wnegative = 0
        snegative = 0
        neutral = 0

        csvFile = open('result.csv', 'a', newline='', encoding='utf-8')
        csvWriter = csv.writer(csvFile)

        for tweet in self.tweets:

            cleaned = self.cleanTweet(tweet)
            self.tweetText.append(cleaned)

            analysis = TextBlob(cleaned)
            polarity += analysis.sentiment.polarity

            if analysis.sentiment.polarity == 0:
                neutral += 1
            elif 0 < analysis.sentiment.polarity <= 0.3:
                wpositive += 1
            elif 0.3 < analysis.sentiment.polarity <= 0.6:
                positive += 1
            elif 0.6 < analysis.sentiment.polarity <= 1:
                spositive += 1
            elif -0.3 < analysis.sentiment.polarity <= 0:
                wnegative += 1
            elif -0.6 < analysis.sentiment.polarity <= -0.3:
                negative += 1
            elif -1 < analysis.sentiment.polarity <= -0.6:
                snegative += 1

        csvWriter.writerow(self.tweetText)
        csvFile.close()

        # percentages
        positive = self.percentage(positive, len(self.tweets))
        wpositive = self.percentage(wpositive, len(self.tweets))
        spositive = self.percentage(spositive, len(self.tweets))
        negative = self.percentage(negative, len(self.tweets))
        wnegative = self.percentage(wnegative, len(self.tweets))
        snegative = self.percentage(snegative, len(self.tweets))
        neutral = self.percentage(neutral, len(self.tweets))

        polarity = polarity / len(self.tweets)

        print("General Report:")
        if polarity == 0:
            print("Neutral")
        elif polarity > 0:
            print("Positive")
        else:
            print("Negative")

        print("\nDetailed Report:")
        print(positive + "% Positive")
        print(wpositive + "% Weakly Positive")
        print(spositive + "% Strongly Positive")
        print(negative + "% Negative")
        print(wnegative + "% Weakly Negative")
        print(snegative + "% Strongly Negative")
        print(neutral + "% Neutral")

        # charts
        self.plotPieChart(positive, wpositive, spositive, negative, wnegative, snegative, neutral, searchTerm, len(self.tweets))
        self.plotBarGraph(positive, wpositive, spositive, negative, wnegative, snegative, neutral, searchTerm)
        self.generateWordCloud(searchTerm)

    def cleanTweet(self, tweet):
        return ' '.join(re.sub(r"(@[A-Za-z0-9_]+)|([^0-9A-Za-z \t])|(https?://\S+)", " ", tweet).split())

    def percentage(self, part, whole):
        return format(100 * float(part) / float(whole), '.2f')

    def plotPieChart(self, positive, wpositive, spositive, negative, wnegative, snegative, neutral, searchTerm, total):
        labels = [
            f'Positive [{positive}%]',
            f'Weakly Positive [{wpositive}%]',
            f'Strongly Positive [{spositive}%]',
            f'Neutral [{neutral}%]',
            f'Negative [{negative}%]',
            f'Weakly Negative [{wnegative}%]',
            f'Strongly Negative [{snegative}%]'
        ]

        sizes = list(map(float, [positive, wpositive, spositive, neutral, negative, wnegative, snegative]))
        colors = ['yellowgreen','lightgreen','darkgreen','gold','red','lightsalmon','darkred']

        plt.pie(sizes, colors=colors, startangle=90)
        plt.legend(labels, loc="best")
        plt.title(f'Sentiment Analysis on "{searchTerm}" ({total} Tweets)')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    def plotBarGraph(self, positive, wpositive, spositive, negative, wnegative, snegative, neutral, searchTerm):

        labels = ['Positive','Weak Positive','Strong Positive','Neutral','Negative','Weak Negative','Strong Negative']
        values = list(map(float, [positive, wpositive, spositive, neutral, negative, wnegative, snegative]))

        plt.figure(figsize=(10,6))
        plt.bar(labels, values)
        plt.xticks(rotation=30)
        plt.xlabel("Sentiment Type")
        plt.ylabel("Percentage")
        plt.title("Sentiment Distribution for " + searchTerm)
        plt.tight_layout()
        plt.show()

    def generateWordCloud(self, searchTerm):

        text = ' '.join(self.tweets)

        wordcloud = WordCloud(width=900, height=500, background_color='white').generate(text)

        plt.figure(figsize=(10,5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title("Common Words Used about " + searchTerm)
        plt.show()


if __name__ == "__main__":
    sa = SentimentAnalysis()
    sa.DownloadData()