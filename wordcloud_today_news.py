## fetch news of today's and create wordcloud of the keywords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def today_news():
  toPublishedDate = datetime.datetime.today().strftime('%Y-%m-%d')
  fromPublishedDate = (datetime.datetime.today() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')

  url = "https://contextualwebsearch-websearch-v1.p.rapidapi.com/api/Search/NewsSearchAPI"
  querystring = {"autoCorrect":"false","pageNumber":"1","pageSize":"30","q":"Bitcoin","safeSearch":"false", "fromPublishedDate":fromPublishedDate, "toPublishedDate":toPublishedDate}
  headers = {
      'x-rapidapi-host': "contextualwebsearch-websearch-v1.p.rapidapi.com",
      'x-rapidapi-key': "6e64d32139msh66d110fec2ee5d6p1b2f8ajsn506563d39c4e"
      }
  response = requests.request("GET", url, headers=headers, params=querystring)
  response_json = response.json()['value']

  descriptions = []
  for i in range(len(response_json)):
    description = response_json[i]['description']
    if description:
      non_symbols = re.sub(r'[^\w]', ' ', description) # remove symbols
      non_digits = re.sub(r'\d+', ' ', non_symbols) # remove digits
      one_space = re.sub(' +', ' ', non_digits) # remove extra whitespaces
      description_lower = one_space.lower() # transform characters into lower-case
      description_words = ' '.join(w for w in description_lower.split() if len(w)>1) # remove single characters
      descriptions.append(description_words) # save descriptions context

  Stopwords = set(stopwords.words('english'))
  Stemmer = nltk.stem.SnowballStemmer('english')
  text = ''
  for sent in descriptions:
    for word in sent.split():
      if word not in Stopwords:
        text = text + ' ' + Stemmer.stem(word)

  wordcloud = WordCloud(background_color='white').generate(text)
  ax = plt.subplot(1,1,1)
  ax.imshow(wordcloud, interpolation='bilinear', aspect='auto')
  ax.axis('off')
today_news()
