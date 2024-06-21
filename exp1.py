import re
import os
import requests
import csv
import logging
from bs4 import BeautifulSoup
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockNewsSummarizer:
    def __init__(self, monitored_tickers, exclude_list=None):
        load_dotenv()
        self.monitored_tickers = monitored_tickers
        self.exclude_list = exclude_list if exclude_list else ['maps', 'policies', 'preferences', 'accounts', 'support']
        self.raw_urls = {}
        self.cleaned_urls = {}
        self.articles = {}
        self.summaries = {}
        self.sentiment_analyzer = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
        self.output_file = 'assetsummaries.csv'

        # Define the prompt template for summarization
        template = """You are a summarization assistant. Your task is to summarize the provided text in a concise and coherent manner. Please summarize the following text:
        =========
        {input}
        =========
        Summary:"""
        
        # Create a PromptTemplate object with the defined template
        self.prompt = PromptTemplate(template=template, input_variables=["input"])
        
        # Initialize the language model with specific parameters
        groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(api_key=groq_api_key, model_name="llama3-8b-8192")
        
        # Create the summarization sequence
        self.summarization_chain = RunnableSequence(self.prompt | self.llm)
    
    def search_for_stock_news_urls(self, ticker):
        try:
            search_url = f"https://www.google.com/search?q=yahoo+finance+{ticker}&tbm=nws"
            r = requests.get(search_url)
            r.raise_for_status()  # Raise exception for non-200 status codes
            soup = BeautifulSoup(r.text, 'html.parser')
            atags = soup.find_all('a')
            hrefs = [link['href'] for link in atags]
            return hrefs
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching URLs for ticker {ticker}: {str(e)}")
            return []
    
    def strip_unwanted_urls(self, urls):
        val = []
        for url in urls:
            if 'https://' in url and not any(exclude_word in url for exclude_word in self.exclude_list):
                res = re.findall(r'(https?://\S+)', url)[0].split('&')[0]
                val.append(res)
        return list(set(val))
    
    def scrape_and_process(self, urls):
        articles = []
        for url in urls:
            try:
                r = requests.get(url)
                r.raise_for_status()  # Raise exception for non-200 status codes
                soup = BeautifulSoup(r.text, 'html.parser')
                paragraphs = soup.find_all('p')
                text = ' '.join([paragraph.text for paragraph in paragraphs])
                words = text.split(' ')[:350]
                article = ' '.join(words)
                articles.append(article)
            except requests.exceptions.RequestException as e:
                logger.error(f"Error scraping article from URL {url}: {str(e)}")
        return articles
    
    def summarize_articles(self, articles):
        summaries = []
        for article in articles:
            try:
                response = self.summarization_chain.invoke({"input": article})
                summary = response.get('output', '')  # Safely get the 'output' key
                summaries.append(summary)
            except Exception as e:
                logger.error(f"Error summarizing article: {str(e)}")
        return summaries
    
    def analyze_sentiment(self, summaries):
        return self.sentiment_analyzer(summaries)
    
    def check_existing_urls(self):
        existing_urls = set()
        try:
            if os.path.exists(self.output_file):
                with open(self.output_file, mode='r', newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    for row in reader:
                        existing_urls.add(row[-1])  # Last column contains URLs
        except Exception as e:
            logger.error(f"Error checking existing URLs in {self.output_file}: {str(e)}")
        return existing_urls
    
    def filter_new_urls(self, urls, existing_urls):
        new_urls = []
        for url in urls:
            if url not in existing_urls:
                new_urls.append(url)
        return new_urls
    
    def run_analysis(self):
        final_output = [['Ticker', 'Summary', 'Label', 'Confidence', 'URL']]
        existing_urls = self.check_existing_urls()
        
        for ticker in self.monitored_tickers:
            try:
                logger.info(f"Fetching URLs for ticker {ticker}...")
                self.raw_urls[ticker] = self.search_for_stock_news_urls(ticker)
                self.cleaned_urls[ticker] = self.strip_unwanted_urls(self.raw_urls[ticker])
                
                if not self.cleaned_urls[ticker]:
                    logger.warning(f"No new URLs found for ticker {ticker}. Skipping...")
                    continue
                
                if existing_urls:
                    self.cleaned_urls[ticker] = self.filter_new_urls(self.cleaned_urls[ticker], existing_urls)
                
                logger.info(f"Scraping and summarizing articles for ticker {ticker}...")
                self.articles[ticker] = self.scrape_and_process(self.cleaned_urls[ticker])
                self.summaries[ticker] = self.summarize_articles(self.articles[ticker])
                
                logger.info(f"Analyzing sentiment for ticker {ticker} summaries...")
                sentiment_scores = self.analyze_sentiment(self.summaries[ticker])
                
                logger.info(f"Creating output for ticker {ticker}...")
                for i in range(len(self.summaries[ticker])):
                    final_output.append([
                        ticker,
                        self.summaries[ticker][i],
                        sentiment_scores[i]['label'],
                        sentiment_scores[i]['score'],
                        self.cleaned_urls[ticker][i]
                    ])
            except Exception as e:
                logger.error(f"Error processing ticker {ticker}: {str(e)}")
        
        # Write data to CSV
        try:
            with open(self.output_file, mode='a', newline='', encoding='utf-8') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for row in final_output[1:]:  # Skip header in final_output
                    csv_writer.writerow(row)
            logger.info(f"Data written to {self.output_file}")
        except Exception as e:
            logger.error(f"Error writing data to {self.output_file}: {str(e)}")
        
        return final_output

# Example Usage:
if __name__ == "__main__":
    monitored_tickers = ['OKLO','AAPL','NVDA']  # List of tickers to monitor
    summarizer = StockNewsSummarizer(monitored_tickers)
    final_output = summarizer.run_analysis()
    print(final_output)
