import re
import os
import requests
import csv
import logging
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load the Groq API key from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    logger.error("GROQ_API_KEY environment variable not found")
    exit(1)

# Initialize the Groq client
client = Groq(api_key=groq_api_key)
def summarize_articles(article):
    summaries = []
    try:
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {
                    "role":"user",
                    "content":"You are helpful fintech assitant"
                },
                {
                    "role": "user",
                    "content": f"Summarize the following news article accurately and impartially. Ensure the summary is devoid of any bias or personal comments:\n{article}. Avoid introductory phrases such as 'here is your summary...'"

                }
            ],
            temperature=0,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )
        
        summary = ""
        for chunk in completion:
            summary += chunk.choices[0].delta.content or ""
        
        # Process the summary text        
        summary = summary.replace('\n\n', ' ')
        
        summaries.append(summary)
    except Exception as e:
        logger.error(f"Error summarizing article: {str(e)}")
    return summaries

# Example article to summarize
article = """SANTA CLARA, Calif., June 20, 2024--(BUSINESS WIRE)--Oklo Inc. (NYSE: OKLO) ("Oklo" or the "Company"), a fast fission clean power technology and nuclear fuel recycling company, earlier today filed a registration statement on Form S-1 (the "Registration Statement") with the U.S. Securities and Exchange Commission ("SEC").
The Registration Statement was filed as a standard administrative matter to initiate the process of registering 62,440,080 shares of common stock of the Company ("Common Stock") that are currently unregistered for resale. The Registration Statement was required to be filed within 30 days of close of the Company’s initial business combination and all unregistered shares of Common Stock covered by the Registration Statement were previously detailed in the registration statement on Form S-4 (the "Form S-4") initially filed with the SEC on September 27, 2023, in connection with the Company’s initial business combination. The Company has not registered any new shares for primary issuance to the public through the Registration Statement.
The Registration Statement does not create freely tradable shares today. All shares of Common Stock covered by the Registration Statement are subject to either contractual lock-up restrictions or performance vesting requirements, as previously disclosed in the Form S-4. In particular, the shares of Common Stock currently held by certain members of Oklo management and AltC Acquisition Sponsor LLC are subject to long-term lock-up restriction that expires in several tranches over a three-year period from the closing of the business combination, subject to earlier expiration upon the satisfaction of certain conditions related to the closing price of the Common Stock.
While the Registration Statement has been filed with the SEC, it has not yet become effective, and the information contained therein is subject to change.
This press release shall not constitute an offer to sell or the solicitation of any offer to buy, nor shall there be any sale of these securities in any state or jurisdiction in which such offer, solicitation or sale would be unlawful prior to registration or qualification under the securities laws of any such state or jurisdiction.
About Oklo Inc.
Oklo Inc. is developing fast fission power plants to provide clean, reliable, and affordable energy at scale. Oklo received a site use permit from the U.S. Department of Energy, was awarded fuel material from Idaho National Laboratory, submitted the first advanced fission custom combined license application to the Nuclear Regulatory Commission, and is developing advanced fuel recycling technologies in collaboration with the U.S. Department of Energy and U.S. National Laboratories.
Forward-Looking Statements
This press release includes statements that express Oklo’s opinions, expectations, objectives, beliefs, plans, intentions, strategies, assumptions, forecasts or projections regarding future events or future results and therefore are, or may be deemed to be, "forward-looking statements" within the meaning of the Private Securities Litigation Reform Act of 1995. The words "anticipate," "believe," "continue," "could," "estimate," "expect," "intends," "may," "might," "plan," "possible," "potential," "predict," "project," "should," "would" or, in each case, their negative or other variations or comparable terminology, and similar expressions may identify forward-looking statements, but the absence of these words does not mean that a statement is not forward-looking. These forward-looking statements include all matters that are not historical facts. They appear in a number of places throughout this press release and include statements regarding our intentions, beliefs or current expectations concerning, among other things, the timing for effectiveness of the Registration Statement, results of operations, financial condition, liquidity, prospects, growth, strategies and the markets in which Oklo operates. Such forward-looking statements are based on information available as of the date of this press release, and current expectations, forecasts and assumptions, and involve a number of judgments, risks and uncertainties.
As a result of a number of known and unknown risks and uncertainties, the actual results or performance of Oklo may be materially different from those expressed or implied by these forward-looking statements. The following important risk factors could affect Oklo’s future results and cause those results or other outcomes to differ materially from those expressed or implied in the forward-looking statements: factors affecting the timing and effectiveness of the registration statement, risks related to the deployment of Oklo’s powerhouses; the risk that Oklo is pursuing an emerging market, with no commercial project operating, regulatory uncertainties; the potential need for financing to construct plants, market, financial, political and legal conditions; the effects of competition; changes in applicable laws or regulations; and the outcome of any government and regulatory proceedings and investigations and inquiries.
The foregoing list of factors is not exhaustive. You should carefully consider the foregoing factors and the other risks and uncertainties of the other documents filed by Oklo from time to time with the U.S. Securities and Exchange Commission. The forward-looking statements contained in this press release and in any document incorporated by reference are based on current expectations and beliefs concerning future developments and their potential effects on Oklo. There can be no assurance that future developments affecting Oklo will be those that Oklo has anticipated. Oklo undertakes no obligation to update or revise any forward-looking statements, whether as a result of new information, future events or otherwise, except as may be required under applicable securities laws.
View source version on businesswire.com: https://www.businesswire.com/news/home/20240620000077/en/
Contacts
Media and Investor Contact for Oklo:
Bonita Chester, Director of Communications and Media, at media@oklo.com and investors@oklo.com
"""

# Summarize the article
summaries = summarize_articles(article)
print(summaries)
