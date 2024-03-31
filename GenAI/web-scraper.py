# web-scraper.py
# V .2 alpha
# Created by James Bandy for James Bandy
# all rights reserved, working on the right license to use 
# Let's get some great vector data from a public web site.
# This is written to run from a windows 10/11 host with the software installed per the readme.txt
#
#
# Inputs

# Outputs

# Statics
PROFILE_NAME='default' # currently unused
REGION_NAME='us-west-2' # currently unused
BROWSER='mozilla' # for now firefox mozilla is the only supported browser

# Includes
import sys
import argparse
import botocore
import boto3
import numpy
import langchain
from langchain_community.document_loaders import PyPDFLoader
# For more info on the SeleniumURLLoader see
#  https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.url_selenium.SeleniumURLLoader.html
from langchain_community.document_loaders import SeleniumURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from selenium import webdriver
import requests
# currently unused
# import xmltodict
from lxml import etree

# completely unused with the langchain Selenium
# I will be using this later...
if (BROWSER=='NONE'):
    # for reference see https://www.selenium.dev/documentation/webdriver/browsers/firefox/
    from selenium.webdriver.firefox.options import Options
    from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
    from selenium.webdriver.firefox.service import Service 
    #from webdriver_manager.firefox import GeckoDriverManager
    
    service= Service("C:\\Program Files\\webdriver\\geckodriver\\geckodriver.exe")
    #service = webdriver.FirefoxService(log_output=log_path, service_args=['--log', 'debug'])
    options=Options()
    firefox_profile = FirefoxProfile()
    firefox_profile.set_preference("javascript.enabled", True)
    #firefox_profile.set_preference("javascript.options.showInConsole", False)
    firefox_profile.set_preference("pref.privacy.disable_button.tracking_protection_exceptions", True)
    firefox_profile.set_preference("pref.privacy.trackingprotection.enabled", False)
    userAgent= 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0'
    options.set_preference("general.useragent.override", userAgent)
    options.profile = firefox_profile
    options.add_argument("-headless")
    driver= webdriver.Firefox(service=service,options=options)
    #driver.get("https://www.selenium.dev")
    #page = driver.current_url
    #print(page)

# Definitions

# Also I plan to build out a way to read local files
# specifically PDF's to build a set of knowledge around a comanies products for example

# I played around with a number of ways to get this data and settled on lxml etree. If the sitemap is not XML, this will fail.
# I plan to make this more robust in the future, but for now I have only ran into xml sitemap files.
def get_urls_from_sitemap(sitemap):
    xml_dict = {}
    try:
        r= requests.get(sitemap)
        if r.status_code != 200:
            raise Exception(f"Failed to get the sitemap using: {sitemap}")
    except Exception as e:
        print(f"ERROR: {e}")
        exit(1)
    root = etree.fromstring(r.content)
    for sitemap in root:
        children = sitemap.getchildren()
        xml_dict[children[0].text] = children[0].text
    # for debug
    # print(xml_dict)
    #for url in xml_dict:
    #    print(url)
    return xml_dict

# gets text from the list of URL's you send. returns raw text chunked aiming for 1k tokens.
# input: URLS - http(s) strings in a list
# output: text extracted from each URL
def get_text_from_urls(urls):
    try:
        # if you need to setup a different profile, check out the mozilla command line wiki
        # https://wiki.mozilla.org/Firefox/CommandLineOptions 
        loader= SeleniumURLLoader(urls,continue_on_failure=1,browser='firefox',headless=1)
        data= loader.load()
    except Exception as e:
        print(f"ERROR: {e}")
        exit(1)
    # ADD code to split by tokens, I wanted to store the text and then test vector store performance
    #   Unfortunately splitting by tokens can lock you in with the tokenizer model needing to match
    #   the embeddings. I plan to do some testing on the OpenAI tokenizer 
    # roughly 4 characters is the average token size, so I am aiming for roughly 1k tokens
    # a chunk overlap of 5% seems reasonable
    text_split= RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    
    texts= text_split.split_documents(data)
    return texts

# Main script
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("sitemap_urls", help="list of XML sitemap URLS to gather URL info from, separated with commas")
    parser.add_argument('-s','--strip', nargs=1,help='string to replace reverse proxy data in each URL for examplei every: https://n9./ becomes https://')
    parser.add_argument("-v", "--verbosity", action="count", default=0, help="increase output verbosity")
    args = parser.parse_args()
    PROXY_STRIP=args.strip[0]
    DEBUG=args.verbosity
    sitemap_urls= args.sitemap_urls.split(',')
    print(f"PROXY_STRIP={PROXY_STRIP}")
    print(f"DEBUG={DEBUG}")
    print(f"sitemap_urls={sitemap_urls}")
    urls= []
    for sitemap_url in sitemap_urls:
        urls.extend(get_urls_from_sitemap(sitemap_url))
    print(f"The number of sitemap urls are {len(urls)}")
    if (DEBUG): print(f"List of URLS extracted from the sitemaps {urls}")
    if (PROXY_STRIP): urls=[url.replace(PROXY_STRIP, 'https://') for url in urls]
    # nice for debugging less data
    urls= urls[:2]
    if (DEBUG): print(f"The number of sitemap urls to test are {len(urls)}")
    full_sites_text= get_text_from_urls(urls)
    # since this is unknown text lets make sure we can write it
    #full_sites_text= [x.encode('utf-8') for x in full_sites_text]
    print(f"recorded text {len(full_sites_text)}") 
    text_fh= open('sites.txt', 'a')
    for doc in full_sites_text:
        try:
            # for extreme debug
            # print(doc.page_content)
            encoded= doc.page_content.encode('utf-8')
            print(encoded, file=text_fh)
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    text_fh.close()
