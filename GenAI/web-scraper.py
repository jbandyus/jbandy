# web-scraper.py
# V .5 alpha
# Created by James Bandy
# See the attached LICENSE file for the MIT Open Source License 
# Let's get some great vector data from a public web site.
# Let's write the data to two local vector stores 1st, Chromadb and FAISS
# Use OpenAI embeddings and all-MiniLM-L6-v2 sentence transformer (called BERT)
# This is written to run from a windows 10/11 host with the software installed per the readme.txt
#
# If DATADIR is unset (-d directory) BERT defaults to ./BERT and OPENAI defaults to ./OPENAI 
# If you are using OpenAI you need to get a OpenAI API Key and share it with:
# set OPENAI_API_KEY=<KEY VALUE>
#
# Inputs
# run with -h to get the full usage information
# usage: web-scraper.py [-h] [-s STRIP] [-d DATADIR] [-v] {OPENAI,BERT} {ChromaDB,FAISS} sitemap_urls
# positional arguments:
#  {OPENAI,BERT}         Function to use for the embeddings
#  {ChromaDB,FAISS}      method to store the embeddings
#  sitemap_urls          list of XML sitemap URLS to gather URL info from, separated with commas
# Outputs
# Descriptive text, enable debugging with -v to get more info, embeds the sitemaps URL's text and
# stores the embedded text in the specified database

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
from itertools import batched

# For more info on the SeleniumURLLoader see
#  https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.url_selenium.SeleniumURLLoader.html
from langchain_community.document_loaders.url_selenium import SeleniumURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from selenium import webdriver
import requests
# currently unused
# import xmltodict
from lxml import etree
from time import perf_counter as pc
# new way from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.document_loaders import PyPDFLoader
from langchain_community.embeddings.sentence_transformer import (SentenceTransformerEmbeddings,)
# new way: from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
# new way from langchain_community.vectorstores.chroma import Chroma
from langchain_community.vectorstores.chroma import Chroma
# old way: from langchain.vectorstores import Chroma
# from langchain_community.vectorstores.faiss import FAISS
from langchain.vectorstores import FAISS
from openai import OpenAI

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
# output: text extracted from each URL, split into docs
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
    
    docs= text_split.split_documents(data)
    return docs

# currently I am using the langchain version of Chroma
# for now only BERT is supported to store in Chroma
def store_with_chroma(docs,data_dir):
    # create the open-source embedding function
    embedding_func = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # save to disk
    db2 = Chroma.from_documents(docs, embedding_func, persist_directory=data_dir)
    db2.persist()

    # extra functions for use later in another script
    #
    # load from disk
    #db3 = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    #docs = db3.similarity_search(query)
    #print(docs[0].page_content)
    # load it into Chroma
    #db = Chroma.from_documents(docs, embedding_function)
    # query it
    #query = "What is being done to protect the wild wolves of North America?"
    #docs = db.similarity_search(query)
    # print results
    #print(docs[0].page_content)

# 
def write_text_to_file(docs):
        
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

# This function can take OpenAPI Embeddings or SentenceTransformer
# OPENAI takes a valid API token as a environment variable 'set OPENAI_API_KEY= sk-XXXX'
# embedding function can be OPENAI or BERT
# defaults to BERT (free)
# For now model is hardcoded
def store_with_faiss(docs,embedding_func,data_dir):
    if (embedding_func=='OPENAI'):
        embedding_func = OpenAIEmbeddings()
    else: embedding_func = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # OpenAI limits me to 150k TPM, so we need to limit the rate bit
    final_embed= 0
    for doclist in (batched(docs,300)):
        if (DEBUG): print(f"Embedding batch of: {len(doclist)}")
        temp_embed= FAISS.from_documents(doclist, embedding_func)
        if (final_embed): final_embed.merge_from(temp_embed)
        else: final_embed= temp_embed
    final_embed.save_local(data_dir)
    # saving to a chroma file
    #vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory=".")
    #vectordb.persist()

# Main script
if __name__ == '__main__':

    # Move this to a def
    parser = argparse.ArgumentParser()
    parser.add_argument('embedding_func',choices=['OPENAI','BERT'], help="Function to use for the embeddings")
    parser.add_argument('embedding_db',choices=['ChromaDB','FAISS'], help="method to store the embeddings")
    parser.add_argument('sitemap_urls', help="list of XML sitemap URLS to gather URL info from, separated with commas")
    parser.add_argument('-s','--strip', nargs=1,help='string to replace reverse proxy data in each URL for example every: https://n9./ becomes https://')
    parser.add_argument('-d','--datadir', nargs=1,help='Directory to store the data files (local models only)')
    parser.add_argument("-v", "--verbosity", action="count", default=0, help="increase output verbosity")
    args = parser.parse_args()
    DEBUG=args.verbosity
    if (DEBUG): print(f"args: {args}")
    embedding_func= args.embedding_func
    embedding_db= args.embedding_db
    if (embedding_func=='OPENAI' and embedding_db=='ChromaDB'):
        print("ERROR: OPENAI is not supported with ChromaDB")
        exit(1)
    if (args.datadir):
        data_dir=args.datadir[0]
    elif (embedding_func=='BERT'):
            data_dir= "./BERT"
    elif (embedding_func=='OPENAI'):
        data_dir="./OPENAI"
    if (args.strip):
        PROXY_STRIP=args.strip[0]
    sitemap_urls= args.sitemap_urls.split(',')
    if (DEBUG): print(f"PROXY_STRIP={PROXY_STRIP}")
    if (DEBUG): print(f"DATA_DIR={data_dir}")
    if (DEBUG): print(f"DEBUG={DEBUG}")
    #exit(0)
    #
    # end move to def
    #

    # get all the URL's under each sitemap entry
    print(f"sitemap_urls={sitemap_urls}")
    urls= []
    for sitemap_url in sitemap_urls:
        urls.extend(get_urls_from_sitemap(sitemap_url))
    print(f"The number of sitemap urls are {len(urls)}")
    if (DEBUG): print(f"List of URLS extracted from the sitemaps {urls}")
    if (PROXY_STRIP): urls=[url.replace(PROXY_STRIP, 'https://') for url in urls]
    # nice for debugging less data
    # urls= urls[:8]
    if (DEBUG): print(f"The number of sitemap urls to extract text from are: {len(urls)}")

    # gets the text from the whole site, this can take a while for a large site
    full_sites_text= get_text_from_urls(urls)
    print(f"Text to record:{len(full_sites_text)}") 
    
    # potentially used for reloading, but for now unused
    #print("Writing to text file on SSD")
    #t0 = pc()
    #write_text_to_file(full_sites_text) 
    #print(f"Total time for writing text file to SSD: {pc()-t0} seconds")

    # lets create embeddings with the chosen function and store them in the chosen db
    if (embedding_db=='ChromaDB'):
        print("Writing to BERT/ChromaDB on SSD")
        t0 = pc()
        store_with_chroma(full_sites_text,data_dir) 
        print(f"Total time for writing BERT embeddings to ChromaDB on SSD: {pc()-t0} seconds")
    elif (embedding_db=='FAISS'):
        if (embedding_func=='OPENAI'):
          print("Writing to OpenAI/FAISS on SSD")
        else:
          print("Writing to BERT/FAISS on SSD")
        t0 = pc()
        store_with_faiss(full_sites_text,embedding_func,data_dir) 
        print(f"Total time for writing to FAISS db on SSD: {pc()-t0} seconds")