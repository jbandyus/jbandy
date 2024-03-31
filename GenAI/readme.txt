This project will test a number of vector store databases for overall performance and measure their impact on actual Generative AI query
performance. We will scape a websites sitemap (XML only for now) to get content to build out the vector database. I also plan to add a PDF
scraper. 

Scripts:
web-scraper.py: 1st script, to get the data for the vector database

Requirements (for Windows 11):
# for now these are not necessary since I am testing with Chroma and FAISS 1st
# Install the AWS CLI from aws.amazon.com/cli
# Setup IAM or Identity Center access to your AWS account where you have access to Bedrock
# Enable the models in bedrock
Install Visual Studio Build Tools - https://visualstudio.microsoft.com/visual-cpp-build-tools/
Install python (latest)
python -m pip install --upgrade pip
pip install numpy
pip install botocore
pip install boto3
pip install langchain
pip install requests
pip install selenium
pip install webdriver_manager
pip install unstructured
# for a GUI later
#pip install streamlit