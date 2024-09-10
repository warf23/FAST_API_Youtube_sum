from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
import validators
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
import os
from langcorn import create_service



groq_api_key = os.environ.get("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model="llama3-groq-70b-8192-tool-use-preview")


# Define the prompt template
prompt_template = """
Please provide a concise and informative summary of the physics content from the following text. 

Focus on the following aspects:
1. Key physics concepts and principles introduced
2. Important formulas or equations, if any
3. Main theoretical explanations or arguments (if there are equations or formulas added)
4. Significant experimental results or observations, if mentioned
5. Practical applications or real-world examples of the concepts
6. Any historical context or important physicists mentioned
7. Add all the equations with bold format 

The summary should be approximately 250-300 words in {language}, written in clear and accessible language. Ensure that the summary highlights the most relevant information and provides a solid foundation for understanding the topic.

Text to summarize:
{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text", "language"])
# Language options
language_codes = {'English': 'en', 'Arabic': 'ar', 'Spanish': 'es', 'French': 'fr', 'German': 'de', 
              'Italian': 'it', 'Portuguese': 'pt', 'Chinese': 'zh', 'Japanese': 'ja', 'Korean': 'ko'}

#  Load the URL content
if "youtube.com" in url:
    loader = YoutubeLoader.from_youtube_url(url, language=language_codes[language], add_video_info=True)
else:
    loader = UnstructuredURLLoader(
        urls=[url], ssl_verify=False, headers={"User-Agent": "Mozilla/5.0"}
          )

docs = loader.load()

# Summarize the content
summarize_chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)
output = summarize_chain.run(input_documents=docs, language=language)

