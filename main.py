import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
import validators
from fastapi.middleware.cors import CORSMiddleware

# FastAPI app instance
app = FastAPI()

# Add CORS middleware
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],  # Add your React app's URL
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

# Define API key security scheme
API_KEY_NAME = "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"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Depends(api_key_header)):
  if api_key_header == os.environ.get("API_KEY"):
      return api_key_header
  raise HTTPException(status_code=403, detail="Could not validate credentials")

# Define a request body model
class SummarizeRequest(BaseModel):
  url: str
  language: str = "English"

# Define the prompt template
prompt_template = """
Please provide a concise and informative summary in the language {language} of the content found at the following URL. The summary should be approximately 300 words and should highlight the main points, key arguments, and any significant conclusions or insights presented in the content. Ensure that the summary is clear and easy to understand for someone who has not accessed the original content.

URL Content:
{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text", "language"])

# Language options
language_codes = {'English': 'en', 'Arabic': 'ar', 'Spanish': 'es', 'French': 'fr', 'German': 'de', 
                'Italian': 'it', 'Portuguese': 'pt', 'Chinese': 'zh', 'Japanese': 'ja', 'Korean': 'ko'}

@app.post("/summarize")
async def summarize(request: SummarizeRequest, api_key: str = Depends(get_api_key)):
  groq_api_key = os.environ.get("GROQ_API_KEY")
  if not groq_api_key:
      raise HTTPException(status_code=500, detail="GROQ API key not configured")

  url = request.url
  language = request.language

  # Validate input
  if not validators.url(url):
      raise HTTPException(status_code=400, detail="Invalid URL")

  if language not in language_codes:
      raise HTTPException(status_code=400, detail="Invalid language")

  try:
      # Initialize the language model
      llm = ChatGroq(groq_api_key=groq_api_key, model="llama3-groq-70b-8192-tool-use-preview")

      # Load the URL content
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

      return {"summary": output}

  except Exception as e:
      raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")