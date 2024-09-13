from fastapi import FastAPI, HTTPException
from pydantic import BaseModel 
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import validators
from fastapi.middleware.cors import CORSMiddleware
import requests
from youtube_transcript_api import YouTubeTranscriptApi
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app instance
app = FastAPI()

# Add CORS middleware
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

# Define a request body model
class SummarizeRequest(BaseModel):
  groq_api_key: str
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

def get_youtube_transcript(video_id, language='en'):
  try:
      transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
      return " ".join([entry['text'] for entry in transcript])
  except Exception as e:
      logger.error(f"Error fetching YouTube transcript: {str(e)}")
      return None

def get_youtube_video_id(url):
  video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
  if video_id_match:
      return video_id_match.group(1)
  return None

def load_url_content(url, language):
  try:
      if "youtube.com" in url or "youtu.be" in url:
          video_id = get_youtube_video_id(url)
          if video_id:
              content = get_youtube_transcript(video_id, language_codes.get(language, 'en'))
              if content:
                  return content
          
      response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
      response.raise_for_status()
      return response.text
  except Exception as e:
      logger.error(f"Error loading URL content: {str(e)}")
      raise HTTPException(status_code=500, detail=f"Error loading URL content: {str(e)}")

@app.post("/summarize")
async def summarize(request: SummarizeRequest):
  groq_api_key = request.groq_api_key
  url = request.url
  language = request.language

  # Validate input
  if not validators.url(url):
      raise HTTPException(status_code=400, detail="Invalid URL")

  if language not in language_codes:
      raise HTTPException(status_code=400, detail="Invalid language")

  try:
      # Initialize the language model
      model = ChatGroq(groq_api_key=groq_api_key, model_name="gemma-7b-it")

      # Load the URL content
      content = load_url_content(url, language)

      # Split the content
      text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
      texts = text_splitter.split_text(content)
      combined_text = " ".join(texts)

      # Create the chain
      chain = (
          {"text": RunnablePassthrough(), "language": lambda _: language}
          | prompt
          | model
          | StrOutputParser()
      )

      # Run the chain
      output = chain.invoke(combined_text)

      return {"summary": output}

  except Exception as e:
      logger.error(f"Error occurred: {str(e)}")
      raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")

# Add a simple health check endpoint
@app.get("/health")
async def health_check():
  return {"status": "ok"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)