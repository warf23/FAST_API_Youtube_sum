from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
import validators
from fastapi.middleware.cors import CORSMiddleware
from youtube_transcript_api import YouTubeTranscriptApi
import re
import logging
import requests
from langchain.docstore.document import Document
# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

app = FastAPI() 

# Add CORS middlewa
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],  # Allows all origins
  allow_credentials=True,
  allow_methods=["*"],  # Allows all methods
  allow_headers=["*"],  # Allows all headers
)

# Define a request body model
class SummarizeRequest(BaseModel):
  open_api_key: str
  url: str
  language: str = "English"



prompt_template = PromptTemplate(template="""
Please provide a concise and informative summary in the language {language} of the content found at the following URL. The summary should be approximately 300 words and should highlight the main points, key arguments, and any significant conclusions or insights presented in the content. Ensure that the summary is clear and easy to understand for someone who has not accessed the original content.

URL Content:
{text}
""", 
input_variables=["text", "language"])

# Language options
language_codes = {'English': 'en', 'Arabic': 'ar', 'Spanish': 'es', 'French': 'fr', 'German': 'de', 
              'Italian': 'it', 'Portuguese': 'pt', 'Chinese': 'zh', 'Japanese': 'ja', 'Korean': 'ko'}

@app.get("/")   
def read_root():
  return {"Hello": "World"}

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
  openai_api_key = request.open_api_key
  url = request.url
  language = request.language

  # Validate input
  if not validators.url(url):
      raise HTTPException(status_code=400, detail="Invalid URL")

  if language not in language_codes:
      raise HTTPException(status_code=400, detail="Invalid language")

  try:
      # Initialize the language model
      llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")

      # Load the URL content
      content = load_url_content(url, language)

      # Split the content into chunks
      text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
      texts = text_splitter.split_text(content)
      combined_text = " ".join(texts)

      docs = [Document(page_content=t) for t in texts]

      chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt_template)
      summary = chain.run(input_documents=docs, language=language)

      return {"summary": summary}
  
  except Exception as e:
      raise HTTPException(status_code=500, detail=str(e))
