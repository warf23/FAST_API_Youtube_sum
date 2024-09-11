from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document
import validators
from fastapi.middleware.cors import CORSMiddleware
import logging
import time
import traceback
from io import StringIO
from youtube_transcript_api import YouTubeTranscriptApi

# Custom logging handler
class MemoryHandler(logging.Handler):
  def __init__(self):
      super().__init__()
      self.logs = StringIO()

  def emit(self, record):
      log_entry = self.format(record)
      self.logs.write(f"{log_entry}\n")

  def get_logs(self):
      return self.logs.getvalue()

  def clear(self):
      self.logs.truncate(0)
      self.logs.seek(0)

# FastAPI app instance
app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

memory_handler = MemoryHandler()
memory_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(memory_handler)

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

def extract_video_id(url):
  if "youtube.com" in url:
      return url.split("v=")[-1].split("&")[0]
  elif "youtu.be" in url:
      return url.split("/")[-1]
  else:
      raise ValueError("Not a valid YouTube URL")

@app.post("/summarize")
async def summarize(request: SummarizeRequest):
  memory_handler.clear()
  start_time = time.time()
  groq_api_key = request.groq_api_key
  url = request.url
  language = request.language

  logger.info(f"Received request - URL: {url}, Language: {language}")

  if not url:
      logger.error("URL is missing")
      raise HTTPException(status_code=400, detail="URL is missing")

  if not validators.url(url):
      logger.error(f"Invalid URL: {url}")
      raise HTTPException(status_code=400, detail="Invalid URL")

  if language not in language_codes:
      logger.error(f"Invalid language: {language}")
      raise HTTPException(status_code=400, detail="Invalid language")

  try:
      logger.info("Initializing language model")
      model = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile")

      logger.info(f"Loading content from URL: {url}")
      if "youtube.com" in url or "youtu.be" in url:
          logger.info("Detected YouTube URL, using YouTube Transcript API")
          try:
              video_id = extract_video_id(url)
              logger.info(f"Extracted video ID: {video_id}")
              transcript = YouTubeTranscriptApi.get_transcript(video_id)
              logger.info("Successfully fetched transcript")
              text = " ".join([entry['text'] for entry in transcript])
              docs = [Document(page_content=text)]
              logger.info(f"Created document with transcript. Text length: {len(text)}")
          except Exception as yt_error:
              logger.error(f"Error fetching YouTube transcript: {str(yt_error)}")
              logger.error(f"YouTube error traceback: {traceback.format_exc()}")
              raise HTTPException(status_code=500, detail=f"Error fetching YouTube transcript: {str(yt_error)}")
      else:
          logger.info("Using UnstructuredURLLoader")
          try:
              loader = UnstructuredURLLoader(
                  urls=[url], ssl_verify=False, headers={"User-Agent": "Mozilla/5.0"}
              )
              docs = loader.load()
              logger.info(f"Successfully loaded URL content. Number of documents: {len(docs)}")
          except Exception as url_error:
              logger.error(f"Error loading URL content: {str(url_error)}")
              logger.error(f"URL error traceback: {traceback.format_exc()}")
              raise HTTPException(status_code=500, detail=f"Error loading URL content: {str(url_error)}")

      if not docs:
          logger.error("No content loaded from the URL")
          raise HTTPException(status_code=500, detail="No content could be loaded from the provided URL")

      logger.info("Processing loaded content")
      text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
      texts = text_splitter.split_documents(docs)
      combined_text = " ".join([doc.page_content for doc in texts])
      logger.info(f"Combined text length: {len(combined_text)}")

      if len(combined_text) < 10:  # Arbitrary small number to check if text is essentially empty
          logger.error("Processed text is too short or empty")
          raise HTTPException(status_code=500, detail="Processed text is too short or empty")

      logger.info("Creating and running the chain")
      chain = (
          {"text": RunnablePassthrough(), "language": lambda _: language}
          | prompt
          | model
          | StrOutputParser()
      )

      output = chain.invoke(combined_text)
      logger.info(f"Chain output length: {len(output)}")

      if len(output) < 10:  # Another arbitrary check for very short outputs
          logger.error("Generated summary is too short")
          raise HTTPException(status_code=500, detail="Generated summary is too short")

      execution_time = time.time() - start_time
      logger.info(f"Total execution time: {execution_time:.2f} seconds")

      return {
          "summary": output, 
          "execution_time": f"{execution_time:.2f} seconds",
          "logs": memory_handler.get_logs()  # Include logs in the response
      }

  except Exception as e:
      logger.error(f"Unexpected error occurred: {str(e)}")
      logger.error(f"Full traceback: {traceback.format_exc()}")
      return {
          "error": f"An unexpected error occurred: {str(e)}",
          "logs": memory_handler.get_logs()  # Include logs even when there's an error
      }
