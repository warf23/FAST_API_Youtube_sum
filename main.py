from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
import validators
from fastapi.middleware.cors import CORSMiddleware
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app instance
app = FastAPI()

# Add CORS middleware
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],  # Allow all origins for now, you can restrict this later
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

@app.post("/summarize")
async def summarize(request: SummarizeRequest):
  logger.info(f"Received summarization request for URL: {request.url}")
  
  groq_api_key = request.groq_api_key
  url = request.url
  language = request.language

  # Validate input
  if not validators.url(url):
      logger.error(f"Invalid URL provided: {url}")
      raise HTTPException(status_code=400, detail="Invalid URL")

  if language not in language_codes:
      logger.error(f"Invalid language provided: {language}")
      raise HTTPException(status_code=400, detail="Invalid language")

  try:
      # Initialize the language model
      logger.info("Initializing ChatGroq model")
      llm = ChatGroq(groq_api_key=groq_api_key, model="llama3-groq-70b-8192-tool-use-preview")

      # Load the URL content
      logger.info(f"Loading content from URL: {url}")
      if "youtube.com" in url:
          loader = YoutubeLoader.from_youtube_url(url, language=language_codes[language], add_video_info=True)
      else:
          loader = UnstructuredURLLoader(
              urls=[url], ssl_verify=False, headers={"User-Agent": "Mozilla/5.0"}
          )

      docs = loader.load()
      logger.info(f"Content loaded successfully. Document count: {len(docs)}")

      # Summarize the content
      logger.info("Starting summarization process")
      summarize_chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)
      output = summarize_chain.run(input_documents=docs, language=language)
      logger.info("Summarization completed successfully")

      return {"summary": output}

  except Exception as e:
      logger.error(f"An error occurred during summarization: {str(e)}", exc_info=True)
      raise HTTPException(status_code=500, detail=f"An error occurred during summarization: {str(e)}")

# Root route
@app.get("/")
async def root():
  logger.info("Root endpoint accessed")
  return {"message": "Welcome to the Physics Content Summarizer API"}

# Health check route
@app.get("/health")
async def health_check():
  logger.info("Health check endpoint accessed")
  return {"status": "healthy"}

# Error handling for unhandled exceptions
@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
  logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
  return {"detail": "An unexpected error occurred. Please try again later."}

# Print environment variables (for debugging purposes)
@app.on_event("startup")
async def startup_event():
  logger.info("Starting up the application")
  logger.info("Environment variables:")
  for key, value in os.environ.items():
      logger.info(f"{key}: {value}")

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)