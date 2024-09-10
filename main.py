from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
import validators
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
import os
from langcorn import create_service
# FastAPI app instance
app = FastAPI()

# Add CORS middleware
app.add_middleware(
  CORSMiddleware,
  allow_origins=["https://your-frontend-domain.com"],  # Replace with your actual frontend domain
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

# API key security
API_KEY = os.environ.get("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key")

def get_api_key(api_key: str = Depends(api_key_header)):
  if api_key == API_KEY:
      return api_key
  raise HTTPException(status_code=403, detail="Could not validate credentials")

# Define a request body model
class SummarizeRequest(BaseModel):
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
async def summarize(request: SummarizeRequest, api_key: str = Depends(get_api_key)):
  url = request.url
  language = request.language

  # Validate input
  if not validators.url(url):
      raise HTTPException(status_code=400, detail="Invalid URL")

  if language not in language_codes:
      raise HTTPException(status_code=400, detail="Invalid language")

  try:
      # Initialize the language model
      groq_api_key = os.environ.get("GROQ_API_KEY")
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
  

# Create a Langcorn service
service = create_service(app)

# Include the Langcorn router
app.include_router(service.router)