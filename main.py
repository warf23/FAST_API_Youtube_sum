import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, HttpUrl
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
import validators
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key security
API_KEY = os.environ.get("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key")

def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate API key")
    return api_key

class SummarizeRequest(BaseModel):
    url: HttpUrl
    language: str = "English"

prompt_template = """
Please provide a concise and informative summary in the language {language} of the content found at the following URL. The summary should be approximately 300 words and should highlight the main points, key arguments, and any significant conclusions or insights presented in the content. Ensure that the summary is clear and easy to understand for someone who has not accessed the original content.

URL Content:
{text}

Summary:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text", "language"])

language_codes = {'English': 'en', 'Arabic': 'ar', 'Spanish': 'es', 'French': 'fr', 'German': 'de', 
                'Italian': 'it', 'Portuguese': 'pt', 'Chinese': 'zh', 'Japanese': 'ja', 'Korean': 'ko'}

@app.post("/api/summarize")
async def summarize(request: SummarizeRequest, api_key: str = Depends(get_api_key)):
    if request.language not in language_codes:
        raise HTTPException(status_code=400, detail="Invalid language")

    try:
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise HTTPException(status_code=500, detail="GROQ API key not configured")

        model = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

        if "youtube.com" in str(request.url):
            loader = YoutubeLoader.from_youtube_url(
                str(request.url),
                language=language_codes[request.language],
                add_video_info=True
            )
        else:
            loader = UnstructuredURLLoader(
                urls=[str(request.url)],
                ssl_verify=False,
                headers={"User-Agent": "Mozilla/5.0"}
            )

        docs = loader.load()

        # Combine the documents
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(docs)
        combined_text = " ".join([doc.page_content for doc in texts])

        # Create the chain
        chain = (
            {"text": RunnablePassthrough(), "language": lambda _: request.language}
            | prompt
            | model
            | StrOutputParser()
        )

        # Run the chain
        output = chain.invoke(combined_text)

        return {"summary": output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")

# Health check endpoint
@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}