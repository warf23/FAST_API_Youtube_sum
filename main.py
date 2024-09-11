   from fastapi import FastAPI, HTTPException
   from pydantic import BaseModel
   from langchain_core.prompts import PromptTemplate
   from langchain_groq import ChatGroq
   from langchain.chains import create_extraction_chain
   from langchain.text_splitter import CharacterTextSplitter
   from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
   from langchain_core.output_parsers import StrOutputParser
   from langchain_core.runnables import RunnablePassthrough
   import validators
   from fastapi.middleware.cors import CORSMiddleware

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
           model = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile")

           # Load the URL content
           if "youtube.com" in url:
               loader = YoutubeLoader.from_youtube_url(url, language=language_codes[language], add_video_info=True)
           else:
               loader = UnstructuredURLLoader(
                   urls=[url], ssl_verify=False, headers={"User-Agent": "Mozilla/5.0"}
               )

           docs = loader.load()

           # Combine the documents
           text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
           texts = text_splitter.split_documents(docs)
           combined_text = " ".join([doc.page_content for doc in texts])

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
           raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")