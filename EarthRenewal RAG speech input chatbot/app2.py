from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores.base import VectorStoreRetriever
from dotenv import load_dotenv
import speech_recognition as sr
import pyttsx3

load_dotenv()


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text():
    text = ""
    pdf_reader = PdfReader(r"C:\Users\Nizam\Desktop\Earthrenewal chatbot\Restoration_of_Degraded_Agricultural_Lan (1).pdf")
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    
    
    return vector_store.as_retriever()

def get_conversational_chain(retriever: VectorStoreRetriever):
    prompt_template = """Answer the question as detailed as possible using the provided context.
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0.3, max_tokens=150)
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    
    chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",  
        retriever=retriever,  
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}  
    )
    
    return chain



def get_audio_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please ask your question:")
        audio = recognizer.listen(source)
        
        try:
            
            user_question = recognizer.recognize_google(audio)
            print(f"You said: {user_question}")
            return user_question
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
            return None
        except sr.RequestError:
            print("Could not request results from the service.")
            return None

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    new_db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
    
   
    retriever = new_db.as_retriever()
    chain = get_conversational_chain(retriever)
    
    
   # response = chain({"query": user_question})
    response = chain.invoke({"query": user_question})
    result_text = response['result']

      
    engine = pyttsx3.init()
    engine.say(result_text)
    engine.runAndWait()

    print(result_text)


#     print(response['result'])
# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
#     new_db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
    
#     # Get the retriever from the FAISS index
#     retriever = new_db.as_retriever()
#     chain = get_conversational_chain(retriever)
    
#     # Use the new RetrievalQA chain
#     response = chain.invoke({"query": user_question})
#     result_text = response['result']
    
#     # Convert the result into speech
#     engine = pyttsx3.init()
#     engine.say(result_text)
#     engine.runAndWait()

#     print(result_text)

# Main function to execute the chatbot process
# def main():
#     user_question = 'tell me everything you know about land'
#     user_input(user_question)
#     #If needed, uncomment the following lines to process the PDF
#     #raw_text = get_pdf_text()
#     #text_chunks = get_text_chunks(raw_text)
#     #get_vector_store(text_chunks)

def main():
    user_question = get_audio_input()
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
