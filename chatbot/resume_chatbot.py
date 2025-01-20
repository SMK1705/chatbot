import streamlit as st
import PyPDF2
import groq
from dotenv import load_dotenv
import os
from pydantic import BaseModel, ValidationError
from sentence_transformers import SentenceTransformer, util

# Load environment variables
load_dotenv()

# Access API key for Groq
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.sidebar.error("🚨 API key not found! Please check your `.env` file and ensure `GROQ_API_KEY` is set.")

# Initialize Groq client
client = groq.Groq(api_key=api_key)

# Function to extract text from a PDF
def extract_pdf_text(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to generate a response using Gemma
def generate_response(question, context):
    prompt = f"""
    You are a resume analysis assistant. Answer the question based on the provided resume content:

    Resume Content:
    {context}

    Question: {question}
    Answer:
    """
    response = client.chat.completions.create(
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        model="gemma2-9b-it",  # Specify the Gemma model version
        temperature=0.7,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

# Streamlit App
def main():
    st.set_page_config(page_title="Resume Chatbot", page_icon="📄")
    st.title("📄 Resume Chatbot")
    st.write("Upload your resume and ask questions about it in real-time!")

    # Sidebar for status updates
    st.sidebar.header("Status")
    st.sidebar.write("Upload your resume to get started.")

    # File uploader
    pdf_file = st.file_uploader("Upload a PDF resume", type=["pdf"])

    if pdf_file:
        with st.spinner("Processing the file..."):
            try:
                # Extract text from the uploaded PDF
                extracted_text = extract_pdf_text(pdf_file)

                # Ask the model if the text is a valid resume
                prompt = f"Based on the provided text, is this document a valid resume? Please respond with 'Yes' or 'No'.\n\nText:\n{extracted_text}"
                model_response = generate_response(prompt, extracted_text)

                # If model indicates it's not a resume
                if model_response.lower() != "yes":
                    st.error("🚨 The uploaded document does not appear to be a valid resume. Please upload a valid resume.")
                    return

                st.success("Resume uploaded and validated successfully! 🎉")

                # Store the extracted text in session state
                if "resume_text" not in st.session_state:
                    st.session_state.resume_text = extracted_text

                # Chat functionality
                if "conversation" not in st.session_state:
                    st.session_state.conversation = [{"role": "system", "content": "You are a helpful assistant."}]

                # Chat input
                user_query = st.chat_input("Ask a question about the resume:")
                
                if user_query:
                    # Append user query to conversation
                    st.session_state.conversation.append({"role": "user", "content": user_query})

                    # Generate response
                    with st.spinner("Generating response..."):
                        response_text = generate_response(user_query, st.session_state.resume_text)
                        
                        # Generate JSON response
                        try:
                            json_response = {
                                "question": user_query,
                                "answer": response_text,
                                "confidence": 0.95  # Simulated confidence
                            }
                            st.session_state.conversation.append({"role": "assistant", "content": json_response})
                        except ValidationError as e:
                            st.error(f"Validation error: {e}")

                # Display chat history
                for msg in st.session_state.conversation:
                    if msg["role"] == "user":
                        st.chat_message("user").write(msg["content"])
                    else:
                        st.chat_message("assistant").write(msg["content"])

            except Exception as e:
                st.error(f"Failed to process the file. Error: {e}")

if __name__ == "__main__":
    main()
