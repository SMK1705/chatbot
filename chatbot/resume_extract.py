import streamlit as st
import PyPDF2
import groq
from dotenv import load_dotenv
import os
from pydantic import BaseModel, ValidationError
from sentence_transformers import SentenceTransformer
from io import BytesIO
import json

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("🚨 Groq API key not found. Please add it to your .env file.")
    st.stop()

# Initialize Groq client
client = groq.Client(api_key=GROQ_API_KEY)

# Initialize SentenceTransformer model (can be extended for future enhancements)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Pydantic model for resume details
class ResumeDetails(BaseModel):
    name: str
    email: str
    phone: str
    education: list
    experience: list
    skills: list
    projects: list
    certifications: list

# Function to extract text from the uploaded PDF
def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(file.getvalue()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        if not text:
            raise ValueError("No text extracted from the PDF. It might be a scanned image or an unsupported format.")
        return text
    except Exception as e:
        st.error(f"Failed to extract text from the PDF: {str(e)}")
        return ""

# Function to check if the uploaded file is a resume
def is_resume(resume_text):
    # Check for presence of common resume keywords
    keywords = ["name", "education", "experience", "skills", "contact", "summary", "projects", "certifications"]
    
    # Checking for keywords in the resume text (case-insensitive)
    for keyword in keywords:
        if keyword.lower() in resume_text.lower():
            return True
    return False

# Function to analyze the resume using the Groq LLM
def analyze_resume(resume_text):
    prompt = f"""
    Analyze the following resume text and extract key information. Provide the response in JSON format with the following fields:
    - name: Full name of the applicant (string)
    - email: Email address of the applicant (email)
    - phone: Phone number of the applicant (string)
    - education: Educational background of the applicant (list of dictionaries)
    - experience: Work experience of the applicant (list of dictionaries)
    - skills: Skills of the applicant (list of strings)
    - projects: List of projects (list of dictionaries)
    - certifications: List of certifications (list of strings)

    Resume text:
    {resume_text}
    """

    try:
        response = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes resumes."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error while analyzing the resume: {str(e)}")
        return ""

# Streamlit UI
st.title("Resume Analyzer")

# Upload file widget
uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

if uploaded_file is not None:
    try:
        # Step 1: Extract text from the uploaded PDF
        resume_text = extract_text_from_pdf(uploaded_file)

        if resume_text:  # Only proceed if the text extraction is successful
            st.success("Resume uploaded and text extracted successfully!")

            # Step 2: Validate if the extracted text is from a valid resume
            if not is_resume(resume_text):
                st.error("❌ This is not a valid resume. Please upload a valid resume file.")
                st.stop()  # Stop further processing

            # Step 3: Display the extracted resume text
            st.subheader("Extracted Resume Text")
            st.text_area("Resume Content", resume_text, height=300)

            # Step 4: Analyze the resume content using Groq
            analysis_result = analyze_resume(resume_text)

            if analysis_result:
                try:
                    # Parse the analysis result (expected to be in JSON format)
                    parsed_result = json.loads(analysis_result)

                    # Step 5: Validate and structure the response using Pydantic
                    resume_details = ResumeDetails(**parsed_result)

                    # Display the structured resume details in JSON format
                    st.subheader("Resume Analysis Result")
                    st.json(resume_details.dict())

                except json.JSONDecodeError:
                    st.error("Error parsing the analysis result: Invalid JSON response.")
                except ValidationError as e:
                    st.error(f"Error validating the analysis result: {str(e)}")

    except Exception as e:
        st.error(f"An error occurred while processing the resume: {str(e)}")

# Sidebar: Instructions and About section
st.sidebar.header("Instructions")
st.sidebar.info(
    "1. Upload a PDF resume.\n"
    "2. View extracted text.\n"
    "3. View AI-powered analysis."
)

st.sidebar.header("About")
st.sidebar.info(
    "This app demonstrates resume analysis using Streamlit, "
    "PyPDF2 for text extraction, Groq for AI-powered analysis, "
    "and Sentence Transformers for potential future enhancements."
)
