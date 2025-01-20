import PyPDF2
import groq
import os
from pydantic import BaseModel, ValidationError
import spacy
from io import BytesIO
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise EnvironmentError("🚨 Groq API key not found. Please set it in your environment variables.")

# Initialize Groq client
client = groq.Client(api_key=GROQ_API_KEY)

# Initialize spaCy model for NLP (replace SentenceTransformer)
nlp = spacy.load("en_core_web_sm")

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
        pdf_reader = PyPDF2.PdfReader(BytesIO(file))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        if not text:
            raise ValueError("No text extracted from the PDF. It might be a scanned image or unsupported format.")
        return text
    except Exception as e:
        raise Exception(f"Failed to extract text from the PDF: {str(e)}")

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
        raise Exception(f"Error while analyzing the resume: {str(e)}")

# Lambda handler
def lambda_handler(event, context):
    try:
        # Assume 'body' contains the PDF file data as binary (base64-encoded)
        body = json.loads(event['body'])
        file_data = BytesIO(base64.b64decode(body['file']))

        # Step 1: Extract text from the uploaded PDF
        resume_text = extract_text_from_pdf(file_data)

        # Step 2: Validate if the extracted text is from a valid resume
        if not is_resume(resume_text):
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "This is not a valid resume. Please upload a valid resume file."})
            }

        # Step 3: Analyze the resume content using Groq
        analysis_result = analyze_resume(resume_text)

        if analysis_result:
            try:
                # Parse the analysis result (expected to be in JSON format)
                parsed_result = json.loads(analysis_result)

                # Validate and structure the response using Pydantic
                resume_details = ResumeDetails(**parsed_result)

                return {
                    "statusCode": 200,
                    "body": json.dumps(resume_details.dict())
                }
            except (json.JSONDecodeError, ValidationError) as e:
                return {
                    "statusCode": 500,
                    "body": json.dumps({"error": f"Error processing the analysis result: {str(e)}"})
                }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"An error occurred: {str(e)}"})
        }
