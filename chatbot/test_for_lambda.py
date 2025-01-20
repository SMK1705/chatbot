import base64

# Use the correct file path (update as per your system)
file_path = r"chatbot\resume.pdf"

try:
    # Open the file and encode it in Base64
    with open(file_path, 'rb') as pdf_file:
        encoded_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
        print("Base64 Encoded PDF Content:")
        print(encoded_pdf)
except FileNotFoundError:
    print(f"Error: File not found at path: {file_path}")
except OSError as e:
    print(f"Error: {e}")
