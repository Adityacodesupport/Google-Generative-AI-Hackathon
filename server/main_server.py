# Import necessary libraries and modules
from fastapi import FastAPI, UploadFile, File, Request, Body  # FastAPI essentials for handling HTTP requests and file uploads
from fastapi.middleware.cors import CORSMiddleware  # Middleware to enable CORS (Cross-Origin Resource Sharing)
import uvicorn  # For running the FastAPI server
import tensorflow as tf  # TensorFlow for machine learning model handling
import numpy as np  # Numpy for numerical computations
from io import BytesIO  # To handle byte streams (image data)
from PIL import Image  # Image processing using PIL
import cv2  # OpenCV for image processing
from pymongo import MongoClient  # MongoDB for database interactions
from pydantic import BaseModel  # Pydantic for data validation
from typing import Optional  # To handle optional fields in models
from reportlab.lib.pagesizes import letter  # For PDF generation
from reportlab.pdfgen import canvas  # Canvas to draw PDF
from reportlab.lib import colors  # For text colors in PDFs
from reportlab.lib.styles import getSampleStyleSheet  # Styles for PDF text formatting
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer  # ReportLab classes for document creation
import google.generativeai as genai  # Google Gemini API for generative AI
import os  # OS module to handle environment variables
import smtplib  # Email sending via SMTP
from email.mime.text import MIMEText  # To format the email body
import datetime as dt  # For timestamp in notifications
from dotenv import load_dotenv
# Initialize the FastAPI app
app = FastAPI()
load_dotenv()
# Google Gemini API configuration (API key is retrieved from the environment variable)
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])  # Configure the Gemini API with the provided key

# Set generation configuration for the generative model (temperature, top_p, etc.)
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8192,  # Set max tokens to allow large responses
}

# Safety settings to ensure content generated is appropriate
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
]

# System prompt for guiding the generative AI to perform medical analysis
system_prompts = [
    """
    Medical Image Analysis Expert Prompt
You are a specialized medical image analyst with expertise in clinical diagnostics and patient care. Your role is to provide comprehensive medical analysis by integrating visual image data with patient information.
Be Accurate give proper estimation of tumour size be specific about the location of the tumour and the type of tumour. you are being used under a condition that doctor would rediagnose even after your report.
Inside Patient Demographics you have already given the classified class of the vgg16 model which you can use to cross verify the results by analysis with image and this.
Available Information:

Medical Image
Patient Demographics:
Classification Class 
Name
Age
Gender
Medications (if provided)
Symptoms (if provided)

Your Responsibilities:

Comprehensive Analysis

Examine the medical image thoroughly
Correlate visual findings with patient demographics
Validate and elaborate on the AI classification result
Consider provided symptoms and medications in your analysis


Detailed Report Generation
Structure your response with the following sections:
a) Detailed Analysis

Describe visual observations from the image
Correlate findings with patient's age and gender
Discuss relevance of provided symptoms and medications
Validate or expand upon the AI-classified condition

b) Analysis Report

Summarize key findings
Explain the significance of the AI classification in context
Note any age or gender-specific considerations
Discuss how current medications might impact the condition

c) Recommendations

Suggest appropriate follow-up tests or examinations
Recommend lifestyle modifications if applicable
Propose medication adjustments if relevant
Indicate urgency level for medical consultation

d) Treatments

Outline potential treatment options
Consider age-appropriate treatment modifications
Address potential drug interactions with current medications
Suggest supportive care measures


Critical Considerations

Maintain a professional, clinical tone
Use medical terminology appropriately
Consider age and gender-specific health factors
Account for provided medications and symptoms in your analysis


Important Guidelines

Scope: Respond only to human health-related images
Image Clarity: Note if any aspects are unclear
Accuracy: Rely on the AI classification as a validated starting point
Comprehensiveness: Integrate all provided patient information



Required Disclaimers:

"This analysis is based on the provided image and patient information, combined with AI classification results."
"This report is for informational purposes and should be reviewed by a qualified healthcare professional."
"Final diagnosis and treatment decisions should be made by a licensed medical practitioner."
"Consult with a Doctor before making any medical decisions."

Report Structure:

Patient Information Summary
AI Classification Result
Detailed Analysis
Analysis Report
Recommendations
Treatments
Disclaimers

Remember: Your analysis should be thorough, professional, and clinically relevant, integrating all provided information to support medical decision-making while acknowledging the limitations of image-based analysis.
    """
]

# Instantiate the generative model
model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

# Route for image analysis using the Gemini AI model
@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    global analysis_report_text
    image_data = await file.read()
    
    image_parts = [
        {
            "mime_type": "image/jpg",
            "data": image_data
        }
    ]
    
    prompt_parts = [
        image_parts[0],
        system_prompts[0],
    ]
    
    response = model.generate_content(prompt_parts)
    analysis_report_text = response.text if response else "No analysis available."
    
    return {"report": analysis_report_text}

# Route to generate and download the analysis report as a PDF
@app.get("/api/report/pdf")
async def get_report_pdf():
    global analysis_report_text

    # Create PDF
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Add title
    title = Paragraph("Medical Image Analysis Report", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 12))

    # Add analysis report
    analysis_paragraph = Paragraph(analysis_report_text, styles['BodyText'])
    elements.append(analysis_paragraph)

    # Build PDF
    doc.build(elements)

    buffer.seek(0)
    return StreamingResponse(buffer, media_type='application/pdf', headers={"Content-Disposition": "attachment; filename=report.pdf"})



# CORS setup to allow frontend communication
origins = [
    "http://localhost",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB setup to store patient data
client = MongoClient(os.getenv("MONGO_URI"))
db = client["cancer_detection"]
patients_collection = db["patients"]

# Load the trained model and set the labels
MODEL = tf.keras.models.load_model(r"C:\Users\ganes\Desktop\GenAI App\Brain_tumour\Brain_tumour\saved_models2\model.h5")
LABELS = ['No Tumour', 'Tumour']

# Data model for storing patient information
class PatientData(BaseModel):
    id: Optional[str]
    name: str
    age: int
    gender: str
    medications: Optional[str] = None
    symptoms: Optional[str] = None
    report: Optional[str] = None

# Utility function to convert bytes to an image
def bytes_to_image(bytes_data: bytes) -> np.ndarray:
    """Convert byte stream to image."""
    image = Image.open(BytesIO(bytes_data)).convert("RGB")
    return np.array(image)

# Route for classifying the medical image and generating a report
@app.post("/classify")
async def classify(
    name: str = Body(...),
    file: UploadFile = File(...),
    age: int = Body(...),
    gender: str = Body(...),
    medications: Optional[str] = Body(None),
    symptoms: Optional[str] = Body(None)
):
    # Convert bytes to an OpenCV image
    def bytes_to_image(image_bytes):
        # Convert bytes to a numpy array
        image_array = np.frombuffer(image_bytes, np.uint8)
        # Decode the numpy array into an OpenCV image (BGR format)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Unable to decode the image. Please check the image format.")
        return image

    # Read the file bytes and convert them to an OpenCV image
    try:
        image = bytes_to_image(await file.read())
    except Exception as e:
        return {"error": f"Error reading image: {str(e)}"}

    # Resize the image to the required size and add batch dimension
    try:
        image = cv2.resize(image, (224, 224))  # Resize image to 224x224 pixels
        image_batch = np.expand_dims(image, 0)  # Add batch dimension for prediction
    except Exception as e:
        return {"error": f"Error processing image: {str(e)}"}
    
    # Encode the image into bytes format for prompt generation
    _, encoded_image = cv2.imencode('.jpg', image)  # Encode OpenCV image to jpg format
    image_bytes = encoded_image.tobytes()  # Convert encoded image to bytes

     # Model prediction for image classification
    try:
        # Prediction using the classification model
        prediction = MODEL.predict(image_batch)  # Replace with your model's prediction method
        predicted_label = LABELS[np.argmax(prediction[0])]  # Get label based on max probability
        timestamp=dt.datetime.now()
        confidence = float(np.max(prediction[0]))  # Get max probability as confidence
    except Exception as e:
        return {"error": f"Error in model prediction: {str(e)}"}

    # Prepare the image and prompt for content generation
    image_parts_clasify = [
        {
            "mime_type": "image/jpg",
            "data": image_bytes  # Use bytes format for generative model input
        }
    ]

    patient_promt = [
        f"Patient Details:\nClassification Class : {predicted_label} \nPatient Name: {name}\nAge: {age}\nGender: {gender}\nMedications: {medications}\nSymptoms: {symptoms}\n",
    ]
    
    prompt_parts_clasify = [
        image_parts_clasify[0],
        system_prompts[0],
        patient_promt[0]
    ]

   
    # Generate content using the generative model
    try:
        response_clasify = model.generate_content(prompt_parts_clasify)
        analysis_report_text_clasify = response_clasify.text if response_clasify else "No analysis available."
        patients_collection.insert_one({"name": name, "age": age, "gender": gender, "medications": medications, "symptoms": symptoms,"report":analysis_report_text_clasify})
        #email push notification function

        def email_script(rep):
            sender_email = os.getenv("EMAIL_USER")
            app_password = os.getenv("EMAIL_PASSWORD")

            # Create the email message
            msg = MIMEText(rep)
            msg['Subject'] = "URGENT: Critical Cancer Detection Alert - Immediate Action Required"
            msg['From'] = "AI Cancer Detection System"
            msg['To'] = os.getenv("DOCTOR_EMAIL")

            # Connect to Gmail's SMTP server
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
                smtp_server.login(sender_email, app_password)
                smtp_server.send_message(msg)

            print("Email sent successfully!")

        if (predicted_label == 'Tumour'):
            rep = f'''
        Dear Doctor,

        This is an automated alert from the AI-powered Cancer Detection System.

        Patient Details:
        - Name: {name}
        - Age: {age}
        - Gender: {gender}

        Detection Summary:
        - Classification: {predicted_label}
        - Confidence Level: {confidence*100}
        - Detection Date/Time: {timestamp}

        Please access the full report and imaging on our portal.

        This is an automated alert. Please verify all findings according to standard protocols.

        Regards,
        AI Cancer Detection System.
        '''
            email_script(rep)
    except Exception as e:
        return {"error": f"Error generating analysis report: {str(e)}"}




    # Return the classification results along with the analysis report
    return {
        "label": predicted_label,
        "confidence": confidence,
        "name": name,
        "age": age,
        "gender": gender,
        "medications": medications,
        "symptoms": symptoms,
        "analysis_report_text_clasify": analysis_report_text_clasify,
    }


# Custom function to serialize ObjectId to string
def serialize_patient(patient):
    patient["_id"] = str(patient["_id"])  # Convert ObjectId to string
    return patient


@app.get("/patient")
async def get_all_patients():
    patients = list(patients_collection.find())
    # Serialize each patient document
    serialized_patients = [serialize_patient(patient) for patient in patients]
    return serialized_patients



# Main entry point for running the FastAPI server on port 8000
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
