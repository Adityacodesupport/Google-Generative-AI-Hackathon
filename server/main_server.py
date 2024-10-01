from fastapi import FastAPI, UploadFile, File, Request, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
import cv2
from pymongo import MongoClient
from pydantic import BaseModel
from typing import Optional
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
import google.generativeai as genai
import os
from pydantic import BaseModel


app = FastAPI()


# Google Gemini API configuration
os.environ['GOOGLE_API_KEY'] = "AIzaSyBM_I5M5d51BnnbQ4-XLoJ8i3bCOrA1i0E"
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8192,
}

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

system_prompts = [
    """
    You are a domain expert in medical image analysis. You are tasked with 
    examining medical images for a renowned hospital.
    Your expertise will help in identifying or 
    discovering any anomalies, diseases, conditions or
    any health issues that might be present in the image.
    
    Your key responsibilities:
    1. Detailed Analysis : Scrutinize and thoroughly examine each image, 
    focusing on finding any abnormalities.
    2. Analysis Report : Document all the findings and 
    clearly articulate them in a structured format.
    3. Recommendations : Basis the analysis, suggest remedies, 
    tests or treatments as applicable.
    4. Treatments : If applicable, lay out detailed treatments 
    which can help in faster recovery.
    
    Important Notes to remember:
    1. Scope of response : Only respond if the image pertains to 
    human health issues.
    2. Clarity of image : In case the image is unclear, 
    note that certain aspects are 
    'Unable to be correctly determined based on the uploaded image'
    3. Disclaimer : Accompany your analysis with the disclaimer: 
    "Consult with a Doctor before making any decisions."
    4. Your insights are invaluable in guiding clinical decisions. 
    Please proceed with the analysis, adhering to the 
    structured approach outlined above.
    
    Please provide the final response with these 4 headings : 
    Detailed Analysis, Analysis Report, Recommendations and Treatments
    """
]

model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

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

# MongoDB setup
client = MongoClient("mongodb://localhost:27017")
db = client["cancer_detection"]
patients_collection = db["patients"]

# Load the trained model and set the labels
MODEL = tf.keras.models.load_model(r"../saved_models2/pretrained/1")
LABELS = ['No Tumour', 'Tumour']


class PatientData(BaseModel):
    id: Optional[str]
    name: str
    age: int
    gender: str
    medications: Optional[str] = None
    symptoms: Optional[str] = None
    report: Optional[str] = None

def bytes_to_image(bytes_data: bytes) -> np.ndarray:
    """Convert byte stream to image."""
    image = Image.open(BytesIO(bytes_data)).convert("RGB")
    return np.array(image)


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

    # Prepare the image and prompt for content generation
    image_parts_clasify = [
        {
            "mime_type": "image/jpg",
            "data": image_bytes  # Use bytes format for generative model input
        }
    ]

    patient_promt = [
        f"Patient Details:\nPatient Name: {name}\nAge: {age}\nGender: {gender}\nMedications: {medications}\nSymptoms: {symptoms}\n",
    ]
    
    prompt_parts_clasify = [
        image_parts_clasify[0],
        system_prompts[0],
        patient_promt[0]
    ]

    # Model prediction for image classification
    try:
        # Prediction using the classification model
        prediction = MODEL.predict(image_batch)  # Replace with your model's prediction method
        predicted_label = LABELS[np.argmax(prediction[0])]  # Get label based on max probability
        confidence = float(np.max(prediction[0]))  # Get max probability as confidence
    except Exception as e:
        return {"error": f"Error in model prediction: {str(e)}"}

    # Generate content using the generative model
    try:
        response_clasify = model.generate_content(prompt_parts_clasify)
        analysis_report_text_clasify = response_clasify.text if response_clasify else "No analysis available."
        patients_collection.insert_one({"name": name, "age": age, "gender": gender, "medications": medications, "symptoms": symptoms,"report":analysis_report_text_clasify})
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)