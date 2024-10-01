# Import necessary packages.
# Author: Aditya
# Date: 25/12/2022

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import absl.logging
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import cv2
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
import google.generativeai as genai
import os

# Ignoring all unnecessary warnings.
absl.logging.set_verbosity(absl.logging.ERROR)
tf.get_logger().setLevel("ERROR")

# Instantiate fastapi server
app = FastAPI()

# Define links that CORS policy should allow.
origins = [
    "http://localhost",
    "http://localhost:3000"
]

# Add middlewares to the fastapi instance.
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
MODEL_PATH = r"../saved_models2/pretrained/1"
MODEL = tf.keras.layers.TFSMLayer(MODEL_PATH, call_endpoint='serving_default')

# Note: This should correspond with the one in the notebooks. 
LABELS = ['No Tumour', 'Tumour']

# Global variable to store the analysis report
analysis_report_text = ""

def bytes_to_image(bytes) -> np.ndarray:
    image = np.array(Image.open(BytesIO(bytes)))
    return image

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    # file.read returns a byte array which is converted to an image
    image = bytes_to_image(await file.read())

    # Resize the image because the model expects (224, 224)
    image = cv2.resize(image, (224, 224))

    # Convert image to float32
    image = image.astype(np.float32)

    # Create a batch.
    image_batch = np.expand_dims(image, 0)

    prediction = MODEL(image_batch)
    predicted_label = LABELS[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    print(prediction[0])
    return {
        "label": predicted_label,
        "confidence": float(confidence)
    }

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

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)