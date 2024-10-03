# Brain Tumor Diagnosis 🧠

> "Mind the Tumor: Catching Brain Bumps Before They Bump You!"

[![Devfolio](https://img.shields.io/badge/Devfolio-GenAI_Hackathon-blue)](https://devfolio.co)
[![Google Cloud](https://img.shields.io/badge/Used_Gemini_Models_from-Google_AI-4285F4?logo=google-cloud)]([https://cloud.google.com/](https://gemini.google.com/?hl=en-IN))
[![AWS](https://img.shields.io/badge/Deployed_on-AWS-232F3E?logo=amazon-aws)](https://aws.amazon.com/)

## 📜 Table of Contents
- [About the Project](#-about-the-project)
- [Features](#-features)
- [Demo](#-demo)
- [Live Demo](#-live-demo)
- [Technology Stack](#-technology-stack)
- [Architecture Design](#-architecture-design)
- [Project Setup](#-project-setup)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure-for-relevant-files)
- [Future Enhancements](#-future-enhancements)
- [License](#-license)
- [Contact](#-contact)

## 🎯 About the Project

Brain Tumor Diagnosis is an innovative web application designed to assist in the early detection and analysis of brain tumors using advanced AI technologies. Our platform combines cutting-edge image analysis with natural language processing to provide comprehensive medical reports based on MRI scans and patient information.

### Key Highlights:

1. Responsive React.js based Frontend User Interface
2. MRI Image Upload and Analysis
3. Patient Information Input Form
4. AI-powered Image Classification using VGG16 (89.92% Accuracy)
5. Detailed Medical Report Generation using Gemini 1.5 Flash Model
6. PDF Report Download Feature
7. Patient Records Management
8. Automatic Doctor Notification System

## ✨ Features

- **User-Friendly Interface**: Built with React.js for a smooth and responsive user experience.
- **MRI Image Analysis**: Upload and analyze brain MRI images with high accuracy.
- **Patient Information Collection**: Easily input and manage patient details.
- **AI-Powered Diagnosis**: Utilizes VGG16 model for image classification with 89.92% accuracy.
- **Comprehensive Reporting**: Gemini 1.5 Flash Model generates detailed medical reports by analyzing MRI images, classification results, and patient information.
- **PDF Report Generation**: Download and share medical reports in PDF format.
- **Patient Record Management**: Maintain and access patient records efficiently.
- **Instant Doctor Alerts**: Automatic email notifications to doctors upon tumor detection.

##  🌐 Live Demo

Experience our Brain Tumor Diagnosis tool in action:

[![Live Demo](https://img.shields.io/badge/LIVE_DEMO-Visit_Website-FF5722?style=for-the-badge)](http://54.197.15.179:3000/)

## 🎬 Demo

Check out our project demo video:

[![Project Demo](https://img.shields.io/badge/Watch-Demo_Video-red?style=for-the-badge&logo=youtube)](https://drive.google.com/file/d/1yOogRvFUUJPaCwesDb4zGiG1h9q5J2y2/view?usp=sharing)

## 📸 Screenshots

Here's a glimpse of our application in action:

<div align="center">
  <img src="Website Images/Home Page.png" alt="Homepage" width="45%">
  <img src="Website Images/MRI upload.png" alt="MRI Upload" width="45%">
</div>

<div align="center">
  <img src="Website Images/Download PDF.png" alt="Analysis Result" width="45%">
  <img src="Website Images/Download PDF 2.png" alt="Medical Report" width="45%">
</div>

<div align="center">
  <img src="Website Images/Download PDF 3.png" alt="Patient List" width="45%">
  <img src="Website Images/Patients List 1.png" alt="PDF Download" width="45%">
</div>

<div align="center">
  <img src="Website Images/Patients List 2.png" alt="Doctor Notification" width="45%">
  <img src="Website Images/Mail Alert.png" alt="Doctor Notification" width="45%">
</div>

## 🛠 Technology Stack

<table>
  <tr>
    <td align="center"><img src="https://img.shields.io/badge/-React-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React.js"></td>
    <td align="center"><img src="https://img.shields.io/badge/-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"></td>
    <td align="center"><img src="https://img.shields.io/badge/-TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"></td>
  </tr>
  <tr>
    <td align="center"><img src="https://img.shields.io/badge/-Gemini_1.5-4285F4?style=for-the-badge&logo=google&logoColor=white" alt="Gemini 1.5"></td>
    <td align="center"><img src="https://img.shields.io/badge/-Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></td>
    <td align="center"><img src="https://img.shields.io/badge/-AWS_EC2-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white" alt="AWS EC2"></td>
  </tr>
</table>

## 💻 Architecture Design

This is our System Architecture:

<div align="center">
  <img src="Website Images/Architecture design.png" alt="Arch Design" width="70%">
</div>

## 🚀 Project Setup

### Prerequisites

- Node.js (v14 or later)
- Python (v3.8 or later)
- pip

### Installation

1. Clone the repository
   ```sh
   git clone https://github.com/your_username/brain-tumor-diagnosis.git
   cd brain-tumor-diagnosis
   ```

2. Install frontend dependencies
   ```sh
   cd client
   npm install
   ```

3. Install backend dependencies
   ```sh
   cd ../server
   pip install -r requirements.txt
   ```

## 🖥 Usage

1. Start the frontend server
   ```sh
   cd client
   npm start
   ```

2. Start the backend server
   ```sh
   cd server
   uvicorn main_server:app --reload
   ```

3. Open your browser and navigate to `http://localhost:3000`

## 📁 Project Structure for Relevant files

```
brain-tumor-diagnosis/
├── client/               # React.js frontend
│   ├── src/
│   │   ├── components/   # images used
│   │   └── App.js
│   └── package.json
├── server/               # FastAPI backend
│   ├── main_server.py
│   └── requirements.txt
├── notebooks/               # Pre-trained VGG16 model
│   └── vgg16.ipynb
├── saved_models2/               # Pre-trained VGG16 model
│   ├── pretrained/1
│   |    └── model.h5
└── README.md
```

## 🔮 Future Enhancements

- Implement user authentication for doctors to access individual patient records
- Enhance the AI model to support multiple types of brain scans
- Develop a mobile application for on-the-go access

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📞 Contact

Team Name : Envision 3.0

Team Members :
1.  Aditya Khatode - aditya.khatode@pibythree.com
2.  Ganeshan Arumuganainar - ganeshan.nainar@pibythree.com
3.  Kaliraj Konar - kaliraj.konar@pibythree.com
4.  Vaishnavi Dhimate - vaishnavi.dhimate@pibythree.com

Website Link : http://54.197.15.179:3000/

---

Made with ❤️ for the GenAI Hackathon by Google
