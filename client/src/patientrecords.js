// PatientsDataWithImageUpload.js
import React, { useState, useEffect } from 'react';
import { makeStyles, withStyles } from '@material-ui/core/styles';
import {
  Grid, Card, CardActionArea, CardMedia, CardContent, TextField, MenuItem, Typography, Button, CircularProgress
} from '@material-ui/core';
import { DropzoneArea } from 'material-ui-dropzone';
import { common } from '@material-ui/core/colors';
import axios from 'axios';
import jsPDF from 'jspdf';
import 'jspdf-autotable';
import htmlToPdfmake from 'html-to-pdfmake';
import pdfMake from 'pdfmake/build/pdfmake';
import pdfFonts from 'pdfmake/build/vfs_fonts';

pdfMake.vfs = pdfFonts.pdfMake.vfs;

const ColorButton = withStyles((theme) => ({
  root: {
    color: theme.palette.getContrastText(common.white),
    backgroundColor: common.white,
    '&:hover': {
      backgroundColor: '#ffffff7a',
    },
  },
}))(Button);

const useStyles = makeStyles((theme) => ({
  clearButton: {
    width: '100%',
    borderRadius: '15px',
    padding: '15px 22px',
    color: '#082567',
    fontSize: '18px',
    fontWeight: 800,
    backgroundColor: '#F8F0E3',
    textShadow: '1.1px 1.1px #36454F',
    marginTop: '16px',
  },
  downloadButton: {
    width: '100%',
    borderRadius: '15px',
    padding: '15px 22px',
    color: '#ffffff',
    fontSize: '18px',
    fontWeight: 800,
    backgroundColor: '#082567',
    '&:hover': {
      backgroundColor: '#0a3d7a',
    },
    marginTop: '16px',
  },
  generateButton: {
    width: '100%',
    borderRadius: '15px',
    padding: '15px 22px',
    color: '#082567',
    fontSize: '18px',
    fontWeight: 800,
    backgroundColor: '#F8F0E3',
    textShadow: '1.1px 1.1px #36454F',
    '&:hover': {
      backgroundColor: '#e0d4c3',
    },
    marginTop: '16px',
  },
  media: {
    height: 400,
    borderRadius: '12px',
  },
  analysisText: {
    margin: '16px 0',
    fontSize: '18px',
    fontWeight: 600,
    color: '#1F2937',
    textAlign: 'justify',
    border: '1px solid #ddd',
    borderRadius: '8px',
    padding: '12px',
    backgroundColor: '#ffffff',
    boxShadow: '0px 0px 10px rgba(0, 0, 0, 0.1)',
  },
  sectionTitle: {
    fontWeight: 700,
    fontSize: '24px',
    color: '#082567',
    marginBottom: '12px',
  },
  imageCard: {
    margin: 'auto',
    maxWidth: 450,
    height: 'auto',
    backgroundColor: 'transparent',
    boxShadow: '0px 9px 70px rgba(0, 0, 0, 0.2)',
    borderRadius: '15px',
    padding: '20px',
  },
  loader: {
    color: '#082567 !important',
  },
  buttonContainer: {
    marginTop: '16px',
    width: '100%',
  },
  formField: {
    marginTop: '16px',
  },
  mainContainer: {
    padding: '20px',
    backgroundColor: '#f4f6f8',
    borderRadius: '15px',
    boxShadow: '0px 4px 20px rgba(0, 0, 0, 0.1)',
  },
  reportContainer: {
    marginTop: '20px',
    padding: '20px',
    borderRadius: '15px',
    backgroundColor: '#ffffff',
    boxShadow: '0px 4px 20px rgba(0, 0, 0, 0.1)',
  },
}));

const PatientsDataWithImageUpload = () => {
  const classes = useStyles();
  const [selectedFile, setSelectedFile] = useState();
  const [preview, setPreview] = useState();
  const [data, setData] = useState();
  const [isLoading, setIsloading] = useState(false);
  const [patientName, setPatientName] = useState('');
  const [patientAge, setPatientAge] = useState('');
  const [patientGender, setPatientGender] = useState('');
  const [patientSymptoms, setPatientSymptoms] = useState('');
  const [patientMedications, setPatientMedications] = useState('');
  const [analysis, setAnalysis] = useState('');

  const sendFile = async () => {
    if (!selectedFile || !patientName || !patientAge || !patientGender) {
      alert('Please fill in all required fields and upload an image.');
      return;
    }

    setIsloading(true);
    let formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('name', patientName);
    formData.append('age', patientAge);
    formData.append('gender', patientGender);
    formData.append('symptoms', patientSymptoms);
    formData.append('medications', patientMedications);

    try {
      let res = await axios.post(process.env.REACT_APP_API_URL, formData);
      if (res.status === 200) {
        setData(res.data);
        setAnalysis(res.data.analysis_report_text_clasify);
      }
    } catch (error) {
      console.error('Error uploading the image:', error);
    } finally {
      setIsloading(false);
    }
  };

  const clearData = () => {
    setData(null);
    setSelectedFile(null);
    setPreview(null);
    setPatientName('');
    setPatientAge('');
    setPatientGender('');
    setPatientSymptoms('');
    setPatientMedications('');
  };

  const downloadPdf = () => {
    const doc = new jsPDF();
    let yOffset = 20;

    // Title
    doc.setFontSize(16);
    doc.text('Patient Analysis Report', 20, yOffset);
    yOffset += 10;

    // Patient Details
    doc.setFontSize(14);
    doc.text(`Name: ${patientName}`, 20, yOffset);
    yOffset += 10;
    doc.text(`Age: ${patientAge}`, 20, yOffset);
    yOffset += 10;
    doc.text(`Gender: ${patientGender}`, 20, yOffset);
    yOffset += 10;
    if (patientSymptoms) {
      doc.text(`Symptoms: ${patientSymptoms}`, 20, yOffset);
      yOffset += 10;
    }
    if (patientMedications) {
      doc.text(`Medications: ${patientMedications}`, 20, yOffset);
      yOffset += 10;
    }

    // Analysis Section
    doc.setFontSize(14);
    doc.text('Detailed Analysis:', 20, yOffset);
    yOffset += 10;

    const analysisLines = doc.splitTextToSize(analysis, 180);
    for (let i = 0; i < analysisLines.length; i++) {
      if (yOffset > 270) {
        doc.addPage();
        yOffset = 20;
      }
      doc.text(analysisLines[i], 20, yOffset);
      yOffset += 10;
    }

    // Recommendations Section
    doc.setFontSize(14);
    doc.text('Recommendations:', 20, yOffset);
    yOffset += 10;

    const recommendations = `Follow-Up Tests:\n- Repeat MRI\n- MRI with contrast\n- CSF analysis\n- Neurological Examination\n- Blood tests\n\nLifestyle Modifications:\n- Stress management\n- Healthy diet\n\nUrgency Level:\n- Urgent medical consultation recommended.`;
    
    const recommendationsLines = doc.splitTextToSize(recommendations, 180);
    for (let i = 0; i < recommendationsLines.length; i++) {
      if (yOffset > 270) {
        doc.addPage();
        yOffset = 20;
      }
      doc.text(recommendationsLines[i], 20, yOffset);
      yOffset += 10;
    }

    // Additional Styling
    doc.setFontSize(12);
    doc.text('Generated on: ' + new Date().toLocaleString(), 20, yOffset);

    // Download the PDF
    doc.save('patient_analysis_report.pdf');
  };

  useEffect(() => {
    if (!selectedFile) {
      setPreview(undefined);
      return;
    }
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreview(objectUrl);
  }, [selectedFile]);

  const onSelectFile = (files) => {
    if (!files || files.length === 0) {
      setSelectedFile(undefined);
      return;
    }
    setSelectedFile(files[0]);
  };

  const downloadReport = () => {
    const htmlContent = document.getElementById('report-content').innerHTML;
    const pdfContent = htmlToPdfmake(htmlContent);
    const documentDefinition = { content: pdfContent };
    pdfMake.createPdf(documentDefinition).download('report.pdf');
  };

  const renderReport = (report) => {
    const lines = report.split('\n');
    return lines.map((line, index) => {
      if (line.startsWith('**')) {
        return <Typography key={index} variant="h5" className={classes.reportTitle}>{line.replace(/\*\*/g, '').trim()}</Typography>;
      } else if (line.startsWith('##')) {
        return <Typography key={index} variant="h6" className={classes.reportSubtitle}>{line.replace('##', '').trim()}</Typography>;
      } else if (line.startsWith('*')) {
        const parts = line.split('**');
        return (
          <Typography key={index} variant="body1" className={classes.reportText}>
            {parts.map((part, i) => (
              i % 2 === 1 ? <strong key={i}>{part}</strong> : part
            ))}
          </Typography>
        );
      } else {
        return <Typography key={index} variant="body1" className={classes.reportText}>{line.trim()}</Typography>;
      }
    });
  };

  return (
    <Grid container direction="column" alignItems="center" spacing={2} className={classes.mainContainer}>
      {/* Patient Details Section */}
      <Grid item xs={12}>
        <Typography className={classes.sectionTitle}>Patient Details</Typography>
        <TextField
          fullWidth
          label="Patient Name"
          variant="outlined"
          value={patientName}
          onChange={(e) => setPatientName(e.target.value)}
          required
          className={classes.formField}
        />
        <TextField
          fullWidth
          label="Patient Age"
          variant="outlined"
          type="number"
          value={patientAge}
          onChange={(e) => setPatientAge(e.target.value)}
          required
          className={classes.formField}
        />
        <TextField
          fullWidth
          select
          label="Gender"
          variant="outlined"
          value={patientGender}
          onChange={(e) => setPatientGender(e.target.value)}
          required
          className={classes.formField}
        >
          <MenuItem value="Male">Male</MenuItem>
          <MenuItem value="Female">Female</MenuItem>
          <MenuItem value="Other">Other</MenuItem>
        </TextField>
        <TextField
          fullWidth
          label="Symptoms"
          variant="outlined"
          value={patientSymptoms}
          onChange={(e) => setPatientSymptoms(e.target.value)}
          className={classes.formField}
        />
        <TextField
          fullWidth
          label="Medications"
          variant="outlined"
          value={patientMedications}
          onChange={(e) => setPatientMedications(e.target.value)}
          className={classes.formField}
        />
      </Grid>

      {/* Image Upload Section */}
      <Grid item xs={12}>
        <Typography className={classes.sectionTitle}>Upload MRI Image</Typography>
        <DropzoneArea
          acceptedFiles={['image/jpeg', 'image/png', 'image/tiff']}
          dropzoneText="Drag and drop an MRI image here or click"
          onChange={onSelectFile}
          filesLimit={1}
        />
        {preview && (
          <Card className={classes.imageCard}>
            <CardActionArea>
              <CardMedia
                className={classes.media}
                image={preview}
                title="MRI Image Preview"
              />
              <CardContent>
                <Typography variant="h6" component="h2">
                  MRI Image Preview
                </Typography>
              </CardContent>
            </CardActionArea>
          </Card>
        )}
      </Grid>

      {/* Action Buttons */}
      <Grid item className={classes.buttonContainer}>
        <ColorButton variant="contained" className={classes.generateButton} onClick={sendFile} disabled={isLoading}>
          {isLoading ? <CircularProgress size={24} className={classes.loader} /> : 'Analyze MRI'}
        </ColorButton>
        {data && (
          <ColorButton variant="contained" className={classes.downloadButton} onClick={downloadReport}>
            Download Report
          </ColorButton>
        )}
        <ColorButton variant="contained" className={classes.clearButton} onClick={clearData}>
          Clear
        </ColorButton>
      </Grid>

      {/* Analysis Section */}
      {data && (
        <Grid item xs={12} className={classes.reportContainer}>
          <Typography className={classes.sectionTitle}>Analysis</Typography>
          <Typography variant="body1" className={classes.analysisText} id='report-content'>
            {renderReport(analysis)}
          </Typography>
        </Grid>
      )}
    </Grid>
  );
};

export default PatientsDataWithImageUpload;