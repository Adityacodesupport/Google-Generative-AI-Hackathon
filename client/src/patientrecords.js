// PatientsDataWithImageUpload.js
import React, { useState, useEffect } from 'react';
import { makeStyles, withStyles } from '@material-ui/core/styles';
import {
  Grid, Card, CardActionArea, CardMedia, CardContent, TextField, MenuItem, TableContainer,
  Paper, Table, TableBody, TableHead, TableRow, TableCell, Button, CircularProgress, Typography
} from '@material-ui/core';
import { DropzoneArea } from 'material-ui-dropzone';
import { common } from '@material-ui/core/colors';
import axios from 'axios';
import jsPDF from 'jspdf'; // Import jsPDF

// Customized Button
const ColorButton = withStyles((theme) => ({
  root: {
    color: theme.palette.getContrastText(common.white),
    backgroundColor: common.white,
    '&:hover': {
      backgroundColor: '#ffffff7a',
    },
  },
}))(Button);

// Styling for the component
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
  media: {
    height: 400,
    borderRadius: '12px',
  },
  patientDetails: {
    marginBottom: '16px',
    color: 'white',
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
  detail: {
    backgroundColor: '#F8F0E3',
    display: 'flex',
    justifyContent: 'center',
    flexDirection: 'column',
    alignItems: 'center',
    borderRadius: '12px',
    padding: '20px',
    margin: '10px 0',
    boxShadow: '0px 4px 20px rgba(0, 0, 0, 0.1)',
  },
  loader: {
    color: '#082567 !important',
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
  formField: {
    marginTop: '16px',
  },
  dropzoneText: {
    color: '#1F2937',
  },
}));

const PatientsDataWithImageUpload = () => {
  const classes = useStyles();
  const [selectedFile, setSelectedFile] = useState();
  const [preview, setPreview] = useState();
  const [data, setData] = useState();
  const [image, setImage] = useState(false);
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

    setIsloading(true); // Show loading spinner

    let formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('name', patientName);
    formData.append('age', patientAge);
    formData.append('gender', patientGender);
    formData.append('symptoms', patientSymptoms); // Optional
    formData.append('medications', patientMedications); // Optional

    try {
      let res = await axios.post(process.env.REACT_APP_API_URL, formData);
      if (res.status === 200) {
        setData(res.data);
        setAnalysis(res.data.analysis_report_text_clasify);
        console.log(res.data);
      }
    } catch (error) {
      console.error('Error uploading the image:', error);
    } finally {
      setIsloading(false); // Hide loading spinner
    }
  };

  const clearData = () => {
    setData(null);
    setImage(false);
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
    doc.setFontSize(16);
    doc.text('Patient Analysis Report', 20, 20);

    // Patient Details
    doc.setFontSize(14);
    doc.text(`Name: ${patientName}`, 20, 30);
    doc.text(`Age: ${patientAge}`, 20, 40);
    doc.text(`Gender: ${patientGender}`, 20, 50);
    if (patientSymptoms) {
      doc.text(`Symptoms: ${patientSymptoms}`, 20, 60);
    }
    if (patientMedications) {
      doc.text(`Medications: ${patientMedications}`, 20, 70);
    }

    // Analysis Details
    if (data) {
      doc.text(`Label: ${data.label}`, 20, 90);
      doc.text(`Confidence: ${(parseFloat(data.confidence) * 100).toFixed(2)}%`, 20, 100);
    }
    if (analysis) {
      doc.text('Analysis:', 20, 110);
      doc.text(analysis, 20, 120, { maxWidth: 170 }); // Wrap long text
    }

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
      setImage(false);
      setData(undefined);
      return;
    }
    setSelectedFile(files[0]);
    setData(undefined);
    setImage(true);
  };

  return (
    <Grid container direction="column" alignItems="center" spacing={2} style={{ padding: '20px', backgroundColor: '#f4f6f8', borderRadius: '15px' }}>
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
          label="Patient Gender"
          variant="outlined"
          select
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
        <Card className={classes.imageCard}>
          <CardActionArea>
            {image && preview ? (
              <CardMedia
                className={classes.media}
                image={preview}
                title="Patient MRI"
              />
            ) : (
              <DropzoneArea
                dropzoneText={<Typography className={classes.dropzoneText}>Drag and drop an image here or click to upload</Typography>}
                onChange={onSelectFile}
                filesLimit={1}
                acceptedFiles={['image/jpeg', 'image/png', 'image/jpg']}
              />
            )}
          </CardActionArea>
          <CardContent>
            <Typography variant="body2" color="textSecondary">
              Please upload the patient's MRI image.
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      {/* Button to send the file and generate report */}
      <Grid item xs={12}>
        <ColorButton variant="contained" onClick={sendFile} disabled={isLoading}>
          {isLoading ? <CircularProgress size={24} className={classes.loader} /> : 'Generate Report'}
        </ColorButton>
      </Grid>

      {/* Analysis Section */}
      <Grid item xs={12}>
        {data && (
          <Paper className={classes.detail}>
            <Typography variant="h6">Analysis Result</Typography>
            <Typography variant="body1" className={classes.analysisText}>
              Label: {data.label}<br />
              Confidence: {(parseFloat(data.confidence) * 100).toFixed(2)}%
            </Typography>
            <Typography variant="body1" className={classes.analysisText}>
              Analysis: {analysis}
            </Typography>
            <ColorButton className={classes.downloadButton} variant="contained" onClick={downloadPdf}>
              Download PDF Report
            </ColorButton>
          </Paper>
        )}
      </Grid>

      {/* Clear Data Button */}
      <Grid item xs={12}>
        <Button className={classes.clearButton} variant="contained" onClick={clearData}>
          Clear
        </Button>
      </Grid>
    </Grid>
  );
};

export default PatientsDataWithImageUpload;
