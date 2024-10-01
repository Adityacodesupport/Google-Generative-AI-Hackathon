// src/ReportViewer.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { makeStyles, withStyles } from "@material-ui/core/styles";
import AppBar from "@material-ui/core/AppBar";
import Toolbar from "@material-ui/core/Toolbar";
import Typography from "@material-ui/core/Typography";
import Container from "@material-ui/core/Container";
import Card from "@material-ui/core/Card";
import CardContent from "@material-ui/core/CardContent";
import { Grid, Button, CircularProgress, CardMedia } from "@material-ui/core";
import { DropzoneArea } from 'material-ui-dropzone';
import { common } from '@material-ui/core/colors';
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
  grow: {
    flexGrow: 1,
  },
  clearButton: {
    width: "100%",
    borderRadius: "15px",
    padding: "15px 22px",
    color: "#082567",
    fontSize: "18px",
    fontWeight: 600,
    backgroundColor: '#F8F0E3',
    textShadow: '1.1px 1.1px #36454F',
    marginTop: theme.spacing(2),
  },
  root: {
    maxWidth: 800,
    margin: "auto",
    marginTop: theme.spacing(4),
    padding: theme.spacing(2),
    backgroundColor: '#ffffff',
    boxShadow: '0px 9px 70px 0px rgb(0 0 0 / 30%)',
    borderRadius: '15px',
  },
  media: {
    height: 400,
    marginBottom: theme.spacing(2),
  },
  gridContainer: {
    justifyContent: "center",
    padding: theme.spacing(4),
  },
  mainContainer: {
    backgroundColor: '#f5f5f5',
    minHeight: "100vh",
    paddingTop: theme.spacing(8),
  },
  detail: {
    backgroundColor: '#F8F0E3',
    padding: theme.spacing(2),
    borderRadius: '15px',
  },
  appbar: {
    background: '#F8F0E3',
    boxShadow: 'none',
    color: '#082567',
  },
  loader: {
    color: '#082567 !important',
  },
  reportTitle: {
    fontWeight: 700,
    marginBottom: theme.spacing(2),
  },
  reportSubtitle: {
    fontWeight: 600,
    marginBottom: theme.spacing(1),
  },
  reportText: {
    marginBottom: theme.spacing(2),
    lineHeight: 1.6,
  },
  dropzoneText: {
    color: 'black',
  },
  mainTitle: {
    fontWeight: 700,
    fontSize: '24px',
    marginBottom: theme.spacing(4),
    textAlign: 'center',
  },
}));

const ReportViewer = () => {
  const classes = useStyles();
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [report, setReport] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    // Fetch report data from the backend
    axios.get('http://localhost:8000/api/report')
      .then(response => {
        setReport(response.data.report);
      })
      .catch(error => {
        console.error('There was an error fetching the report!', error);
      });
  }, []);

  const handleFileChange = (files) => {
    if (files.length > 0) {
      const selectedFile = files[0];
      setFile(selectedFile);
      const objectUrl = URL.createObjectURL(selectedFile);
      setPreview(objectUrl);
    }
  };

  const handleSubmit = async () => {
    if (!file) {
      alert("Please upload an image.");
      return;
    }

    setIsLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:8000/api/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setReport(response.data.report);
    } catch (error) {
      console.error('There was an error uploading the file!', error);
    } finally {
      setIsLoading(false);
    }
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
    <React.Fragment>
      <AppBar position="static" className={classes.appbar}>
        <Toolbar>
          <Typography className={classes.title} variant="h6" noWrap>
            Report Viewer
          </Typography>
          <div className={classes.grow} />
        </Toolbar>
      </AppBar>
      <Container maxWidth={false} className={classes.mainContainer}>
        <Grid
          className={classes.gridContainer}
          container
          direction="row"
          justifyContent="center"
          alignItems="center"
          spacing={2}
        >
          <Grid item xs={12}>
            <Card className={classes.root}>
              {preview && (
                <CardMedia
                  className={classes.media}
                  image={preview}
                  title="Uploaded Image"
                />
              )}
              <CardContent>
                <DropzoneArea
                  acceptedFiles={['image/*']}
                  dropzoneText={<span className={classes.dropzoneText}>Drag and drop an image to upload and analyze. Or click to upload.</span>}
                  onChange={handleFileChange}
                />
              </CardContent>
              <CardContent>
                {isLoading ? (
                  <CircularProgress className={classes.loader} />
                ) : (
                  <ColorButton className={classes.clearButton} onClick={handleSubmit}>
                    Upload and Analyze
                  </ColorButton>
                )}
              </CardContent>
            </Card>
          </Grid>
          {report && (
            <Grid item xs={12}>
              <Card className={classes.root}>
                <CardContent className={classes.detail} id="report-content">
                  <Typography variant="h4" className={classes.mainTitle}>
                    Medical Report Generated by Gemini
                  </Typography>
                  {renderReport(report)}
                  <ColorButton className={classes.clearButton} onClick={downloadReport}>
                    Download Report as PDF
                  </ColorButton>
                </CardContent>
              </Card>
            </Grid>
          )}
        </Grid>
      </Container>
    </React.Fragment>
  );
};

export default ReportViewer;