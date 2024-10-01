// patientslist.js
import React, { useEffect, useState } from 'react';
import { Container, Grid, Paper, Typography, CircularProgress, Button } from '@material-ui/core';
import { makeStyles } from '@material-ui/core/styles';
import axios from 'axios';
import jsPDF from 'jspdf';

const useStyles = makeStyles((theme) => ({
  root: {
    padding: theme.spacing(3),
  },
  paper: {
    padding: theme.spacing(2),
    marginBottom: theme.spacing(2),
  },
  title: {
    fontWeight: 'bold',
    marginBottom: theme.spacing(1),
  },
  loader: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    height: '100vh', // Center loader vertically
  },
  button: {
    marginTop: theme.spacing(2),
  },
}));

const PatientsList = () => {
  const classes = useStyles();
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchPatients = async () => {
      try {
        const response = await axios.get('http://localhost:8000/patient');
        setPatients(response.data); // Set fetched data
      } catch (error) {
        setError('Failed to load patients data.'); // Handle error
        console.error(error);
      } finally {
        setLoading(false); // Hide loading spinner
      }
    };

    fetchPatients(); // Call the fetch function on component mount
  }, []);

  const downloadReport = (patient) => {
    const doc = new jsPDF();
    doc.text(`Patient Report for ${patient.name}`, 10, 10);
    doc.text(`Age: ${patient.age}`, 10, 20);
    doc.text(`Gender: ${patient.gender}`, 10, 30);
    doc.text(`Diagnosis: ${patient.report}`, 10, 40);
    doc.save(`${patient.name}_report.pdf`);
  };

  if (loading) {
    return (
      <div className={classes.loader}>
        <CircularProgress />
      </div>
    );
  }

  if (error) {
    return (
      <Container className={classes.root}>
        <Typography variant="h6" color="error">
          {error}
        </Typography>
      </Container>
    );
  }

  return (
    <Container className={classes.root}>
      <Typography variant="h4" gutterBottom>
        Patients List
      </Typography>
      <Grid container spacing={3}>
        {patients.map((patient) => (
          <Grid item xs={12} key={patient._id}>
            <Paper className={classes.paper}>
              <Typography variant="h6" className={classes.title}>
                {patient.name}
              </Typography>
              <Typography>Age: {patient.age}</Typography>
              <Typography>Gender: {patient.gender}</Typography>
              <Typography>Diagnosis: {patient.report}</Typography>
              <Button
                variant="contained"
                color="primary"
                className={classes.button}
                onClick={() => downloadReport(patient)}
              >
                Download Report
              </Button>
            </Paper>
          </Grid>
        ))}
      </Grid>
    </Container>
  );
};

export default PatientsList;
