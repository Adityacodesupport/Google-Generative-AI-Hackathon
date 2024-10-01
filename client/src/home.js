// home.js
import React from 'react';
import { Container, Grid } from '@material-ui/core';
import { makeStyles } from '@material-ui/core/styles';
import PatientsDataWithImageUpload from './patientrecords'; // Your form component
import backgroundImage from './components/backgroundimg.jpeg';

const useStyles = makeStyles((theme) => ({
  mainContainer: {
    backgroundImage: `url(${backgroundImage})`,
    backgroundRepeat: 'no-repeat',
    backgroundPosition: 'center',
    backgroundSize: 'cover',
    minHeight: '100vh',
    marginTop: '8px',
    padding: theme.spacing(2), // Adds padding to the entire container
  },
  contentContainer: {
    flexGrow: 1,
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    padding: theme.spacing(2),
  },
  uploadContainer: {
    margin: theme.spacing(2), // Add margin to the PatientsDataWithImageUpload component
    backgroundColor: 'rgba(255, 255, 255, 0.8)', // Optional: light background for better visibility
    borderRadius: '8px', // Optional: rounded corners
  },
}));

const Home = () => {
  const classes = useStyles();

  return (
    <Container maxWidth={false} className={classes.mainContainer} disableGutters={true}>
      <div className={classes.contentContainer}>
        <Grid container direction="row" justifyContent="center" alignItems="center" spacing={2}>
          <Grid item xs={12} className={classes.uploadContainer}>
            <PatientsDataWithImageUpload />
          </Grid>
        </Grid>
      </div>
    </Container>
  );
};

export default Home;
