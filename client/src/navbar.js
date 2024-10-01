// navbar.js
import React from 'react';
import { AppBar, Toolbar, Typography, Button, Avatar } from '@material-ui/core';
import iS from './components/logo.jpg';
import { makeStyles } from '@material-ui/core/styles';

const useStyles = makeStyles((theme) => ({
  appbar: {
    background: '#F8F0E3',
    boxShadow: 'none',
    color: '#082567',
  },
  grow: {
    flexGrow: 1,
  },
  button: {
    marginLeft: theme.spacing(2),
    color: '#082567',
    fontWeight: 'bold',
  },
}));

const Navbar = ({ onNavigate }) => {
  const classes = useStyles();

  return (
    <AppBar position="static" className={classes.appbar}>
      <Toolbar>
        <Typography variant="h6" noWrap>
          Brain Tumor Diagnosis
        </Typography>
        <div className={classes.grow} />
        <Button className={classes.button} onClick={() => onNavigate('home')}>
          Home
        </Button>
        <Button className={classes.button} onClick={() => onNavigate('patients')}>
          Patients
        </Button>
        <Button className={classes.button} onClick={() => onNavigate('about')}>
          About
        </Button>
        <Avatar src={iS} />
      </Toolbar>
    </AppBar>
  );
};

export default Navbar;
