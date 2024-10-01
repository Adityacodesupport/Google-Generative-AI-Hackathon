// App.js
import React, { useState } from 'react';
import Home from './home'; // Home page (form and image upload)
import PatientsList from './patientslist'; // Patients list page
import Navbar from './navbar';
import ReportViewer from './ReportViewer';

function App() {
  const [currentView, setCurrentView] = useState('home'); // State to manage the current view

  const handleNavigate = (view) => {
    setCurrentView(view);
  };

  const renderCurrentView = () => {
    switch (currentView) {
      case 'home':
        return (
          <>
            <Home />
            {/* <ReportViewer /> */}
          </>
        );
 
      case 'patients':
        return <PatientsList />;
      case 'about':
        return <div>About Page Content</div>;
      default:
        return <Home />;
    }
  };

  return (
    <>
      <Navbar onNavigate={handleNavigate} />
      {renderCurrentView()} {/* Render the current view */}
    </>
  );
}

export default App;
