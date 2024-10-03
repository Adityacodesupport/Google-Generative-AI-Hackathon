// Import necessary dependencies
import React, { useState } from 'react'; 
import Home from './home'; // Home component: form and image upload
import PatientsList from './patientslist'; // Patients list component: displays patient records
import Navbar from './navbar'; // Navbar component: handles navigation
import ReportViewer from './ReportViewer'; // (optional) Report viewer for patient analysis reports

function App() {
  // State to manage the currently active view, default is 'home'
  const [currentView, setCurrentView] = useState('home');

  // Function to handle navigation between views
  const handleNavigate = (view) => {
    setCurrentView(view); // Set the current view based on user selection
  };

  // Function to conditionally render the current view based on state
  const renderCurrentView = () => {
    switch (currentView) {
      case 'home': // If current view is 'home', render Home component
        return (
          <>
            <Home />
            {/* Uncomment below to include ReportViewer on the Home page */}
            {/* <ReportViewer /> */}
          </>
        );

      case 'patients': // If current view is 'patients', render PatientsList component
        return <PatientsList />;

      case 'about': // If current view is 'about', render static content
        return <div>About Page Content</div>;

      default: // Default fallback to Home component if no match
        return <Home />;
    }
  };

  // Render the Navbar and the selected view
  return (
    <>
      {/* Navbar component: provides navigation options */}
      <Navbar onNavigate={handleNavigate} />
      
      {/* Render the current view based on user navigation */}
      {renderCurrentView()}
    </>
  );
}

export default App; // Export the App component as default
