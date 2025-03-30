import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import DownloadSection from './components/DownloadSection';
import FeaturesPage from './components/FeaturesPage';
import HowItWorksPage from './components/HowItWorksPage';
import LegalPage from './components/LegalPage';

const App: React.FC = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<DownloadSection />} />
        <Route path="/features" element={<FeaturesPage />} />
        <Route path="/how-it-works" element={<HowItWorksPage />} />
        <Route path="/legal" element={<LegalPage />} />
      </Routes>
    </Router>
  );
};

export default App;
