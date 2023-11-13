import { useState } from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Link,
  useNavigate,
} from "react-router-dom";

import Login from "./pages/Login";
import Init from "./pages/Init";
import DoctorPrescription from "./pages/DoctorPrescription";
import ConsumingPrescription from "./pages/ConsumingPrescription";
import Admin from "./pages/Admin";
import Medical from "./pages/Medical";
// import '/src/assets/css/app.min.css'
// import '/src/assets/css/icons.min.css'
// import '/src/assets/css/app-dark.min.css'

function App() {
  return (
    <>
      <Router>
        <Routes>
          <Route path="/" element={<Init />} />
          <Route path="/login" element={<Login />} />
          <Route path="/admin" element={<Admin />} />
          <Route path="/medical" element={<Medical />} />
          <Route path="/doctorPrescription" element={<DoctorPrescription />} />
          <Route path="/consumingPrescription" element={<ConsumingPrescription />} />
        </Routes>
      </Router>
    </>
  );
}

export default App;
