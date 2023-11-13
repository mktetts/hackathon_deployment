import React, { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom';
import axios from "axios";
import $ from 'jquery';

function Admin() {

    const [deployed, setDeployed] = useState(false)
    const [cid, setCid] = useState("QmPzoqWhDxgVhwyFAj638LQReXuZUtLByoBE6VUEDJAudg")
    const [modelMessage, setModalMessage] = useState({
        heading : "",
        message : ""
    })
    useEffect(() =>{
        let isDeployed = sessionStorage.getItem("isDeployed")
        if(isDeployed !== null)
            setDeployed(true)
        if(isDeployed){
            getAllDoctors();
            getAllPatients();
        }
           
    }, [])
    const [allDoctors, setAllDoctors] = useState([])
    const [allPatients, setAllPatients] = useState([])
    const getAllPatients = async () =>{
        let receivedResponse;
        await axios.get(process.env.REACT_APP_JAVA_BACKEND_URL + '/getAllPatients')
			.then((response) => {
			  receivedResponse = response
			}, (error) => {
			  console.log(error);
			});
			setAllPatients(receivedResponse.data)
    }
    const getAllDoctors = async () =>{
        let receivedResponse;
        await axios.get(process.env.REACT_APP_BACKEND_URL + '/getAllDoctors')
          .then((response) => {
            receivedResponse = response
          }, (error) => {
            console.log(error);
          });
          setAllDoctors(receivedResponse.data.Message)
    } 

    const navigate = useNavigate();
    const [doctorForm, setDoctorForm] = useState({
        name : "Muthu",
        age : "23",
        email : "mku",
        qualification : "asdsad",
        hospital : "asdsad",
        specialist : "asdsad",
        phone : "+919894591458"                                             
      });
      const handleDoctorFormChange = (e) => {
        const { name, value } = e.target;
        setDoctorForm((prevForm) => ({
          ...prevForm,
          [name]: value,
        }));
      };
      const handleSubmit = async () => {
        console.log(doctorForm)
        setModalMessage({
            heading:"Adding Doctors",
            message : "Doctor Details are being added to Blockchain..."
        })
        $('#top-modal').modal('show');
        let receivedResponse;
        await axios.post(process.env.REACT_APP_BACKEND_URL + '/addDoctor', doctorForm)
          .then((response) => {
            receivedResponse = response
            $('#top-modal').modal('hide');
          }, (error) => {
            console.log(error);
          });
          console.log(receivedResponse.data)
        // navigate('/admin')
      }

      const handleDeploy = async () =>{
        setModalMessage({
            heading:"Deployment",
            message : "Smart contracts are being deployed..."
        })
        $('#top-modal').modal('show');
        let receivedResponse;
        let cidObject = {
            cid : cid
        }
        await axios.post(process.env.REACT_APP_BACKEND_URL + '/deployContract', cidObject)
          .then((response) => {
            receivedResponse = response
            $('#top-modal').modal('hide');
            setDeployed(true)
            sessionStorage.setItem("isDeployed", "deployed")
          }, (error) => {
            console.log(error);
          });
          console.log(receivedResponse.data)
      }
    return (
        <>
            <div className="content-page">
                <div className="content">
                    <div
                        className="navbar-custom topnav-navbar"
                        style={{ paddingLeft: "80px", paddingRight: "80px" }}
                    >
                        <div className="container-fluid">
                            <a href="" className="topnav-logo">
                                <span style={{ fontSize: '20px', color: 'blue' }}>
                                    Vino Pharmacy Shift
                                </span>



                            </a>

                            <ul className="list-unstyled topbar-menu float-end mb-0">
                                <li className="dropdown notification-list">
                                    <a className="nav-link dropdown-toggle nav-user arrow-none me-0" data-bs-toggle="dropdown" id="topbar-userdrop" href="#" role="button" aria-haspopup="true" aria-expanded="false">
                                        <span className="account-user-avatar">
                                            <img src="assets/images/user.jpg" alt="user-image" className="rounded-circle" />
                                        </span>
                                        <span>
                                            <span className="account-user-name">Admin</span>
                                            <span className="account-position">Network Admin</span>
                                        </span>
                                    </a>
                                    <div className="dropdown-menu dropdown-menu-end dropdown-menu-animated topbar-dropdown-menu profile-dropdown" aria-labelledby="topbar-userdrop">

                                        <div className=" dropdown-header noti-title">
                                            <h6 className="text-overflow m-0">Welcome !</h6>
                                        </div>




                                        <Link className="dropdown-item notify-item" to='/'>
                                            <i className="mdi mdi-logout me-1"></i>
                                            <span>Logout</span>

                                        </Link>

                                    </div>
                                </li>
                            </ul>
                        </div>
                    </div>

                   
                    <div
                        className="container-fluid"
                        style={{ paddingLeft: "90px", paddingRight: "80px" }}
                    >
                        <div className="row">
                            <div className="col-12">
                                <div className="page-title-box">
                                    <h4></h4>
                                </div>
                            </div>
                        </div>
                        <div className="row">
                            <div className="col-12 d-flex justify-content-center align-items-center">
                                <div className="page-title-box text-center">
                                    <h4 style={{ fontSize: "30px" }}>Welcome to the Vino Pharmacy Shift</h4>
                                    <p style={{ fontSize: "25px" }}>Providing healthcare solutions with a personal touch</p>
                                    {!deployed ? <p style={{ fontSize: "20px" }}>Looks like Smart Contracts not initialized</p> : <p style={{ fontSize: "20px" }}>Add Doctors to the Pharmacy Network</p>}
                                    
                                </div>
                            </div>
                        </div>
                        <br />
                        {!deployed &&(
                            <div>
                                 <input type="text" value={cid} onChange={(e) => setCid(e.target.value)} style={{width:"430px", marginLeft:"450px"}} placeholder='Enter the Medicine file CID'/>
                                 <br /><br/>
                            <div className="mb-3 mb-0 text-center">
                            <button
                                className="btn btn-primary"
                                onClick={handleDeploy}
                                style={{padding: '15px 30px' }}
                                disabled={cid === ''}
                            >
                                Deploy Smart Contracts
                            </button>
                        </div>
                            </div>
                        )}
                       
                        {deployed && (
                            <div>
                             <div className="row" style={{ justifyContent: 'center' }}>
                             <div className="col-5">
                                 <div className="mb-3">
                                     <label className="form-label">Doctor's Name</label>
                                     <input type="text" id="simpleinput" className="form-control" autoComplete='off' name="name" value={doctorForm.name} onChange={handleDoctorFormChange}/>
                                 </div>
 
                                 <div className="mb-3">
                                     <label className="form-label">Doctor's Age</label>
                                     <input type="text" id="simpleinput" className="form-control" autoComplete='off' name="age" value={doctorForm.age} onChange={handleDoctorFormChange}/>
                                 </div>
                                 <div className="mb-3">
                                     <label className="form-label">Doctor's Email</label>
                                     <input type="text" id="simpleinput" className="form-control" autoComplete='off' name="email" value={doctorForm.email} onChange={handleDoctorFormChange}/>
                                 </div>
                                 <div className="mb-3">
                                     <label className="form-label">Doctor's Qualification</label>
                                     <input type="text" id="simpleinput" className="form-control"autoComplete='off' name="qualification" value={doctorForm.qualification} onChange={handleDoctorFormChange}/>
                                 </div>
                                 <div className="mb-3">
                                     <label className="form-label">Doctor's Hospital</label>
                                     <input type="text" id="simpleinput" className="form-control" autoComplete='off' name="hospital" value={doctorForm.hospital} onChange={handleDoctorFormChange}/>
                                 </div>
                                 <div className="mb-3">
                                     <label className="form-label">Doctor's Specialized Area</label>
                                     <input type="text" id="simpleinput" className="form-control" autoComplete='off' name="specialist" value={doctorForm.specialist} onChange={handleDoctorFormChange}/>
                                 </div>
                                 <div className="mb-3">
                                     <label className="form-label">Doctor's Phone Number</label>
                                     <input type="text" id="simpleinput" className="form-control" autoComplete='off' name="phone" value={doctorForm.phone} onChange={handleDoctorFormChange}/>
                                 </div>
 
                             </div>
                             <div className="mb-3 mb-0 text-center">
                                 <button
                                     className="btn btn-primary"
                                     onClick={handleSubmit}
                                 >
                                     Submit
                                 </button>
                             </div>
 
                         </div>
                         <div style={{borderStyle:'groove', padding:'10px 10px 10px 10px'}}>
                         <ul className="nav nav-pills bg-nav-pills nav-justified mb-3">
                            <li className="nav-item">
                                <a href="#home1" data-bs-toggle="tab" aria-expanded="false" className="nav-link rounded-0 active">
                                    <i className="mdi mdi-home-variant d-md-none d-block"></i>
                                    <span className="d-none d-md-block">Doctor's List</span>
                                </a>
                            </li>
                            <li className="nav-item">
                                <a href="#profile1" data-bs-toggle="tab" aria-expanded="true" className="nav-link rounded-0 ">
                                    <i className="mdi mdi-account-circle d-md-none d-block"></i>
                                    <span className="d-none d-md-block">Patient's List</span>
                                </a>
                            </li>
                           
                        </ul>

                        <div className="tab-content">
                            <div className="tab-pane show active" id="home1">
                            <h3>List Of All Doctors</h3><br />
                            <table className="table table-striped table-centered mb-0">
                                <thead>
                                    <tr>
                                        <th>S.No</th>
                                        <th>Doctor' Name</th>
                                        <th>Age</th>
                                        <th>Email</th>
                                        <th>Qualification</th>
                                        <th>Hospital</th>
                                        <th>Specialist</th>
                                        <th>Phone Number</th>
                                    </tr>
                                </thead>
                                <tbody>
                                {allDoctors.map((item, index) => (
                                    <tr key={index}>
                                        <td>{index + 1}</td>
                                        <td>{item[0]}</td>
                                        <td>{item[1]}</td>
                                        <td>{item[2]}</td>
                                        <td>{item[3]}</td>
                                        <td>{item[4]}</td>
                                        <td>{item[5]}</td>
                                        <td>{item[6]}</td>
                                        {/* Add more columns to display other properties */}
                                    </tr>
                                    ))}
                                </tbody>
                            </table>
                            </div>
                            <div className="tab-pane " id="profile1">
                            <h3>List Of All Patients</h3><br />
                            <table className="table table-striped table-centered mb-0">
                                <thead>
                                    <tr>
                                        <th>S.No</th>
                                        <th>Patients's Id</th>
                                        <th>Patients's Name</th>
                                        <th>Email</th>
                                    </tr>
                                </thead>
                                <tbody>
                                {allPatients.map((item, index) => (
                                    <tr key={index}>
                                        <td>{index + 1}</td>
                                        <td>{item.id}</td>
                                        <td>{item.name}</td>
                                        <td>{item.email}</td>
                                       
                                        {/* Add more columns to display other properties */}
                                    </tr>
                                    ))}
                                </tbody>
                            </table>
                            </div>
                            
                        </div>
                         
                         </div>
                         </div>
                        )}

                    </div>
                </div>
            </div>
            <div id="top-modal" className="modal fade" tabIndex="-1" role="dialog" aria-hidden="true">
                <div className="modal-dialog modal-top">
                    <div className="modal-content">
                        <div className="modal-header">
                            <h4 className="modal-title" id="topModalLabel">{modelMessage.heading}</h4>
                            {/* <button type="button" className="btn-close" data-bs-dismiss="modal" aria-hidden="true"></button> */}
                        </div>
                        <div className="modal-body" style={{ wordWrap: 'break-word' }}>
                            {modelMessage.message}
                        </div>
                        
                    </div>
                </div>
            </div>
        </>
    )
}

export default Admin