import axios from 'axios';
import React, { useState } from 'react'
import { Link } from 'react-router-dom'

function Medical() {

    const [image, setImage] = useState(null);

    const handleImageChange = (event) => {
        const selectedImage = event.target.files[0];
        setImage(selectedImage);
    };

    const [res, setRes] = useState("")
    const predictIndividualPrescription = async () =>{
        // setPredictedMedicine("Loading ...")
        const formData = new FormData();
        formData.append('image', image);
        let receivedResponse;
        await axios.post(process.env.REACT_APP_BACKEND_URL +'/decodeQRCode', formData)
        .then((response) => {
          receivedResponse = response.data
        })
        .catch((error) => {
          console.error('Error:', error);
        });
        // setPredictedMedicine(receivedResponse.Message)
        console.log(receivedResponse.Message)
        if(receivedResponse.Message.hash === null){
            setRes("No Prescription found!!!")
        }
        else{
            setRes(JSON.stringify(receivedResponse.Message))
        }
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
                                    <span className="account-user-name">Medical Shop</span>
                                    <span className="account-position">Guest</span>
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
                            <p style={{ fontSize: "20px" }}>Upload the Transaction QR code to check</p>
                            <input type="file"  id="fileToUpload" name='file' onChange={handleImageChange}/>
                                            <button
                                            className="btn btn-primary"
                                            onClick={predictIndividualPrescription}
                                        >
                                            Check
                                        </button>

                            <h5>{res}</h5>
                        </div>
                    </div>
                </div>
                <br />
               

            </div>
        </div>
    </div>
   
</>
  )
}

export default Medical