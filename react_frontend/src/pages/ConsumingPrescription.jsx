import React, { useEffect, useRef, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom';
import axios from "axios";
import Peer from 'peerjs'
import "../css/App.css"
import CryptoJS from 'crypto-js';
import $ from 'jquery';

function ConsumingPrescription() {

    const [deployed, setDeployed] = useState(false)
    const [myInfo, setMyInfo] = useState([])
    const peerInstance = useRef(null);
    const connectionInstance = useRef(null);
    const remoteVideoRef = useRef(null);
    const currentUserVideoRef = useRef(null);
    const [peerId, setPeerId] = useState('');
    // let message;
    // let allMessages = [
    //     { text: "Hello ", userType: "sender" },
    //     { text: "Hi there!", userType: "receiver" },
        
    //   ]

    const [allMessaages, setAllMessages] = useState([])
    const [allPrescriptions, setAllPrescriptions] = useState([])
      useEffect(() =>{          
          let userInfo = JSON.parse(sessionStorage.getItem('myInfo'))
        //   console.log(sha256(myInfo.email))
          const peer = new Peer(CryptoJS.SHA256(userInfo.email).toString());
        //   {
        //     host: import.meta.env.VITE_APP_PEER_HOST,
        //     port: import.meta.env.VITE_APP_PEER_PORT,
        //     path: import.meta.env.VITE_APP_PEER_PATH,
        //     secure: false,
        //     config: {
        //         allow_discovery: true,
        //         iceServers: [{ url: "stun:stun.l.google.com:19302" }],
        //     },
        // }

          peer.on('open', (id) => {
            console.log("My Peer id : ", id)
            setPeerId(id)
          });
          peer.on('connection', function(conn) {
            conn.on('data', function(data){
                if(data.userType === "sender"){
                    setAllMessages([...allMessaages, {
                        "userType" : "receiver",
                        "text" : data.text
                    }])
                    console.log(data);
                    console.log(allMessaages)
                }
                else if(data === "Accept"){
                    connectCall();
                }
                else{
                    handleReceivedImageBlob(data);
                }
              // Will print 'hi!'
            });
          });
          peer.on('call', (call) => {
            var getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
      
            getUserMedia({ video: true, audio:true}, (mediaStream) => {
            //   currentUserVideoRef.current.srcObject = mediaStream;
            //   currentUserVideoRef.current.play();
              
              call.answer(mediaStream)
              call.on('stream', function(remoteStream) {
                remoteVideoRef.current.srcObject = remoteStream
                remoteVideoRef.current.play();
              });
            });
          })
          peerInstance.current = peer;
          setMyInfo(JSON.parse(sessionStorage.getItem('myInfo')))
          console.log(userInfo)
          
          let isDeployed = sessionStorage.getItem("isDeployed")
          // if(isDeployed !== null)
          //     setDeployed(true)
          // if(isDeployed)
          getAllDoctors();
          getAllPrescriptions();
           
    }, [])

    const connectCall = () =>{
        var getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia;

    getUserMedia({ video: true, audio:true}, (mediaStream) => {

    //   currentUserVideoRef.current.srcObject = mediaStream;
    //   currentUserVideoRef.current.play();
     

      const call = peerInstance.current.call(CryptoJS.SHA256(doctorProfile[2]).toString(), mediaStream)

      call.on('stream', (remoteStream) => {
        remoteVideoRef.current.srcObject = remoteStream
        remoteVideoRef.current.play();
      });
    });
    }

    const connectToPeer = (remotePeerId) =>{
        var conn =  peerInstance.current.connect(remotePeerId);
            conn.on('open', function(){
                let mes = {
                    userType : "call",
                    text : CryptoJS.SHA256(myInfo.email).toString()
                }
                conn.send(mes);
            });
        connectionInstance.current = conn     
    }
    const [message, setMessagee] = useState({
        userType : "sender",
        text : ""
    })
    const setMessage = (mes) =>{
        setMessagee({"userType" : "sender", "text" : mes})
        // setAllMessages([...allMessaages, {"userType" : "sender", "text" : mes}])
    }
    
    const handleSendMessage = () =>{
        // setMessagee({"userType" : "sender", "text" : mes})
        setAllMessages([...allMessaages, {"userType" : "sender", "text" : message.text}])
        connectionInstance.current.send(message)
    }
    
    const [allDoctors, setAllDoctors] = useState([])
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


    const getAllPrescriptions = async () =>{
        let receivedResponse;
        await axios.get(process.env.REACT_APP_BACKEND_URL + '/getAllPrescriptions')
          .then((response) => {
            receivedResponse = response.data
          }, (error) => {
            console.log(error);
          });
          console.log(receivedResponse.Message)
          setAllPrescriptions(receivedResponse.Message)
    } 

    const navigate = useNavigate();
    let dprofile = []
    const [selectProfile, setSelectProfile] = useState(false)
    const [doctorProfile, setDoctorProfile] = useState([])
    const viewProfile = (profile) =>{
      setDoctorProfile(profile)
      dprofile = profile
      setSelectProfile(true)
      connectToPeer(CryptoJS.SHA256(profile[2]).toString())
      
    }
    
    
    
    const [image, setImage] = useState(null);

    const handleImageChange = (event) => {
        const selectedImage = event.target.files[0];
        setImage(selectedImage);
    };
    const [predictedPrescription, setPredictedPrescription] = useState("")
    const [predictedMedicine, setPredictedMedicine] = useState("")
    const predictPrescription = async () =>{
        setPredictedMedicine("Loading ...")
        const formData = new FormData();
        formData.append('image', image);
        let receivedResponse;
        await axios.post(process.env.REACT_APP_BACKEND_URL +'/predictPrescription', formData)
        .then((response) => {
          receivedResponse = response.data
        })
        .catch((error) => {
          console.error('Error:', error);
        });
        setPredictedMedicine("These are the medicines present in the prescription : " + (receivedResponse.Message).join(', '))
        console.log(receivedResponse)
    }

    const predictIndividualPrescription = async () =>{
        setPredictedMedicine("Loading ...")
        const formData = new FormData();
        formData.append('image', image);
        let receivedResponse;
        await axios.post(process.env.REACT_APP_BACKEND_URL +'/predictIndividualPrescription', formData)
        .then((response) => {
          receivedResponse = response.data
        })
        .catch((error) => {
          console.error('Error:', error);
        });
        setPredictedMedicine(receivedResponse.Message)
        console.log(receivedResponse)
    }

    const [symptoms, setSymptoms] = useState("")
    const [symptomResult, setSymptonResult] = useState("")
    const getSymptomResult = async () =>{
        setSymptonResult("Loading")
        console.log(symptoms)
        let receivedResponse;
        await axios.post(process.env.REACT_APP_BACKEND_URL +'/getMedicineForSymptoms', {
            symptoms : symptoms
        })
        .then((response) => {
          receivedResponse = response.data
          setSymptonResult(receivedResponse.Medicine)
        })
        .catch((error) => {
          console.error('Error:', error);
        });
        console.log(receivedResponse)
        setSymptonResult(receivedResponse.Message)
    }

    const [selectedValue, setSelectedValue] = useState('');

    const [diseaseResult, setDiseaseResult] = useState({})
  // Event handler for option selection
        const handleSelectChange = async (event) => {
            setSelectedValue(event.target.value);
            
            let receivedResponse;
            await axios.post(process.env.REACT_APP_BACKEND_URL +'/getPrescriptionDetails', {
                key : event.target.value
            })
            .then((response) => {
              receivedResponse = response.data
              setDiseaseResult(receivedResponse.Message)
            })
            .catch((error) => {
              console.error('Error:', error);
            });
            console.log(receivedResponse)
        };

        const fileInputRef = useRef();
        const [imageBlob, sendImageBlob] = useState("")
        function handleImageSelect(event) {
            const file = event.target.result;
            if (file) {
              const reader = new FileReader();
              reader.onload = function (e) {
                const imageBlob = new Blob([e.target.result], { type: file.type });
                // Now you can send the imageBlob through the PeerJS data channel
                handleReceivedImageBlob(imageBlob)
                sendImageBlob(imageBlob);
              };
              reader.readAsArrayBuffer(file);
            }
          }
          function handleReceivedImageBlob(imageBlob) {
            // Convert the Blob to a data URL for display
            const reader = new FileReader();
            reader.onload = function () {
              const imageDataURL = reader.result;
              // Display the received image data
              const receivedImage = document.createElement('img');
              receivedImage.src = imageDataURL;
              document.body.appendChild(receivedImage);
            };
            reader.readAsDataURL(imageBlob);
          }
          function sendImageBlobToPeer() {
            const fileInput = fileInputRef.current;
            const file = fileInput.files[0];
            const reader = new FileReader();

            reader.onload = function (e) {
                const imageData = e.target.result;
                
                connectionInstance.current.send(imageData)
              };
        
              reader.readAsDataURL(file);

          }

          const [receivedImage, setReceivedImage] = useState('');
        function handleReceivedImageBlob(data) {
            setReceivedImage(data);
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
                                            <span className="account-user-name">{myInfo.name}</span>
                                            <span className="account-position">Patient</span>
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
                                    <p style={{ fontSize: "20px" }}>Please Choose your Doctor</p> 
                                    
                                </div>
                            </div>
                        </div>
                        <br />
                        {!selectProfile && (
                          <div>
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
                                        <th>View Profile</th>
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
                                        <td><a className="action-icon" onClick={() => viewProfile(item)}> <i className="mdi mdi-eye"></i></a></td>
                                        {/* Add more columns to display other properties */}
                                    </tr>
                                    ))}
                                </tbody>
                            </table>
                            <br /><br />
                            <div className='row'>
                                <div className='col-6'>
                                    <h3>Enter your symptoms and get medicines name:</h3>
                                    <input type="text" value={symptoms} onChange={(e) => setSymptoms(e.target.value)} style={{width:"400px"}}/>
                                            <button
                                            className="btn btn-primary"
                                            onClick={getSymptomResult}
                                        >
                                            Ask
                                        </button>
                                        {(symptomResult !== "Loading" && symptomResult) ? (
                                            <h5>The Medicine for your Symptoms is : <p style={{color:'green'}}>{symptomResult}</p></h5>

                                        ) : (
                                            <h5>{symptomResult} ...</h5>
                                        )}

                                    <br /><br />
                                    

                                        <h3>Check individual Prescription:</h3>
                                    <input type="file"  id="fileToUpload" name='file' onChange={handleImageChange}/>
                                            <button
                                            className="btn btn-primary"
                                            onClick={predictIndividualPrescription}
                                        >
                                            Check
                                        </button>
                                        {(predictedMedicine !== ""  || predictedMedicine === "Loading ...")&& (
                                    <h3>{predictedMedicine}</h3>
                                 )}
                                </div>
                                <div className='col-4'>
                                <div className="mb-3">
                                    <label className="form-label">Select Symptoms or Disease to get your prescription</label>
                                    <select className="form-select" id="example-select" onChange={handleSelectChange} value={selectedValue}>
                                        <option>Select options</option>
                                        {/* Mapping over the options array to generate <option> elements */}
                                        {allPrescriptions.map((option) => (
                                        <option >
                                            {option[3]}
                                        </option>
                                        ))}
                                    </select>
                                </div>
                                <p>Prescribed by : {diseaseResult.PrescribedBy}</p>
                                <p>Doctor's Email : {diseaseResult.DoctorsEmail}</p>
                                <p>Disease : {diseaseResult.Prescription}</p>
                                <p>Prescription : {diseaseResult.Comments}</p>
                                <p>Additional Comments : {diseaseResult.Disease}</p>
                                </div>
                            </div>

                            </div>
                        )}
                        
                       {selectProfile && (
                        <div>
                            <div className='row'>
                           <div className=" col-xxl-5 col-xl-8 order-xl-1 order-xxl-3" style={{ "marginLeft":"100px" }}>
                                <div className="card">
                                    <div className="card-body">


                                        <div className="mt-3 text-center">
                                            <img src="/assets/images/user.jpg" alt="shreyu" className="img-thumbnail avatar-lg rounded-circle" />
                                            <h4>{doctorProfile[0]}</h4>
                                            
                                        </div>

                                        <div className="mt-3">
                                            <hr className="" />

                                            <p className="mt-4 mb-1"><strong><i className='uil uil-at'></i> Email:</strong></p>
                                            <p>{doctorProfile[2]}</p>

                                            <p className="mt-3 mb-1"><strong><i className='uil uil-phone'></i> Phone Number:</strong></p>
                                            <p>{doctorProfile[6]}</p>

                                           

                                            <p className="mt-4 mb-1"><strong> Age:</strong></p>
                                            <p>{doctorProfile[1]}</p>

                                            <p className="mt-4 mb-1"><strong> Hospital:</strong></p>
                                            <p>{doctorProfile[4]}</p>

                                            <p className="mt-4 mb-1"><strong> Specialization:</strong></p>
                                            <p>{doctorProfile[5]}</p>

                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            </div>
                            <div className='row'>
                                <div className='col-7'>
                                    <h5>Remote Video</h5>
                                    <video ref={remoteVideoRef} />
                                
                                </div>
                                <div className='col-3'>
                                <div className=" col-xxl-6 col-xl-8 order-xl-1 order-xxl-3" >
                            <div className="chat-box">
                                <div className="messages">
                                {allMessaages.map((msg, index) => (
                                <div
                                    key={index}
                                    className={`message ${msg.userType === 'sender' ? 'sender' : 'receiver'}`}
                                >
                                    {msg.text}
                                </div>
                                ))}
                                </div>
                                <div className="input">
                                    <input
                                    type="text"
                                    value={message.text}
                                    onChange={(e) => setMessage(e.target.value)}
                                    placeholder="Type your message..."
                                    />
                                    <button onClick={handleSendMessage}>Send</button>
                                </div>
                                <input
                                    type="file"
                                    ref={fileInputRef}
                                    onChange={(e) => handleImageSelect(e)}
                                    placeholder="Attach your wound"
                                    />
                                    <button onClick={sendImageBlobToPeer}>Send</button>
                                </div>
                                {receivedImage && <img src={receivedImage}  />}
                            </div>
                                </div>

                            </div>
                        </div>
                       )}

                    </div>
                </div>
            </div>
        </>
    )
}

export default ConsumingPrescription