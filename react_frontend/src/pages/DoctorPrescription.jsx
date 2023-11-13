import axios from 'axios';
import React, { useEffect, useRef, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom';
import "../css/App.css"
import Peer from 'peerjs'
import CryptoJS from 'crypto-js';
import { Buffer } from 'buffer';
import $ from 'jquery';


function DoctorPrescription() {
    const navigate = useNavigate();
    const peerInstance = useRef(null);
    const connectionInstance = useRef(null);
    const remoteVideoRef = useRef(null);
    const currentUserVideoRef = useRef(null);
    const [peerId, setPeerId] = useState('');
    const [mediaRecorder, setMediaRecorder] = useState(null);
    const [isCall, setIsCall] = useState(true)
    
  const [audioChunks, setAudioChunks] = useState([]);
  const [audioBlob, setAudioBlob] = useState(null);
    const [modelMessage, setModalMessage] = useState({
        heading : "",
        message : ""
    })
    const [remotePeerId, setRemotePeerId] = useState("")
    // let message;
    // let allMessages = [
    //     { text: "Hello ", userType: "sender" },
    //     { text: "Hi there!", userType: "receiver" },
        
    //   ]
  
    const [recording, setRecording] = useState(false);
    const [mediaStream, setMediaStream] = useState(null);

    const startRecording = (stream) => {
        setMediaStream(stream); 
        const recorder = new MediaRecorder(stream);

        recorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            audioChunks.push(event.data);
          }
        };
        recorder.onstop = () => {
          const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
          setAudioBlob(audioBlob);
        };
        recorder.start();
        setMediaRecorder(recorder);
        setRecording(true);
      };
    
      const stopRecording = async () => {
        if (mediaRecorder && mediaRecorder.state === 'recording') {

          mediaRecorder.stop();
          setRecording(false);
          if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
            setMediaStream(null);
            const formData = new FormData();
            formData.append('speech', "/home/mktetts/Documents/codeshift/models/speech_to_text/data/recorded-audio.wav");
            let receivedResponse;
            await axios.post(process.env.REACT_APP_BACKEND_URL +'/speechToText', formData)
            .then((response) => {
              receivedResponse = response.data
            })
            .catch((error) => {
              console.error('Error:', error);
            });
          }
        }
      };

    const [allMessaages, setAllMessages] = useState([])
    const [myInfo, setMyInfo] = useState([])
    useEffect(() =>{
        let userInfo = JSON.parse(sessionStorage.getItem('myInfo'))
        // $('#bs-example-modal-sm').modal('hide');
        setMyInfo(JSON.parse(sessionStorage.getItem('myInfo')))
        console.log(userInfo)

        const peer = new Peer(CryptoJS.SHA256(userInfo[2]).toString());
        // {
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
              if (data.userType === "call"){
                  setRemotePeerId(data.text)
                  // modal.heading = "Incoming Call!!!"
                  // modal.message = "The person with ID" + data + "Wants to connect with you!!!"
                  setModalMessage({
                      heading : "Incoming Call !!!",
                      message : "The person with ID" + data.text + "Wants to connect with you!!!"
                  })
                  $('#top-modal').modal('show');
              }
              else if(data.userType === "sender"){
                  setAllMessages([...allMessaages, {
                      "userType" : "receiver",
                      "text" : data.text
                    }])
                    console.log(allMessaages)
                }
                
                else {
                    // Assuming the received data is a Blob
                    handleReceivedImageBlob(data);
                  }
          });
        });

        peer.on('call', (call) => {
            var getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
      
            getUserMedia({ video: true, audio:true}, (mediaStream) => {
            //   currentUserVideoRef.current.srcObject = mediaStream;
            //   currentUserVideoRef.current.play();
              
              call.answer(mediaStream)
              call.on('stream', function(remoteStream) {
                stream = remoteStream;
                remoteVideoRef.current.srcObject = remoteStream
                remoteVideoRef.current.play();
              });
            });
          })

        peerInstance.current = peer;
    }, [])

    const acceptCall = (reply) =>{
        setIsCall(false)
        var conn =  peerInstance.current.connect(remotePeerId);
        conn.on('open', function(){
            conn.send(reply);
        });
        
        connectionInstance.current = conn  
        if(reply === "Accept")
            connectCall();   
    }
    const connectCall = () =>{
        var getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
        getUserMedia({ video: true, audio:true}, (mediaStream) => {
            const call = peerInstance.current.call(remotePeerId, mediaStream)
            call.on('stream', (remoteStream) => {
                startRecording(remoteStream);
            remoteVideoRef.current.srcObject = remoteStream
            remoteVideoRef.current.play();
        });
        });
    }
   
    const stopMedia = () => {
        if (mediaStream) {
          mediaStream.getTracks().forEach(track => track.stop());
          setMediaStream(null);
        }
      };
    const [prescriptionForm, setPrescriptionForm] = useState({
        name : myInfo[0],
        email : myInfo[2],
        disease : "Muthu",
        prescription : "23",
        comments : "",
      });
      const handlePrescriptionFormChange = (e) => {
        const { name, value } = e.target;
        setPrescriptionForm((prevForm) => ({
          ...prevForm,
          [name]: value,
        }));
      };
      const [image, setImage] = useState(null);
      const [imageSrc, setImageSrc] = useState('');
      const SubmitPrescription = async () =>{
        setIsCall(false)
        let userInfo = JSON.parse(sessionStorage.getItem('myInfo'))
        setModalMessage({
            heading:"Adding Precription",
            message : "Prescription Details are being added to Blockchain..."
        })
        let presForm = {
            ...prescriptionForm,
            name : userInfo[0], email : userInfo[2]
        }
        setPrescriptionForm({...prescriptionForm, name : userInfo[0], email : userInfo[2]})
        $('#top-modal').modal('show');
        
        let receivedResponse;
        await axios.post(process.env.REACT_APP_BACKEND_URL +'/addPrescription', presForm, {
            responseType: 'arraybuffer',
        })
        .then((response) => {
          receivedResponse = response.data
          const base64Image = Buffer.from(response.data, 'binary').toString('base64');
            const imageSrc = `data:image/jpg;base64,${base64Image}`;
            
            // Set the image source in the state
            connectionInstance.current.send(imageSrc)
          $('#top-modal').modal('hide');
        })
        .catch((error) => {
          console.error('Error:', error);
        });
        console.log(receivedResponse)

      }
    const handleImageChange = (event) => {
        const selectedImage = event.target.files[0];
        setImage(selectedImage);
    };
    const [predictedWound, setPredictedWound] = useState("")
    const predictWound = async () =>{
        setPredictedWound("Loading...")
        const formData = new FormData();
        formData.append('image', image);
        let receivedResponse;
        await axios.post(process.env.REACT_APP_BACKEND_URL +'/predictWound', formData)
        .then((response) => {
          receivedResponse = response.data
        })
        .catch((error) => {
          console.error('Error:', error);
        });
        setPredictedWound(receivedResponse.Message)
        console.log(receivedResponse)
    }

    const [conn, setConn] = useState(undefined)
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

                                {/* <span style={{fontSize:'10px', color:'blue'}}>
                 asdasdasdas
                </span> */}

                            </a>

                            <ul className="list-unstyled topbar-menu float-end mb-0">
                                <li className="dropdown notification-list">
                                    <a className="nav-link dropdown-toggle nav-user arrow-none me-0" data-bs-toggle="dropdown" id="topbar-userdrop" href="#" role="button" aria-haspopup="true" aria-expanded="false">
                                        <span className="account-user-avatar">
                                            <img src="assets/images/user.jpg" alt="user-image" className="rounded-circle" />
                                        </span>
                                        <span>
                                            <span className="account-user-name">{myInfo[0]}</span>
                                            <span className="account-position">Doctor</span>
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
                                    <p style={{ fontSize: "20px" }}>Bringing hope to online patients one click at a time</p>
                                </div>
                            </div>
                        </div>
                        <br /><br />
                        <div className="row">
                            <div className="col-7">
                                <div className="mb-3">
                                    <label className="form-label">Enter Disease Name</label>
                                    <input type="text" id="simpleinput" className="form-control" name='disease' onChange={handlePrescriptionFormChange} value={prescriptionForm.disease} autoComplete='off'/>
                                </div>

                                <div className="mb-3">
                                    <label className="form-label">Enter Your Prescription</label>
                                    <input type="text" id="simpleinput" className="form-control" name='prescription' onChange={handlePrescriptionFormChange} value={prescriptionForm.prescription} autoComplete='off'/>
                                </div>
                                <div className="form-floating">
                                    <textarea className="form-control" placeholder="Leave a comment here" id="floatingTextarea" style={{ "height": "250px" }} name='comments' onChange={handlePrescriptionFormChange} value={prescriptionForm.comments} autoComplete='off'></textarea>
                                    <label>Additional Comments</label>
                                </div>
                                <br />
                                <button
                                     className="btn btn-primary"
                                     onClick={SubmitPrescription}
                                 >
                                     Submit Prescription
                                 </button>
                                <br /><br />
                                    <input type="file"  id="fileToUpload" name='file' onChange={handleImageChange}/>
                                    <button
                                     className="btn btn-primary"
                                     onClick={predictWound}
                                 >
                                     Predict
                                 </button>
                                 {predictedWound !== "" && (
                                    <h3>{predictedWound}</h3>
                                 )}
                                
                            </div>
                            <div className=" col-xxl-5 col-xl-7 order-xl-1 order-xxl-3" style={{ "paddingLeft": "140px" }}>
                                <div className="card">
                                    <div className="card-body">


                                        <div className="mt-3 text-center">
                                            <img src="/assets/images/user.jpg" alt="shreyu" className="img-thumbnail avatar-lg rounded-circle" />
                                            <h4>{myInfo[0]}</h4>
                                            
                                        </div>

                                        <div className="mt-3">
                                            <hr className="" />

                                            <p className="mt-4 mb-1"><strong><i className='uil uil-at'></i> Email:</strong></p>
                                            <p>{myInfo[2]}</p>

                                            <p className="mt-3 mb-1"><strong><i className='uil uil-phone'></i> Phone Number:</strong></p>
                                            <p>{myInfo[6]}</p>

                                           

                                            <p className="mt-4 mb-1"><strong> Age:</strong></p>
                                            <p>{myInfo[1]}</p>

                                            <p className="mt-4 mb-1"><strong> Hospital:</strong></p>
                                            <p>{myInfo[4]}</p>

                                            <p className="mt-4 mb-1"><strong> Specialization:</strong></p>
                                            <p>{myInfo[5]}</p>

                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                          
                        </div>
                        {!isCall && (
                            <>
                        <div className='row'>
                                <div className='col-7'>
                                    <h5>Remote Video</h5>
                                    <video ref={remoteVideoRef} />
                                    <button type="button" className="btn btn-primary" onClick={stopRecording}>End Call</button>
                                    {/* {audioBlob && (
        <a
          href={URL.createObjectURL(audioBlob)}
          download="recorded-audio.wav"
        >
          Download Recorded Audio
        </a>
      )} */}
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
                                </div>
                            </div>
                                </div>
                                {receivedImage && <img src={receivedImage}  />}
                            </div>
                            </>
                        )

                        }

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
                        {isCall && (
                        <div className="modal-footer">
                            <button type="button" className="btn btn-primary" onClick={() => acceptCall("Accept")} data-bs-dismiss="modal">Accept Call</button>
                            <button type="button" className="btn btn-primary" onClick={() => acceptCall("Reject")} data-bs-dismiss="modal">Reject Call</button>
                        </div>

                        )}
                    </div>
                </div>
                {imageSrc && <img src={imageSrc} alt="Fetched Image" />}
            </div>
        </>
    )
}

export default DoctorPrescription