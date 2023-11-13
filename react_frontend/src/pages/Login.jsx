import React, { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom';
import axios from "axios";
import $ from 'jquery';

function Login() {

	useEffect(() => {
		document.body.classList.add("authentication-bg");
		return () => {
			document.body.classList.remove("authentication-bg");
		};
	}, [])
	const navigate = useNavigate();


	const [role, setRole] = useState(false)
	const [userForm, setUserForm] = useState({
		email: "mku@gmail.com",
		password: "mku@gmail.com@12345",
		role : "",
	});
	const handleUserFormChange = (e) => {
		const { name, value } = e.target;
		setUserForm((prevForm) => ({
			...prevForm,
			[name]: value,
		}));
	};

	const [patientForm, setPatientForm] = useState({
		name : "Patient1",
		email: "patient@gmail.com",
		card : "1234567890123456",
		amount : 1000
	});
	const handlePatientFormChange = (e) => {
		const { name, value } = e.target;
		setPatientForm((prevForm) => ({
			...prevForm,
			[name]: value,
		}));
	};
	const handleLogin = async () => {
		
		console.log(userForm)
		let receivedResponse
		if(userForm.role === "Admin"){
			// $('#top-modal').modal('hide');
			navigate('/admin')

		}
		else if(userForm.role === "Doctor"){
			$('#top-modal').modal('show');
			await axios.post(process.env.REACT_APP_BACKEND_URL + '/login', userForm)
			.then((response) => {
			  receivedResponse = response
			}, (error) => {
			  console.log(error);
			});
			console.log(receivedResponse.data)
			if(receivedResponse.data.Status === "Success"){
				$('#top-modal').modal('hide');
				sessionStorage.setItem("myInfo", JSON.stringify(receivedResponse.data.Message))
				navigate('/doctorPrescription')
			}

		}
		else if(userForm.role === "Patient"){
			$('#top-modal').modal('show');
			let receivedResponse;
			await axios.post(process.env.REACT_APP_JAVA_BACKEND_URL, patientForm)
			.then((response) => {
			  receivedResponse = response
			}, (error) => {
			  console.log(error);
			});
			console.log(receivedResponse)

			// await axios.get(process.env.REACT_APP_JAVA_BACKEND_URL + '/getAllPatients', patientForm)
			// .then((response) => {
			//   receivedResponse = response
			// }, (error) => {
			//   console.log(error);
			// });
			// console.log(receivedResponse)
			$('#top-modal').modal('hide');
			sessionStorage.setItem("myInfo", JSON.stringify(patientForm))
			navigate('/consumingPrescription')

		}
		else{
			navigate('/medical')
		}	
	}
	return (
		<div className="account-pages pt-2 pt-sm-5 pb-4 pb-sm-5">
			<div className="container">
				<div className="row justify-content-center">
					<div className="col-xxl-4 col-lg-5">
						<div className="card">
							<div className="card-header pt-4 pb-4 text-center bg-primary">
								<h5
									style={{
										color: "#fff",
										fontSize: "25px",
									}}
								>
									Vino Pharmacy Shift
								</h5>
							</div>
							<div className="card-body p-4">
								<div className="text-center w-75 m-auto">
									<h4 className="text-dark-50 text-center pb-0 fw-bold">
										For Enter..
									</h4>
									<p className="text-muted mb-4">
										Enter your email address and password.
									</p>
								</div>
								<form>
									{(userForm.role !== "Admin" && userForm.role !== "" && userForm.role !== "Patient" && userForm.role !== "MedicalShop") && (
										<div>
											<div className="mb-3">
										<label className="form-label">
											Email address
										</label>
										<input
											name="email"
											autoComplete="off"
											className="form-control"
											placeholder="Enter your email"
											value={userForm.email}
											onChange={handleUserFormChange}
										/>
									</div>
									<div className="mb-3">
										<a
											href="pages-recoverpw.html"
											className="text-muted float-end"
										>
											{/* <small>Forgot your password?</small> */}
										</a>
										<label className="form-label">
											Password
										</label>
										<div className="input-group input-group-merge">
											<input
												name="password"
												id="password"
												type="password"
												className="form-control"
												placeholder="Enter your password"
												value={userForm.password}
												onChange={handleUserFormChange}
											/>
										</div>


									</div>

										</div>
									)}
									{(userForm.role === "Patient" && userForm.role !== "MedicalShop") && (
										<div>
											<div className="mb-3">
										<label className="form-label">
											Name
										</label>
										<input
											name="name"
											autoComplete="off"
											className="form-control"
											placeholder="Enter your name"
											value={patientForm.name}
											onChange={handlePatientFormChange}
										/>
									</div>
									<div className="mb-3">
										<label className="form-label">
											Email
										</label>
										<input
											name="email"
											autoComplete="off"
											className="form-control"
											placeholder="Enter your emmail"
											value={patientForm.email}
											onChange={handlePatientFormChange}
										/>
									</div>
									<div className="mb-3">
										<label className="form-label">
											Card No for Autopay
										</label>
										<input
											name="card"
											autoComplete="off"
											className="form-control"
											placeholder="Enter your card number"
											value={patientForm.card}
											onChange={handlePatientFormChange}
										/>
									</div>
										</div>
									)}
									<div className="mb-3">

										<label className="form-label">
											Select Role
										</label>
										<select className="form-select mb-3" onChange={handleUserFormChange} name='role'>
											<option >Select role </option>
											<option value="Admin">Admin</option>
											<option value="Doctor">Doctor</option>
											<option value="Patient">Patient</option>
											<option value="MedicalShop">Medical Shop</option>
										</select>


									</div>
								</form>

								<div className="mb-3 mb-0 text-center">
									<button
										className="btn btn-primary"
										onClick={handleLogin}
									>
										Enter
									</button>
								</div>
							</div>
						</div>
					</div>
				</div>
			</div>
			<div id="top-modal" className="modal fade" tabIndex="-1" role="dialog" aria-hidden="true">
                <div className="modal-dialog modal-top">
                    <div className="modal-content">
                        <div className="modal-header">
                            {/* <h4 className="modal-title" id="topModalLabel">{modelMessage.heading}</h4> */}
                            {/* <button type="button" className="btn-close" data-bs-dismiss="modal" aria-hidden="true"></button> */}
                        </div>
                        <div className="modal-body" style={{ wordWrap: 'break-word', textAlign:"center" }}>
                            Logging In ....
                        </div>
                        
                    </div>
                </div>
            </div>
		</div>
	)
}

export default Login