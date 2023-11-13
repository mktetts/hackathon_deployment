// SPDX-License-Identifier: MIT
pragma solidity >=0.4.22 <0.9.0;

contract DoctorDetailContract {
    uint256 public doctorCount = 0;

    constructor() {}

    struct Doctor {
        string name;
        string age;
        string email;
        string qualification;
        string hospital;
        string specialist;
        string phone;
        string password;
        
    }

    Doctor[] public doctor;

    event addDoctors(Doctor _doctor, address account);

    function addDoctor(Doctor memory _doctor) public {
        doctor.push(_doctor);
        doctorCount++;
        emit addDoctors(_doctor, msg.sender);
    }

    event updateDoctors(uint256 _id, Doctor _doctor, address account);

    function updateDoctor(uint256 _id, Doctor memory _doctor) public {
        doctor[_id] = _doctor;

        emit updateDoctors(_id, _doctor, msg.sender);
    }

    event deleteDoctors(Doctor _doctor, address account);

    function deleteDoctor(uint256 _id) public {
        Doctor memory u = doctor[_id];
        if (doctorCount == 1) {
            doctor.pop();
        } else {
            doctor[_id] = doctor[doctorCount - 1];
            doctor.pop();
        }
        doctorCount--;

        emit deleteDoctors(u, msg.sender);
    }

    function getDoctors() public view returns (Doctor[] memory) {
        return doctor;
    }
}
