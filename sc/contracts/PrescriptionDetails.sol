// SPDX-License-Identifier: MIT
pragma solidity >=0.4.22 <0.9.0;

contract PrescriptionDetailContract {
    uint256 public prescriptionCount = 0;

    constructor() {}

    struct Prescription {
        string name;
        string email;
        string comments;
        string disease;
        string prescription;        
    }

    Prescription[] public prescription;

    event addPrescriptions(Prescription _prescription, address account);

    function addPrescription(Prescription memory _prescription) public {
        prescription.push(_prescription);
        prescriptionCount++;
        emit addPrescriptions(_prescription, msg.sender);
    }

    event updatePrescriptions(uint256 _id, Prescription _prescription, address account);

    function updatePrescription(uint256 _id, Prescription memory _prescription) public {
        prescription[_id] = _prescription;

        emit updatePrescriptions(_id, _prescription, msg.sender);
    }

    function getDoctors() public view returns (Prescription[] memory) {
        return prescription;
    }
}
