package controller;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import model.Patient;
import service.PatientService;

@RestController
@RequestMapping(value = {"/patients"})
public class PatientController {
	
	@Autowired
	private PatientService patientService;
	
	@GetMapping(value = {"/getAllPatients"})
	public ResponseEntity<List<Patient>> findAll() {
		return new ResponseEntity<>(this.patientService.findAll(), HttpStatus.OK);
	}
	
	@GetMapping(value = {"/getPatientById/{patientId}"})
	public ResponseEntity<Patient> findById(@PathVariable("patientId") final String patientId) {
		return new ResponseEntity<>(this.patientService.findById(Long.parseLong(patientId)), HttpStatus.OK);
	}

	@GetMapping(value = {"/getPatientByEmail/{patientEmail}"})
	public ResponseEntity<Patient> findByEmail(@PathVariable("patientEmail") final String patientEmail) {
		return new ResponseEntity<>(this.patientService.findByEmail(patientEmail), HttpStatus.OK);
	}
	
	@PostMapping(value = {""})
	public ResponseEntity<?> save(@RequestBody Patient person) {
		this.patientService.save(person);
		return new ResponseEntity<>(HttpStatus.OK);
	}
	
	@PutMapping(value = {""})
	public ResponseEntity<?> update(@RequestBody final Patient person) {
		this.patientService.update(person);
		return new ResponseEntity<>(HttpStatus.OK);
	}
	
	@DeleteMapping(value = {"/deletePatientById/{patientId}"})
	public ResponseEntity<?> deleteById(@PathVariable("patientId") final String patientId) {
		this.patientService.deleteById(Long.parseLong(patientId));
		return new ResponseEntity<>(HttpStatus.OK);
	}
	@DeleteMapping(value = {"/deletePatientByEmail/{patientId}"})
	public ResponseEntity<?> deleteByEmail(@PathVariable("patientId") final String patientId) {
		this.patientService.deleteByEmail(patientId);
		return new ResponseEntity<>(HttpStatus.OK);
	}
	
	
}






