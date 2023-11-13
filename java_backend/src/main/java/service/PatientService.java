package service;

import java.util.List;

import model.Patient;

public interface PatientService {
	
	List<Patient> findAll();
	Patient findById(final Long personId);
	Patient findByEmail(final String email);
	void save(final Patient person);
	void update(final Patient person);
	void deleteById(final Long personId);
	void deleteByEmail(final String email);

	
}
