package dao;

import java.util.List;

import org.springframework.stereotype.Repository;

import model.Patient;

@Repository
public class PatientDaoImpl implements PatientDao {
	
	@Override
	public List<Patient> findAll() {
		return Patient.findAll().list();
	}
	
	@Override
	public Patient findById(final Long personId) {
		return Patient.findById(personId);	
	}

	@Override
	public Patient findByEmail(final String email) {
		return Patient.findByEmail(email);	
	}
	
	@Override
	public void save(final Patient person) {
		Patient.persist(person);
	}
	
	@Override
	public void update(final Patient person) {
		Patient.persist(person);
	}
	
	@Override
	public void deleteById(final Long personId) {
		Patient.deleteById(personId);
	}
	
	@Override
	public void deleteByEmail(String email) {
		Patient.deleteByEmail(email);
	}
	
}




