package service;

import java.util.List;

import javax.transaction.Transactional;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import dao.PatientDao;
import model.Patient;

@Service
@Transactional
public class PatientServiceImpl implements PatientService {
	
	@Autowired
	private PatientDao patientDao;
	
	@Override
	public List<Patient> findAll() {
		return this.patientDao.findAll();
	}
	
	@Override
	public Patient findById(Long personId) {
		return this.patientDao.findById(personId);
	}

	@Override
	public Patient findByEmail(String email) {
		return this.patientDao.findByEmail(email);
	}
	
	@Override
	public void save(Patient person) {
		this.patientDao.save(person);
	}
	
	@Override
	public void update(Patient person) {
		this.patientDao.update(person);
	}
	
	@Override
	public void deleteById(Long personId) {
		this.patientDao.deleteById(personId);
	}
	@Override
	public void deleteByEmail(String email) {
		this.patientDao.deleteByEmail(email);
	}
	
	
}





