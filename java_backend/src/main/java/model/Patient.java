package model;

import javax.persistence.Column;
import javax.persistence.Entity;
// import javax.persistence.Id;
import javax.persistence.Table;

import io.quarkus.hibernate.orm.panache.PanacheEntity;

@Entity
@Table(name = "patients", schema = "PUBLIC")
public class Patient extends PanacheEntity {
	
	// @GeneratedValue(strategy = GenerationType.SEQUENCE)
	// @Id
	@Column(name = "email")
	private String email;
	
	@Column(name = "name")
	private String name;
	
	@Column(name = "card")
	private String card;

	@Column(name = "amount")
	private int amount;
	
	public Patient() {
		
	}
	
	public String getEmail() {
		return email;
	}
	
	public void setId(String emaill) {
		this.email = emaill;
	}
	
	public String getName() {
		return name;
	}
	
	public void setName(String name) {
		this.name = name;
	}
	
	public String getCard() {
		return card;
	}
	
	public void setCard(String card) {
		this.card = card;
	}

	public int getAmount() {
		return amount;
	}
	
	public void setAmount(int amount) {
		this.amount = amount;
	}

	public static Patient findByEmail(String email2) {
		return null;
	}

    public static void deleteByEmail(String email2) {
    }
	
	
	
}







