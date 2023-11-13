const DoctorDetailContract = artifacts.require('DoctorDetailContract')

module.exports = function (deployer) {
  deployer.deploy(DoctorDetailContract)
}
