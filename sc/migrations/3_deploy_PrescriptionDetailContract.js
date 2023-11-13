const PrescriptionDetailContract = artifacts.require('PrescriptionDetailContract')

module.exports = function (deployer) {
  deployer.deploy(PrescriptionDetailContract)
}
