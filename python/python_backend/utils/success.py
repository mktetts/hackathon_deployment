from flask import jsonify


class Success:
    def __new__(self, success, message, status_code):
        self.success = success
        self.data = message
        # self.transaction_receipt = transaction_receipt
        self.status_code = status_code
        return jsonify(
            {
                "Status"  : self.success, 
                "Message" : self.data, 
                # "Receipt" : json.loads(self.transaction_receipt)
            }
        ), status_code
