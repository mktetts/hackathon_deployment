from flask import jsonify


class Error:
    def __new__(self, status, error, status_code):
        self.status = status
        self.error = error
        self.status_code = status_code
        return jsonify(
            {
                "Status"  : self.status,
                "Message" : self.error
            }
        ), status_code
