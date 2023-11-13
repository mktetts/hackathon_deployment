import cv2
from pyzbar.pyzbar import decode

# Read the QR code image
qr_code_image = cv2.imread("qrcode.png")

# Convert the image to grayscale
gray_qr_code = cv2.cvtColor(qr_code_image, cv2.COLOR_BGR2GRAY)

# Decode the QR code
decoded_objects = decode(gray_qr_code)

# Print the decoded data
for obj in decoded_objects:
    print("Data:", obj.data.decode("utf-8"))

# Alternatively, you can access the first decoded data directly
if decoded_objects:
    print("First Decoded Data:", decoded_objects[0].data.decode("utf-8"))
