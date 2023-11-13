from zxing import BarCodeReader

def decode_qr_code(image_path):
    reader = BarCodeReader()
    barcode = reader.decode(image_path)
    return barcode.parsed

qr_code = decode_qr_code("qrcode.png")
print(qr_code)