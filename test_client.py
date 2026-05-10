import urllib.request
import urllib.parse
import time
from PIL import Image
import io
import json

# 1. Create a dummy high-res image (2000x2000)
print("Generating dummy 2000x2000 image...")
img = Image.new('RGB', (2000, 2000), color = 'red')
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='JPEG', quality=95)
img_bytes = img_byte_arr.getvalue()

# 2. Send request to the local Flask server
url = 'http://127.0.0.1:5000/poison'

# Build multipart/form-data manually since urllib doesn't have requests' magic
boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
body = []
body.extend([
    f'--{boundary}',
    'Content-Disposition: form-data; name="epsilon"',
    '',
    '0.01',
    f'--{boundary}',
    'Content-Disposition: form-data; name="attack_mode"',
    '',
    'fgsm',
    f'--{boundary}',
    'Content-Disposition: form-data; name="image"; filename="dummy.jpg"',
    'Content-Type: image/jpeg',
    ''
])

body_bytes = '\r\n'.join(body).encode('utf-8') + b'\r\n' + img_bytes + b'\r\n' + f'--{boundary}--'.encode('utf-8')

req = urllib.request.Request(url, data=body_bytes)
req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')

print("Sending POST request to /poison...")
start_time = time.time()
try:
    response = urllib.request.urlopen(req)
    res_body = response.read()
    end_time = time.time()

    print(f"\n--- CLIENT-SIDE METRICS ---")
    print(f"Total time experienced by client: {end_time - start_time:.2f} seconds")
    print(f"Status Code: {response.status}")

    res_json = json.loads(res_body)
    print("\n--- SERVER-SIDE METRICS (from JSON response) ---")
    if "timings" in res_json:
        print(json.dumps(res_json["timings"], indent=4))
    else:
        print("No timings found in response.")
except Exception as e:
    print("Error:", e)
    if hasattr(e, 'read'):
        print(e.read().decode())
