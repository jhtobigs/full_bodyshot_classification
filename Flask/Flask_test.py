import json
import requests
import numpy as np
from PIL import Image

image = Image.open('/content/drive/My Drive/Flask/test_image.jpg')
pixels = np.array(image)

headers = {'Content-Type':'application/json'}
address = "ngrok.io address input"
data = {'images':pixels.tolist()}

result = requests.post(address, data=json.dumps(data), headers=headers)

print(str(result.content, encoding='utf-8'))
