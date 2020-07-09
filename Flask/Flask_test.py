import os
import torch
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
import io
import torchvision.transforms as transforms
from PIL import Image


UPLOAD_FOLDER = './Image/'  # 이미지 저장할 폴더 지정
ALLOWED_EXTENSIONS = {'JPG','jpg'}  # 허용 가능한 확장자만 ,PNG이면 바꾸기
DEVICE = 'cpu'  # 환경에 맞게 (gpu면 나중에 바꾸기)
IMAGE_SRC = "https://nextstylemag.com/wp-content/uploads/2020/02/men-shirts-style94.jpg" # 배경 이미지

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))  # 이미지 올리면 지정한 폴더 안에 이미지 저장

            model = torch.load('../src-pytorch/musinsa_model.pt', map_location=DEVICE)  # .module -> 멀티 지피유 썼으면 뒤에 붙이기
            model.eval()

            # input image transform
            def transform_image(image_bytes):
                my_transforms = transforms.Compose([transforms.Resize(255),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(
                                                        [0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])
                image = Image.open(io.BytesIO(image_bytes))
                return my_transforms(image).unsqueeze(0)

            # prediction
            tensor = transform_image(filename)
            outputs = model.forward(tensor)
            _, y_hat = outputs.max(1)
            prediction = str(y_hat.item())

            return """
                   <!doctype html>
                   <title>Tobigs Musinsa Style Check</title>

                   <div style="TEXT-ALIGN: center">
                   <h1>Musinsa Style solution</h1>
                   </div> 

                   <div style="TEXT-ALIGN: center">
                   <img src=%s>
                   </div> 

                   <div style="TEXT-ALIGN: center">
                   <form action="" method=post enctype=multipart/form-data>
                     <p><input type=file name=file value=Choose>
                        <input type=submit value=Predict>
                   </form>
                   </div> 
                   <p style="TEXT-ALIGN: center">
                   %s
                   </p>
                   """ % (IMAGE_SRC, str(prediction[0]))
    return """
    <!doctype html>
    <title>Tobigs Musinsa Style Check</title>

    <div style="TEXT-ALIGN: center">
    <h1>Musinsa Style solution</h1>
    </div> 

    <div style="TEXT-ALIGN: center">
    <img src=%s>
    </div> 

    <div style="TEXT-ALIGN: center">
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file value=Choose>
         <input type=submit value=Predict>
    </form>
    </div> 
    """ % IMAGE_SRC


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
