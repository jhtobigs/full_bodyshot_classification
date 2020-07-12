import argparse
import io
import os

import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from flask import Flask, redirect, request, url_for, render_template
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from werkzeug.utils import secure_filename
from lightning_rexnetv1 import CustomReXNetV1

UPLOAD_FOLDER = './static/img_model/'  # 이미지 저장할 폴더 지정
if not os.path.exists(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)
ALLOWED_EXTENSIONS = {'JPG','jpg'}  # 허용 가능한 확장자만 ,PNG이면 바꾸기
DEVICE = 'cpu'  # 환경에 맞게 (gpu면 나중에 바꾸기)
uploaded_icon = "/static/img/upload.png"

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route("/", methods=['GET', 'POST'])
def mubigs_demo():
    global args
    if request.method == 'POST':
        file = request.files['file']
        if file :
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))  # 이미지 올리면 지정한 폴더 안에 이미지 저장
            uploaded_img = "/static/img_model/"+filename

            model = CustomReXNetV1.load_from_checkpoint('./model/sample.ckpt', map_location=DEVICE)
            model.eval()

            # input image transform
            def image_parse(self, image_path):
                image = Image.open(UPLOAD_FOLDER+filename)

                return image

            def transform_image(image_bytes):
                my_transforms = transforms.Compose([transforms.Resize((224,224)),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(
                                                            [0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
                image = image_parse(image_bytes, UPLOAD_FOLDER+"filename")
                return my_transforms(image).unsqueeze(0)

            # prediction
            tensor = transform_image(filename, )

            outputs = model.forward(tensor)
            _, y_hat = outputs.max(0)
            prediction = str(y_hat.item())

            state = "Submit!"
            if prediction == "1" :
                state = "FAIL :("
            else :
                state = "PASS :)"

            return render_template("mubigs_demo.html", uploaded_img=uploaded_img, state = state)  #prediction값을 받아서 html에 넘겨주는 코드

    return render_template("mubigs_demo.html", uploaded_img=uploaded_icon, state = "Submit!")

#predict 가져오기
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))  # 이미지 올리면 지정한 폴더 안에 이미지 저장
        """
        model = torch.load('./model/sample.ckpt', map_location=DEVICE)  # .module -> 멀티 지피유 썼으면 뒤에 붙이기
        model.eval()

        # input image transform
        def image_parse(self, image_path):
            image = Image.open(UPLOAD_FOLDER+filename)

            return image

        def transform_image(image_bytes):
            my_transforms = transforms.Compose([transforms.Resize((224,224)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(
                                                        [0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])
            image = image_parse(image_bytes, UPLOAD_FOLDER+"filename")
            return my_transforms(image).unsqueeze(0)

        # prediction
        tensor = transform_image(filename, )
        outputs = model.forward(tensor)
        _, y_hat = outputs.max(1)
        prediction = str(y_hat.item())

        pred = "%s" %(str(prediction))"""

        return "PASS"  #prediction값을 받아서 html에 넘겨주는 코드


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
