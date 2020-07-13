import argparse
import io
import os

import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from flask import Flask, redirect, request, url_for
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from werkzeug.utils import secure_filename

from lightning_rexnetv1 import CustomReXNetV1

UPLOAD_FOLDER = './Image/'  # 이미지 저장할 폴더 지정
if not os.path.exists(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)
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
    global args
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))  # 이미지 올리면 지정한 폴더 안에 이미지 저장

            model = CustomReXNetV1.load_from_checkpoint('./sample.ckpt', map_location=DEVICE)
            model.eval()

            # input image transform
            def image_parse(self, image_path):
                image = Image.open(UPLOAD_FOLDER+filename)

                return image

            def transform_image(image_bytes):
                my_transforms = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
                    ])
                image = image_parse(image_bytes, UPLOAD_FOLDER+"filename")
                return my_transforms(image).unsqueeze(0)

            # prediction
            tensor = transform_image(filename, )
            
            outputs = model.forward(tensor)
            _, y_hat = outputs.max(0)
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
                """ % (IMAGE_SRC, str(prediction))
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=-1, help='number of gpus')
    parser.add_argument('--path', type=str, default='D:/data/musinsa/train_test_valid', help='parent directory containing train, val, test data')
    parser.add_argument('--epoch', type=int, default=200, help='epochs to train')
    parser.add_argument('--seed', type=int, default=711, help='random seed')
    parser.add_argument('--num_classes', type=int, default=2, help='output class number')
    parser.add_argument('--distributed_backend', type=str, default='dp')
    parser.add_argument('--mode', type=str, default='train', help='train or test')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--model', type=str, default='mobilenet', help='resnet/mobilenet/efficientnet/rexnet')
    parser.add_argument('--pretrain', type=str, default='true', help='using ImageNet-pretrained Model')
    parser.add_argument('--mult', type=float, default=1.0, help='rexnet scale(1.0/1.3/1.5/2.0)')

    parser.add_argument('--step_size', type=int, default=5, help='lr decay step size')
    parser.add_argument('--decay_rate', type=float, default=0.2, help='lr decay rate')

    args = parser.parse_args()
    app.run(host='127.0.0.1', port=5000, debug=True)
