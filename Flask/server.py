from torchvision import models, transforms
from flask import Flask, render_template, jsonify, request
import flask_restful
from flask_ngrok import run_with_ngrok

import torch
import numpy as np

#model load
model = models.mobilenet_v2(pretrained=True)
torch.save(model, '/content/drive/My Drive/Flask/full_bodyshot_classification-master/src-pytorch/musinsa_model.pt')

model.eval()
normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) #데이터에서 전체 평균과 표준편차를 구하여 그 값을 활용하여 정규화를 수행


# Flask Restful API 읽어들일 APP 지정.
app = Flask(__name__)
api = flask_restful.Api(app)

# Flask resource -> Test class 
# get 함수가 HTTP Get으로 결과를 읽어들임
class Test(flask_restful.Resource):
    def inference():
        data = request.json
        _, result = model.forward(normalize(np.array(data['images'], dtype=np.uint8)).unsqueeze(0)).max(1)
        return str(result.item())
    
# Test 클래스를 리소스로 추가. 두번째 인자는 파일의 위치. 
# 우리는 ~/venv/tf_mnist 현재 디렉토리에서 읽을 것이므로 '/'     
api.add_resource(Test, '/')
# 사용하는 포트는 5000번
run_with_ngrok(app)
app.run()
