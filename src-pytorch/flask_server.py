import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from torchvision import models, transforms

import flask_restful
from flask_ngrok import run_with_ngrok

#model load
model = models.mobilenet_v2(pretrained=True)
torch.save(model, '/content/drive/My Drive/Flask/full_bodyshot_classification-master/src-pytorch/musinsa_model.pt')

model.eval()
normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) #�����Ϳ��� ��ü ��հ� ǥ�������� ���Ͽ� �� ���� Ȱ���Ͽ� ����ȭ�� ����


# Flask Restful API �о���� APP ����.
app = Flask(__name__)
api = flask_restful.Api(app)

# Flask resource -> Test class 
# get �Լ��� HTTP Get���� ����� �о����
class Test(flask_restful.Resource):
    def inference():
        data = request.json
        _, result = model.forward(normalize(np.array(data['images'], dtype=np.uint8)).unsqueeze(0)).max(1)
        return str(result.item())
    
# Test Ŭ������ ���ҽ��� �߰�. �ι�° ���ڴ� ������ ��ġ. 
# �츮�� ~/venv/tf_mnist ���� ���丮���� ���� ���̹Ƿ� '/'     
api.add_resource(Test, '/')
# ����ϴ� ��Ʈ�� 5000��
run_with_ngrok(app)
app.run()
