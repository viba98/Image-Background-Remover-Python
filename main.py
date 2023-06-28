from fastapi import FastAPI, Request, File, UploadFile
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import uuid
import os

from model import U2NET
from torch.autograd import Variable
from skimage import io, transform
from PIL import Image
from io import BytesIO
import base64


app = FastAPI()

@app.post("/api/endpoint")
async def your_endpoint(images: List[UploadFile] = File(...)):
    contents_list = []
    for image in images:
        contents = await image.read()
        img = bytearray(contents)
        contents_list.append(contents)
    # data = await request.json()
    # Process the data
    output = removeBg(img)
    return {output}

# Get The Current Directory
currentDir = os.path.dirname(__file__)

# Functions:
# Save Results


def save_output(image_name, output_name, pred, d_dir, type):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict_np*255).convert('RGB')
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]))
    pb_np = np.array(imo)
    if type == 'image':
        # Make and apply mask
        mask = pb_np[:, :, 0]
        mask = np.expand_dims(mask, axis=2)
        imo = np.concatenate((image, mask), axis=2)
        image = cv2.cvtColor(imo, cv2.COLOR_RGB2RGBA)
        temp_file = "temp_image.png"
        cv2.imwrite(temp_file, image)
        imo = Image.fromarray(imo, 'RGBA')
        with open(temp_file, "rb") as f:
            image_data = f.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')

        # print('PIL', imo)
        # print('b64', base64_image)

    imo.save(d_dir+output_name)
    return base64_image

# Remove Background From Image (Generate Mask, and Final Results)


def removeBg(img):
    inputs_dir = os.path.join(currentDir, 'static/inputs/')
    results_dir = os.path.join(currentDir, 'static/results/')
    masks_dir = os.path.join(currentDir, 'static/masks/')

    # convert string of image data to uint8
    # with open(imagePath, "rb") as image:
    #     f = image.read()
    #     img = bytearray(f)
    #     print(img)

    nparr = np.frombuffer(img, np.uint8)

    if len(nparr) == 0:
        return '---Empty image---'

    # decode image
    try:
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except:
        # build a response dict to send back to client
        return "---Empty image---"

    # save image to inputs
    unique_filename = str(uuid.uuid4())
    cv2.imwrite(inputs_dir+unique_filename+'.jpg', img)

    # processing
    image = transform.resize(img, (320, 320), mode='constant')

    tmpImg = np.zeros((image.shape[0], image.shape[1], 3))

    tmpImg[:, :, 0] = (image[:, :, 0]-0.485)/0.229
    tmpImg[:, :, 1] = (image[:, :, 1]-0.456)/0.224
    tmpImg[:, :, 2] = (image[:, :, 2]-0.406)/0.225

    tmpImg = tmpImg.transpose((2, 0, 1))
    tmpImg = np.expand_dims(tmpImg, 0)
    image = torch.from_numpy(tmpImg)

    image = image.type(torch.FloatTensor)
    image = Variable(image)

    d1, d2, d3, d4, d5, d6, d7 = net(image)
    pred = d1[:, 0, :, :]
    ma = torch.max(pred)
    mi = torch.min(pred)
    dn = (pred-mi)/(ma-mi)
    pred = dn

    output = save_output(inputs_dir+unique_filename+'.jpg', unique_filename +
                '.png', pred, results_dir, 'image')

    # save_output(inputs_dir+unique_filename+'.jpg', unique_filename +
    #             '.png', pred, masks_dir, 'mask')
    return output


# ------- Load Trained Model --------
print("---Loading Model---")
model_name = 'u2net'
model_dir = os.path.join(currentDir, 'saved_models',
                         model_name, model_name + '.pth')
net = U2NET(3, 1)
if torch.cuda.is_available():
    net.load_state_dict(torch.load(model_dir))
    net.cuda()
else:
    net.load_state_dict(torch.load(model_dir, map_location='cpu'))
# ------- Load Trained Model --------
