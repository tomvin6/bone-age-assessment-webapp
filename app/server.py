from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50
from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from pathlib import Path
import uvicorn, aiohttp, asyncio
import sys, numpy as np
import time

# imports from jupyter
import tensorflow as tf
import pandas as pd
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, LocallyConnected2D, Lambda, BatchNormalization
from tensorflow.keras.models import Model


path = Path(__file__).parent
model_file_url = 'hhttps://github.com/tvaingart/bone-age-assessment-webapp/blob/main/models/female_model_weights_resnet.h5?raw=true'
model_file_name = 'model'

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

MODEL_PATH = path/'models'/f'{model_file_name}.h5'
IMG_FOLDER = '/tmp/'
IMG_FILE_SRC = '/tmp/saved_image.png'
REL_IMG_FILE_SRC = 'saved_image.png'

x_col = 'path'
y_col = 'boneage'
width = height = 384
target_size = (width, height)


# get resnet model
def get_resnet_model(input_shape=(384, 384, 3)):
  in_lay = Input(input_shape)

  base_pretrained_model = ResNet50(input_shape =  input_shape, include_top = False, weights = 'imagenet')

  #base_pretrained_model.summary()
  base_pretrained_model.trainable = False
  pt_features = base_pretrained_model(in_lay)
  pt_depth = base_pretrained_model.get_output_shape_at(0)[-1]
  bn_features = BatchNormalization()(pt_features)

  # here we do an attention mechanism to turn pixels in the GAP on an off

  attn_layer = Conv2D(64, kernel_size = (1,1), padding = 'same', activation = 'relu')(bn_features)
  attn_layer = Conv2D(16, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)
  attn_layer = LocallyConnected2D(1, 
                                  kernel_size = (1,1), 
                                  padding = 'valid', 
                                  activation = 'sigmoid')(attn_layer)
  # fan it out to all of the channels
  up_c2_w = np.ones((1, 1, 1, pt_depth))
  up_c2 = Conv2D(pt_depth, kernel_size = (1,1), padding = 'same', 
                activation = 'linear', use_bias = False, weights = [up_c2_w])
  up_c2.trainable = False
  attn_layer = up_c2(attn_layer)

  mask_features = multiply([attn_layer, bn_features])
  gap_features = GlobalAveragePooling2D()(mask_features)
  gap_mask = GlobalAveragePooling2D()(attn_layer)
  # to account for missing values from the attention model
  gap = Lambda(lambda x: x[0]/x[1], name = 'RescaleGAP')([gap_features, gap_mask])
  gap_dr = Dropout(0.5)(gap)
  dr_steps = Dropout(0.25)(Dense(1024, activation = 'elu')(gap_dr))
  out_layer = Dense(1, activation = 'linear')(dr_steps) # linear is what 16bit did
  bone_age_model = Model(inputs = [in_lay], outputs = [out_layer])
  #bone_age_model.summary()
  return bone_age_model


def predict(img_path, male=True):
  print('trying to predict internal')
  img = load_image(img_path, False)
  # preprocess: 
  test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
  fake_test_df = pd.DataFrame({
                'id': [1],
                'boneage    male': [True], 
                'boneage_zscore': ['123'],
                'boneage_category': ['abc'],
                'rel_path': [REL_IMG_FILE_SRC],
                'path': [IMG_FILE_SRC]
                })
  test_generator = test_datagen.flow_from_dataframe(
      fake_test_df, directory=IMG_FOLDER, x_col=x_col, 
      y_col='boneage_zscore', target_size=target_size, color_mode='rgb',
      batch_size=1, shuffle=False,
      class_mode = 'sparse', validate_filenames=True)
  test_generator.reset()
  start = time.time()
  score = model.evaluate(test_generator, steps=1)
  print('model output: ', score)
  
def load_image(img_path, show=False):
    img = image.load_img(img_path)
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()
    return img_tensor

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_model():
    #UNCOMMENT HERE FOR CUSTOM TRAINED MODEL
    # await download_file(model_file_url, MODEL_PATH)
    # model = load_model(MODEL_PATH) # Load your Custom trained model
    # model._make_predict_function()
    model = get_resnet_model()
    #model = ResNet50(weights='imagenet') # COMMENT, IF you have Custom trained model
    return model

# Asynchronous Steps
loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_model())]
model = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    img_bytes = await (data["file"].read())
    with open(IMG_FILE_SRC, 'wb') as f: f.write(img_bytes)
    predict(IMG_FILE_SRC)
    return model_predict(IMG_FILE_SRC, model)

def model_predict(img_path, model):
    result = []; img = image.load_img(img_path, target_size=(384, 384))
    x = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
    predictions = decode_predictions(model.predict(x), top=3)[0] # Get Top-3 Accuracy
    for p in predictions: _,label,accuracy = p; result.append((label,accuracy))
    result_html1 = path/'static'/'result1.html'
    result_html2 = path/'static'/'result2.html'
    result_html = str(result_html1.open().read() +str(result) + result_html2.open().read())
    return HTMLResponse(result_html)

@app.route("/")
def form(request):
    index_html = path/'static'/'index.html'
    return HTMLResponse(index_html.open().read())

if __name__ == "__main__":
    if "serve" in sys.argv: uvicorn.run(app, host="0.0.0.0", port=8080)
