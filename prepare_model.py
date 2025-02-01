import pathlib
from pathlib import Path
import os
import requests
import tarfile

url = 'https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/ssd_mobilenet_v1_coco/ssd_mobilenet_v1_coco_2018_01_28.tar.gz'
fn = Path('ssd_mobilenet_v1_coco_2018_01_28.tar.gz')
tf_model = Path('ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb')
ov_model = Path('ssd_mobilenet_v1_coco_2018_01_28.xml')

if not fn.exists():
    print(f'Downloading {url}.')
    data = requests.get(url).content
    with open(fn, 'wb') as f:
        f.write(data)

    print(f'Untaring {fn}.')
    with tarfile.open(fn) as f:
        f.extractall()

import openvino as ov

print('Converting model.')
model = ov.convert_model(tf_model)

ov.save_model(model, './Docker' / ov_model)
print(f'OpenVINO model {ov_model} is generated.')
