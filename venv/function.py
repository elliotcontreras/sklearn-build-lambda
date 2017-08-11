# sklearn Iris classifier meant as an example to be used for aws lambda

import boto3
import os
import ctypes
import uuid
import sklearn
import pickle
import numpy as np

for d, _, files in os.walk('lib'):
    for f in files:
        if f.endswith('.a'):
            continue
        ctypes.cdll.LoadLibrary(os.path.join(d, f))

s3_client = boto3.client('s3')

def handler(event, context):

    #Info
    sepal_length = float(event.get('Iris')['sepal_length'])
    sepal_width = float(event.get('Iris')['sepal_width'])
    petal_length = float(event.get('Iris')['petal_length'])
    petal_width = float(event.get('Iris')['petal_width'])

    #Row
    row_arti = np.array([sepal_length,sepal_width,petal_length,petal_width]).reshape(1,4)

    #load model
    bucket = 'BUCKET-NAME-ON-AWS'
    key = 'model.pckl'
    download_path = '/tmp/{}{}'.format(uuid.uuid4(), key)
    s3_client.download_file(bucket, key, download_path)
    
    f = open(download_path, 'rb')
    model = pickle.load(f)
    f.close()
    
    #Predict
    class_prediced = int(model.predict(row_arti)[0])
    
    return class_prediced
