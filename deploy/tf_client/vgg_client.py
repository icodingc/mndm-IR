from __future__ import print_function
from grpc.beta import implementations
import tensorflow as tf
import predict_pb2
import prediction_service_pb2
import numpy as np

image_path = '/home/zxs/workspace/retrieval_web_demo/test.jpg' 
host, port = 'localhost:7004'.split(':')
channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

with open(image_path, 'rb') as f:
    # See prediction_service.proto for gRPC request/response details.
  data = f.read()
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'inception'
  request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(data, shape=[1]))
  result = stub.Predict(request, 10.0)  # 10 secs timeout
  feat = result.outputs['feats'].float_val
