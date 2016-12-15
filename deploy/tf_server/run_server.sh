bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server \
    --port=7004 \
    --model_base_path=./../retrieval_web_demo/tf_server/model/vgg_serving \
    --model_name="inception" \

