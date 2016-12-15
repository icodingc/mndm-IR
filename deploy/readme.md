实现了两套检索系统
--------------------
1. python rpc 抽特征，计算NN 返回给web_demo [GPU版本，可以用来做Demo]
        server: tf_server
        client: tf_client/client.py + web_demo: controller_old.py
2. tensorflow gRPC ,serving 抽feature，web_demo计算ANN   [CPU,serving,可以写进毕设里]
        server: server/run_server.sh
        client: web_demo:controller_server.py
