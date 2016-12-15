import json
import pyjsonrpc
import base64
hc = pyjsonrpc.HttpClient(url='http://localhost:7004')
rst = hc.names(base64.b64encode(open('./test.jpg').read()))
print rst
print type(rst)

