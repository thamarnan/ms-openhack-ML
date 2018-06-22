# coding: utf-8
# In[10]:
get_ipython().system(' /anaconda/envs/py35/bin/python -m pip freeze')
get_ipython().system(' /anaconda/envs/py35/bin/python -m pip --version')
# In[11]:
import sys
sys.version
# In[12]:
get_ipython().system(' /anaconda/envs/py35/bin/python -m pip install --yes numpy')
# In[21]:
import requests 

headers = dict()
headers['Prediction-Key'] = '1812e5ddea424989a9ead7a92e83a51e'
headers["Content-Type"] = 'application/json'

r = requests.post('https://southcentralus.api.cognitive.microsoft.com/customvision/v2.0/Prediction/a1ba6f7d-57b4-40d3-ba55-f013e7e4fac4/url?iterationId=f63dfbcc-466d-4db0-a18c-436037ce4151')

print(r.status_code)

