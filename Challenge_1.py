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
headers['Prediction-Key'] = 'xxx'
headers["Content-Type"] = 'application/json'

r = requests.post('https://southcentralus.api.cognitive.microsoft.com/customvision/v2.0/Prediction/xx/url?iterationId=xx')

print(r.status_code)

