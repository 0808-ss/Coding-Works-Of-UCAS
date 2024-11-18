import requests


headers = { 'Range': 'bytes=100-200' }
r = requests.get('https://127.0.0.1:4430/index.html', headers=headers, verify=False)
# print(r.status_code)
# print(r.content)
# print(open('./resource_dir/index.html', 'rb').read()[100:201])
assert(r.status_code == 206 and open('./resource_dir/index.html', 'rb').read()[100:201] == r.content)

# r = requests.get('http://127.0.0.1:8888/index.html', verify=False)

# http 301
r = requests.get('http://127.0.0.1:8000/index.html', allow_redirects=False)
assert(r.status_code == 301 and r.headers['Location'] == 'https://127.0.0.1:4430/index.html')

# https 200 OK
r = requests.get('https://127.0.0.1:4430/index.html', verify=False)
assert(r.status_code == 200 and open('./resource_dir/index.html', 'rb').read() == r.content)


# http 404
r = requests.get('https://127.0.0.1:4430/notexist.html', verify=False)
assert(r.status_code == 404)



# http 206
headers = { 'Range': 'bytes=100-' }
r = requests.get('https://127.0.0.1:4430/index.html', headers=headers, verify=False)
assert(r.status_code == 206 and open('./resource_dir/index.html', 'rb').read()[100:] == r.content)
