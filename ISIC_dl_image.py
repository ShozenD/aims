from api import ISICApi
import urllib
import os

api = ISICApi()

savePath = 'ISICArchive/'
os.makedirs(savePath)
imageDetails = []

imageList = api.getJson('image?limit=0&offset=0&sort=name')
for i,image in enumerate(imageList):
    print("Processing : {0} . . . {1}/{2}".format(image['_id'],i,len(imageList)),end="\r")
    imageFileResp = api.get('image/%s/download' % image['_id'])
    imageFileResp.raise_for_status()
    imageFileOutputPath = os.path.join(savePath, '%s.jpg' % image['_id'])
    with open(imageFileOutputPath, 'wb') as imageFileOutputStream:
        for chunk in imageFileResp:
            imageFileOutputStream.write(chunk)
