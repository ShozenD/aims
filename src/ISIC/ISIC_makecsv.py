from api import ISICApi
import csv

api = ISICApi()

with open("label.csv","w") as writeFile:
    fieldnames = ['_id', 'diagnosis']
    head = csv.DictWriter(writeFile,fieldnames=fieldnames)
    head.writeheader()

imageList = api.getJson('image?limit=0&offset=0&sort=name')
for i,image in enumerate(imageList):
    imageId = image['_id']
    imageDetail = api.getJson('image/'+imageId)
    print("processing : {0}  . . . {1}/{2} ".format(imageId,i,len(imageList)),end='\r')
    with open("label.csv","a") as writeFile:
        writer = csv.writer(writeFile)
        data = imageDetail["_id"],imageDetail["meta"]["clinical"]["diagnosis"]
        if data = None:
            data = "none"
        writer.writerow(data)
