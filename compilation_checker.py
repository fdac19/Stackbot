from pymongo import MongoClient

print("Connecting to db...")
client = MongoClient('da1.eecs.utk.edu')
db = client['fdac19-Stackbot']
col = db['SOdata']

for doc in list(col.find({})):
    code = doc['code']
    if doc['language'] == 'python-3.x':
        # try to compile the code
        #code_obj = compile(source=code, mode='exec')
        try:
            print("#_______________________________________________________________________________#")
            exec(code)
            print("#_______________________________________________________________________________#")
        except Exception as e:
            print("#################################################################################")
            print(e)
            print("#################################################################################")
