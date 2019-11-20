from pymongo import MongoClient
from tqdm import tqdm
import ast
from bson.objectid import ObjectId
import math

print("Connecting to db...")
client = MongoClient('da1.eecs.utk.edu')
db = client['fdac19-Stackbot']
col = db['SOdata']

docs = list(col.find({}))
for doc in tqdm(docs):
    all_blocks = doc['code_and_ast']
    for i, each in enumerate(all_blocks):
        #####################################
        if not isinstance(each, dict):
            col.remove({"_id": doc['_id']})
            break
        #####################################
        cmp_str = 'code_and_ast.'+str(i)+'.compilation_message'
        err_str = 'code_and_ast.'+str(i)+'.error_type'
        ast_str = 'code_and_ast.'+str(i)+'.ast'

        if doc['language'] == 'python-3.x' and (not isinstance(each['compilation_message'], str) or not isinstance(each['error_type'], str) or not isinstance(each['ast'], str)):
            code = each['code']
            # try to compile the code
            message = ''
            try:
                tree = ast.parse(code)
                tree_dump = ast.dump(tree)
            except Exception as e:
                message = e
                err = str(type(message)).replace("<class '", '')
                err = err.replace("'>", '')
                #print(err)
            if message == '':
                #print("\nCompilation successful!")
                #print("#_______________________________________________________________________________#")
                col.update({'_id': ObjectId(doc['_id']), 'code_and_ast.code': code}, {"$set": {cmp_str: 'successful'}})
                col.update({'_id': ObjectId(doc['_id']), 'code_and_ast.code': code}, {"$set": {err_str: 'none'}})
                col.update({'_id': ObjectId(doc['_id']), 'code_and_ast.code': code}, {"$set": {ast_str: tree_dump}})
            else:
                if err == "ModuleNotFoundError":
                    fw = open('NEEDED_LIBRARIES', 'a')
                    fr = open('NEEDED_LIBRARIES', 'r')
                    lib = str(message).replace("No module named ", '')
                    lib = lib[1:]
                    lib = lib[:len(lib)-1]
                    if lib not in fr.read():
                        fw.write(lib+'\n')
                    fw.close()
                    fr.close()
                    continue
                #print("\nCompilation failed!")
                #print("#_______________________________________________________________________________#")
                col.update({'_id': ObjectId(doc['_id']), 'code_and_ast.code': code}, {"$set": {cmp_str: str(message)}})
                col.update({'_id': ObjectId(doc['_id']), 'code_and_ast.code': code}, {"$set": {err_str: err}})
                col.update({'_id': ObjectId(doc['_id']), 'code_and_ast.code': code}, {"$set": {ast_str : 'none'}})
            #print()
            #print()

