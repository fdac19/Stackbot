from pymongo import MongoClient
from tqdm import tqdm
import ast

print("Connecting to db...")
client = MongoClient('da1.eecs.utk.edu')
db = client['fdac19-Stackbot']
col = db['SOdata']

docs = list(col.find({}))
for doc in tqdm(docs):
    removed = False
    all_blocks = doc['code_and_ast']
    for i, each in enumerate(all_blocks):
        if not isinstance(each, dict):
            removed = True
            col.remove({"_id": doc['_id']})
            break
        if each['compilation_message'] is None or each['compilation_message'] == 'nan':
            print(each['compilation_message'])
        if doc['language'] == 'python-3.x' and (each['compilation_message'] is None or each['error_type'] is None or each['ast'] is None):
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
                #col.update_one({'code_and_ast': {'code': code}}, {"$set": {'code': {'compilation_message': 'successful'}}})
                #col.update_one({'code_and_ast': {'code': code}}, {"$set": {'code': {'error_type': 'none'}}})
                #col.update_one({'code_and_ast': {'code': code}}, {"$set": {'code': {'ast': tree_dump}}})
                all_blocks[i]['compilation_message'] = 'successful' 
                all_blocks[i]['error_type'] = 'none' 
                all_blocks[i]['ast'] = tree_dump 
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
                #col.update_one({'code_and_ast': {'code': code}}, {"$set": {'code': {'compilation_message': str(message)}}})
                #col.update_one({'code_and_ast': {'code': code}}, {"$set": {'code': {'error_type': err}}})
                #col.update_one({'code_and_ast': {'code': code}}, {"$set": {'code': {'ast': 'none'}}})
                all_blocks[i]['compilation_message'] = str(message) 
                all_blocks[i]['error_type'] = err
                all_blocks[i]['ast'] = 'none'
            #print()
            #print()
    if not removed:
        col.update_one({'code_and_ast': doc['code_and_ast']}, {'$set': {'code_and_ast': all_blocks}})
