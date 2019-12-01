import re
from robobrowser import RoboBrowser
from tqdm import tqdm
import os
import sys
import time
from colorama import Fore, Style
from pymongo import MongoClient

proxy_list = []
with open('proxies.txt', 'r') as f:
    for line in f:
        line = line.replace('\n', '')
        proxy_list.append(line)
print(proxy_list)

if len(sys.argv) is not 4:
    print("usage: \"python3 SOscraper.py 'java, c, or python-3.x' from to\"")
    print("'from' is and integer that specifies the beginning page and 'to' is an integer specifies the end page")
    exit()
if sys.argv[1] != 'java' and sys.argv[1] != 'c' and sys.argv[1] != 'python-3.x':
    print("usage: python3 SOscraper.py 'java, c, or python-3.x' from to")
    print("'from' is and integer that specifies the beginning page and 'to' is an integer specifies the end page")
    print("invalid language")
    exit()
try:
    sys.argv[2] = int(sys.argv[2])
    sys.argv[3] = int(sys.argv[3])
except ValueError:
    print("usage: python3 SOscraper.py 'java, c, or python-3.x' from to")
    print("'from' is and integer that specifies the beginning page and 'to' is an integer specifies the end page")
    print("invalid range. 'from' and 'to' must be integers")
    exit()
if (sys.argv[2]) < 0 or (sys.argv[2] > sys.argv[3]) or (sys.argv[3] < 0):
    print("usage: python3 SOscraper.py 'java, c, or python-3.x' from to")
    print("'from' is and integer that specifies the beginning page and 'to' is an integer specifies the end page")
    print("invalid range. 'from' must be less than 'to'")
    exit()

print("Connecting to db...")
client = MongoClient('da1.eecs.utk.edu')
db = client['fdac19-Stackbot']
col = db['SOdata']
#dirname = os.path.abspath(os.path.dirname('__file__'))
#dirname += '/SOfiles'
#dirname += '/' + sys.argv[1]
stackoverflow = 'http://stackoverflow.com'
browser = RoboBrowser(parser='html.parser')
Name_Code = {} #key is name of webpage, value[0] is Code, value[1] is label
Code = []

def get_name(s):
    name = ""
    x = len(s)-1
    for i in reversed(s):
        if i is '/':
            break
        x -= 1
    for j in range(x+1, len(s)):
        name = name + s[j]
    return name

def get_buggy_code():
    global Code
    global col
    print("GETTING BUGGY CODE\n") 
    pi = 0
    for i in tqdm(range(sys.argv[2],sys.argv[3])):
        #time.sleep(2)
        while(True):
            try:
                browser.open('http://stackoverflow.com/questions/tagged/' + sys.argv[1] + '?sort=newest&page=' + str(i) + '&pagesize=15', proxies={'http': proxy_list[pi]})
                break
            except:
                print("Couldn't open http://stackoverflow.com/questions/tagged/" + sys.argv[1] + "?sort=newest&page=" + str(i) + '&pagesize=15')
                pi += 1
                if pi == len(proxy_list):
                    print("Ran out of proxies :(")
                    exit()
        questions_block = browser.find('div', id= 'questions')
        if questions_block is None:
            print("check link")
            print('http://stackoverflow.com/questions/tagged/' + sys.argv[1] + '?sort=newest&page=' + str(i) + '&pagesize=15')
            continue
        questions = questions_block.find_all('a', class_= 'question-hyperlink')
        if questions is None:
            continue
        for question in questions:
            #time.sleep(11)
            question = question.get('href')
            if question is None:
                continue
            print(Fore.GREEN + stackoverflow + question, end=' ')
            print(Style.RESET_ALL, end='')
            if col.find({'url': stackoverflow + question, 'type': 'buggy'}).count() >= 1:
                print(Fore.RED + '- already seen')
                print(Style.RESET_ALL)
                continue
            else:
                print()
            try:
                browser.open(stackoverflow + question, proxies={'http': proxy_list[pi]})
            except:
                print("Couldn't open " + (stackoverflow + question))
                continue
            Name = get_name(question)
            Name += '_Q'

            if sys.argv[1] == 'python3':
                Name += '.py'
            elif sys.argv[1] == 'java':
                Name += '.java'
            elif sys.argv[1] == 'c':
                Name += '.c'

            ####################################################################
            tag_holder = browser.find('div', class_='grid ps-relative d-block')
            if tag_holder is not None:
                tags = tag_holder.text.replace('\n', '').split(" ")
                if tags is None:
                    tags = []
            else:
                tags = []
            ####################################################################
            question_text = browser.find('div', class_= 'post-text')
            if question_text is None:
                continue
            code_blocks = question_text.find_all('pre')
            # if there's only one code block it is more likely to contain the bug as
            # opposed to a question with multiple blocks of code
            if len(code_blocks) > 0:
                insert_dict = {}
                insert_dict['codes'] = []
                first = False
                for code_block in code_blocks:
                    code = code_block.find('code')
                    if code is not None:
                        code = str(code.text)
                        print(code+'\n')
                        insert_dict['codes'].append({"code": code.strip(), "compilation_message": None, "error_type": None, "ast": None})
                        if not first:
                            insert_dict['type'] = "buggy"
                            insert_dict['url'] = stackoverflow + question # url
                            insert_dict['tags'] = tags
                        #Name_Code[Name] = tuple(Code)
                        #Code = []
                    first = True 
                col.insert_one({'type': insert_dict['type'].strip(), 'tags': insert_dict['tags'], 'url': insert_dict['url'].strip(), 'language': sys.argv[1].strip(), 'code_and_ast': insert_dict['codes']})
                print('\n')
            else:
                continue

def get_good_code():
    global Code
    global col
    print("GETTING GOOD CODE\n")
    pi = 0
    for i in tqdm(range(sys.argv[2],sys.argv[3])):
        #time.sleep(11)
        while(True):
            try:
                browser.open('http://stackoverflow.com/questions/tagged/' + sys.argv[1] + '?sort=newest&page=' + str(i) + '&pagesize=15', proxies={'http': proxy_list[pi]})
                break
            except:
                print("Couldn't open http://stackoverflow.com/questions/tagged/" + sys.argv[1] + "?sort=newest&page=" + str(i) + '&pagesize=15')
                pi += 1
                if pi == len(proxy_list):
                    print("Ran out of proxies :(")
                    exit()
        questions_block = browser.find('div', id= 'questions')
        if questions_block is None:
            continue
        questions = questions_block.find_all('a', class_= 'question-hyperlink')
        if questions is None:
            continue
        for question in questions:
            #time.sleep(11)
            question = question.get('href')
            if question is None:
                continue
            print(Fore.GREEN + stackoverflow + question, end=' ')
            print(Style.RESET_ALL, end='')
            if col.find({'url': stackoverflow + question, 'type': 'good'}).count() >= 1:
                print(Fore.RED + '- already seen')
                print(Style.RESET_ALL)
                continue
            else:
                print()
            try:
                browser.open(stackoverflow + question, proxies={'http': proxy_list[pi]})
            except:
                print("Couldn't open " + (stackoverflow + question))
                continue
            Name = get_name(question)
            Name += '_A'

            if sys.argv[1] == 'python3':
                Name += '.py'
            elif sys.argv[1] == 'java':
                Name += '.java'
            elif sys.argv[1] == 'c':
                Name += '.c'

            #####################################################################
            tag_holder = browser.find('div', class_='grid ps-relative d-block')
            if tag_holder is not None:
                tags = tag_holder.text.replace('\n', '').split(" ")
                if tags is None:
                    tags = []
            else:
                tags = []
            #####################################################################
            answers_container = browser.find('div', id= 'answers')
            if answers_container is None:
                continue
            answers = answers_container.find_all('div', class_='answer')
            if answers is None:
                continue
            for answer in answers:
                # if there's only one code block it is more likely to good code as
                # opposed to an answer with multiple blocks of code
                answer = answer.find('div', class_='post-layout')
                answer = answer.find('div', class_='answercell post-layout--right')
                answer = answer.find('div', class_='post-text')
                code_blocks = answer.find_all('pre')
                if len(code_blocks) > 0:
                    insert_dict = {}
                    insert_dict['codes'] = []
                    first = False
                    for code_block in code_blocks:
                        code = code_block.find('code')
                        if code is not None:
                            code = str(code.text)
                            print(code+'\n')
                            insert_dict['codes'].append({"code": code.strip(), "compilation_message": None, "error_type": None, "ast": None})
                            if not first:
                                insert_dict['type'] = "good"
                                insert_dict['url'] = stackoverflow + question # url
                                insert_dict['tags'] = tags
                            #Name_Code[Name] = tuple(Code)
                            #Code = []
                        first = True 
                    col.insert_one({'type': insert_dict['type'].strip(), 'tags': insert_dict['tags'], 'url': insert_dict['url'].strip(), 'language': sys.argv[1].strip(), 'code_and_ast': insert_dict['codes']})
                    print('\n')
                    break
                else:
                    continue

def main():
    global Name_Code
    global col
    get_buggy_code()
    get_good_code()
    #for key, value in tqdm(Name_Code.items()):
        # WRITE TO MONGO INSTEAD
        #col.insert_one({'type': value[1].strip(), 'tags': value[3].strip(), 'url': value[2].strip(), 'language': sys.argv[1].strip(), 'code': value[0].strip()})
        #with open(dirname + '/' + key, 'w') as f:
        #    f.write(value[0])
    #return Name_Code

main()
