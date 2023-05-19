import json

with open('message_descriptors.json', 'r', encoding='utf-8') as file_handle:
    messages = json.load(file_handle)
    action = 7
    for message in messages:
        if message['ID'] == action:
            print(message['descr'])
            break
