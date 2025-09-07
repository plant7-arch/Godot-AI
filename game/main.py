import json


with open('data/memory.json', 'r') as file:
    p_data = json.load(file)
print(p_data)

while True:
    try:
        with open('data/memory.json', 'r') as file:
            data = json.load(file)
    except json.decoder.JSONDecodeError:
        pass
    else:
        if not data[1] == p_data[1]:
            print(data[1])
        p_data = data