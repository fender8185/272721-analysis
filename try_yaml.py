import yaml

with open('cb_info.yaml', 'r') as file:
    data = yaml.safe_load(file)


for key, value in data['27271'].items():
    print(f"{key}: {value}")