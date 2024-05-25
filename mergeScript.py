import json
def merge_json(file1, file2):
    # Read the contents of both JSON files
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        json1 = json.load(f1)
        json2 = json.load(f2)

    # Merge the lists
    merged_json = json1 + json2

    return merged_json

# Paths to your JSON files
file1_path = './GAME1/baseline.json'
file2_path = './GAME2/baseline.json'

# Merge the JSON files
merged_data = merge_json(file1_path, file2_path)

# Optionally, write the merged data back to a JSON file
with open('baseline_full.json', 'w') as outfile:
    json.dump(merged_data, outfile, indent=4)
