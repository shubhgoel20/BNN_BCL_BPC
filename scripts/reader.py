import numpy as np
import json
import sys

def read_file(file_name):
    with open(file_name, 'r') as file:
        content = file.read()
    return content

def process_data(content):
    start_idx = content.find('{')
    end_idx = content.find('}', start_idx) + 1
    json_str = content[start_idx:end_idx]
    data_dict = json.loads(json_str)
    raw_data = content[end_idx:].strip()
    matrices_raw = raw_data.split('\n\n')
    approaches = data_dict["Approaches"]
    approach_data = {}
    for i, matrix_raw in enumerate(matrices_raw):
        matrix_cleaned = []
        for line_ in matrix_raw.splitlines():
            line = line_.strip().strip(',').split(',')
            cleaned_line = [100*float(l.strip().strip("%")) for l in line]
            matrix_cleaned.append(cleaned_line)
        approach_data[approaches[i]] = np.array(matrix_cleaned)
    return data_dict, approach_data

def get_info(experiment_name):
    file_name = f'data/{experiment_name}.txt'
    content = read_file(file_name)
    return process_data(content)

def main():
    # Get filename from command line arguments
    if len(sys.argv) < 2:
        print("Usage: python reader.py <file_name>")
        sys.exit(1)

    file_name = sys.argv[1]
    content = read_file(file_name)
    data_dict, approach_data = process_data(content)

    print("Data Dictionary:", data_dict)
    print("\nApproach Data Matrices:")
    for approach, matrix in approach_data.items():
        print(f"{approach}:\n", matrix)

if __name__ == "__main__":
    main()
