import glob
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True, type=str, help='The directory of unzipped data.zip')
parser.add_argument('--index_file', required=True, type=str, help='The IDs of selected data, e.g., example-standard.txt')
parser.add_argument('--output_file', required=True, type=str, help="The selected data in LLaVA's training format")

if __name__ == "__main__":
    args = parser.parse_args()

    # load all datasets
    input_files = sorted(glob.glob(f'{args.data_dir}/*.json'))
    dataset = dict()

    for input_file in input_files:
        dataset_name = os.path.basename(input_file).split('.')[0]
        dataset[dataset_name] = dict()

        with open(input_file, 'r') as fin:
            data = json.load(fin)
    
        for item in data:
            dataset[dataset_name][item['id']] = item

    print('number of datasets:', len(dataset))

    # load index file
    with open(args.index_file, 'r') as fin:
        ids = [line.strip() for line in fin.readlines()]

    print('number of samples:', len(ids))

    # sample and write output
    output = list()
    for id in ids:
        dataset_name = id.split('_')[0]
        item = dataset[dataset_name][id]
        output.append(item)

    assert(len(output) == len(ids))

    with open(args.output_file, 'w+') as fout:
        fout.write(json.dumps(output, indent=4))