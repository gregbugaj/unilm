import shutil
import sys
import os
import json
from tqdm import tqdm
import argparse

def dict_compare(d1, d2):
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    shared_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys
    modified = {o : (d1[o], d2[o]) for o in shared_keys if d1[o] != d2[o]}
    same = set(o for o in shared_keys if d1[o] == d2[o])
    return added, removed, modified, same


def combine(tt1,tt2):
    """ Combine two COCO annoatated files and save them into new file
    :param tt1: 1st COCO file path
    :param tt2: 2nd COCO file path
    """
    with open(tt1) as json_file:
        d1 = json.load(json_file)
    with open(tt2) as json_file:
        d2 = json.load(json_file)
    b1={}
    for i,j in enumerate(d1['images']):
        b1[d1['images'][i]['id']]=i

    temp=[cc['file_name'] for cc in d1['images']]
    temp2=[cc['file_name'] for cc in d2['images']]
    for i in temp:
        assert not(i in temp2), "Duplicate filenames detected between the two files! @" + i
    

    # Check if both files have the categories dict using only the value and id to compare
    d1_categories_names = {c['name']: c['id'] for c in d1['categories']}
    d2_categories_names = {c['name']: c['id'] for c in d2['categories']}
    
    for c in d1_categories_names:
        # Check if the category name exists in the second file
        if c in d2_categories_names:
            # Check if the category id is the same
            if d1_categories_names[c] != d2_categories_names[c]:
                assert False, 'Category name: {}, id: {} in file 1 and {} in file 2'.format(c, d1_categories_names[c], d2_categories_names[c])
        else:
            assert False, 'Category name: {} in file 1 does not exist in file 2'.format(c)
    
    for c in d2_categories_names:
        if c in d1_categories_names:
            if d1_categories_names[c] != d2_categories_names[c]:
                assert False, 'Category name: {}, id: {} in file 1 and {} in file 2'.format(c, d1_categories_names[c], d2_categories_names[c])
        else:
            assert False, 'Category name: {} in file 2 does not exist in file 1'.format(c)



    files_check_classes={}
    for i,j in enumerate(d1['images']):
        for ii,jj in enumerate(d1['annotations']):
            if jj['image_id']==j['id']:
                try:
                    files_check_classes[j['file_name']].append(jj['category_id'])
                except:
                    files_check_classes[j['file_name']]=[jj['category_id']]

    for i,j in enumerate(d2['images']):
        for ii,jj in enumerate(d2['annotations']):
            if jj['image_id']==j['id']:
                try:
                    files_check_classes[j['file_name']].append(jj['category_id'])
                except:
                    files_check_classes[j['file_name']]=[jj['category_id']]

    b2={}
    for i,j in enumerate(d2['images']):
        b2[d2['images'][i]['id']]=i+max(b1)+1
        
    #Reset File 1 and 2 images ids
    for i,j in enumerate(d1['images']):
        d1['images'][i]['id']= b1[d1['images'][i]['id']]
    for i,j in enumerate(d2['images']):
        d2['images'][i]['id']= b2[d2['images'][i]['id']]
        
    #Reset File 1 and 2 annotations ids
    b3={}
    for i,j in enumerate(d1['annotations']):
        b3[d1['annotations'][i]['id']]=i
    b4={}
    for i,j in enumerate(d2['annotations']):
        b4[d2['annotations'][i]['id']]=max(b3)+i+1



    for i,j in enumerate(d1['annotations']):
        d1['annotations'][i]['id']= b3[d1['annotations'][i]['id']]
        d1['annotations'][i]['image_id']=b1[d1['annotations'][i]['image_id']]
    for i,j in enumerate(d2['annotations']):
        d2['annotations'][i]['id']= b4[d2['annotations'][i]['id']]
        d2['annotations'][i]['image_id']=b2[d2['annotations'][i]['image_id']]

    files_check_classes_temp={}
    pbar = tqdm(total=len(d1['images'])+len(d2['images']))
    for i,j in enumerate(d1['images']):
        for ii,jj in enumerate(d1['annotations']):
            if jj['image_id']==j['id']:
                try:
                    files_check_classes_temp[j['file_name']].append(jj['category_id'])
                except:
                    pbar.update(1)
                    files_check_classes_temp[j['file_name']]=[jj['category_id']]


    for i,j in enumerate(d2['images']):
        for ii,jj in enumerate(d2['annotations']):
            if jj['image_id']==j['id']:
                try:
                    files_check_classes_temp[j['file_name']].append(jj['category_id'])
                except:
                    pbar.update(1)
                    files_check_classes_temp[j['file_name']]=[jj['category_id']]
    pbar.close()
    added, removed, modified, same = dict_compare(files_check_classes, files_check_classes_temp)
    assert (len(added)==0 and len(removed)==0 and len(modified)==0),"filenames detected before merging error: "+len(added)+" filenames added "+ len(removed)+" filenames removed "+len(modified)+" filenames' classes modified "+ len(same)+ " filenames entries reserved"

    test=d1.copy()
    for i in d2['images']:
        test['images'].append(i)
    for i in d2['annotations']:
        test['annotations'].append(i)
    test['categories']=d2['categories']
    files_check_classes_temp={}
    pbar = tqdm(total=len(test['images']))
    for i,j in enumerate(test['images']):
        for ii,jj in enumerate(test['annotations']):
            if jj['image_id']==j['id']:
                try:
                    files_check_classes_temp[j['file_name']].append(jj['category_id'])
                except:
                    pbar.update(1)
                    files_check_classes_temp[j['file_name']]=[jj['category_id']]

    pbar.close()
    added, removed, modified, same = dict_compare(files_check_classes, files_check_classes_temp)
    assert (len(added)==0 and len(removed)==0 and len(modified)==0),"filenames detected after merging error: "+len(added)+" filenames added "+ len(removed)+" filenames removed "+len(modified)+" filenames' classes modified "+ len(same)+ " filenames entries reserved"

    return test



def get_parser():
    parser = argparse.ArgumentParser(description="COCOC dataset merger")
    parser.add_argument(
        "--src_dir",
        help="Path to source directory, all files in this directory and sub-dir will be merged",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_file",
        help="Name of the merged json file.",
        type=str,
    )

    return parser

def merge_dir(src_dir,output_file):
    """ Combine all COCO annoatated files in a directory and save them into new file
    :param src_dir: source directory, all files in this directory and sub-dir will be merged
    :param output_file: output file path
    """
    if not os.path.exists(src_dir):
        raise FileNotFoundError("Directory not found: {}".format(src_dir))

    if not os.path.isdir(src_dir):
        raise FileNotFoundError("Path is not a directory: {}".format(src_dir))
        
    files = []
    for r, d, f in os.walk(src_dir):
        for file in f:
            if '.json' in file:
                files.append(os.path.join(r, file))

    print("Found {} files in {}".format(len(files), src_dir))

    if len(files) == 0:
        print("No files found in {}".format(src_dir))
        return

    if len(files) < 2:
        print("At least 2 files are needed to merge")
        return

    print("Output file: {}".format(output_file))

    if os.path.exists(output_file):
        print("Output file already exists: {}".format(output_file))
        return

    if len(files) == 1:
        print("Only one file found, copying it to {}".format(output_file))
        shutil.copyfile(files[0], output_file)
        return

    print("Merging {} files".format(len(files)))

    file_a = files[0]
    for i in range(1, len(files)):                
        file_b = files[i]
        print(f"Merging : {i}")
        merged = combine(file_a, file_b)

        if True:
            temp_file = "/tmp/temp.json"
            with open(temp_file, 'w', encoding='utf-8') as f:            
                json.dump(merged, f, indent=2)
                f.write('\n')
            file_a = temp_file
    
    shutil.copyfile(file_a, output_file)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    merge_dir(args.src_dir, args.output_file)


if False:
    if __name__ == '__main__':
        if "-h" in sys.argv:
            print('''\nUsage: python {} <path_to_file_1> <path_to_file_2> <output_file>

            Requirements:
            1- There shouldn't be duplicate image_names in the two files
            2- The two files should have the same categories (same names and ids)
            '''.format(sys.argv[0]))
            exit(1)
        if len(sys.argv) <= 3:
            print('\n3 arguments are needed!!')
            print('''\nUsage: python {} <path_to_file_1> <path_to_file_2> <output_file>

            Requirements:
            1- There shouldn't be duplicate image_names in the two files
            2- The two files should have the same categories (same names and ids)
            '''.format(sys.argv[0]))
            exit(1)

        # combine(sys.argv[1],sys.argv[2],sys.argv[3])
        print("\n\nSuccessfully merged the two files ({} , {}) into {}".format(sys.argv[1],sys.argv[2],sys.argv[3]))
