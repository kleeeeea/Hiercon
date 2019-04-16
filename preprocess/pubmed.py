import json
from lxml import etree
xmlfile = '/disk/home/klee/data/raw/desc2018'
jsonfile = xmlfile + '.json'
jsonfile_hiercon = xmlfile + '_hiercon.json'
uid2hierconlabel_file = xmlfile + '_uid2hierconlabel.json'
# /disk/home/keqian/raw_data/knowledgebase/mesh/desc2018'
INCLUDE = {'DescriptorRecord'}


# , dtd_validation=True, load_dtd=True
def to_json():
    elem_dicts = []
    with open(jsonfile, 'w') as f:
        for _, elem in etree.iterparse(source=xmlfile):
            if elem.tag in INCLUDE:
                print(elem)
                name = elem.find('./DescriptorName/String').text
                descriptorUI = elem.find('./DescriptorUI').text
                treeNumbers = [el.text for el in elem.findall('./TreeNumberList/TreeNumber')]

                elem_dict = {
                    'name': name,
                    'descriptorUI': descriptorUI,
                    'treeNumbers': treeNumbers
                }
                elem_dicts.append(elem_dict)
                print(json.dumps(elem_dict), file=f)


def json_to_hierconLabel():
    uid2hierconlabel = {}
    with open(jsonfile_hiercon, 'w') as f:
        for l in open(jsonfile):
            obj = json.loads(l)
            hierconCat_set = set([t.split('.')[0] for t in obj['treeNumbers']])
            if len(hierconCat_set) == 1:
                del obj['treeNumbers']
                obj['label_hiercon'] = list(hierconCat_set)[0]
                print(json.dumps(obj), file=f)
                uid2hierconlabel[obj['descriptorUI']] = obj['label_hiercon']

    json.dump(uid2hierconlabel, open(uid2hierconlabel_file, 'w'))


json_to_hierconLabel()
