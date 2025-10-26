import os
import json
# 用于统计和记录微代码中操作数的信息的 Python 脚本。它的主要目的是从指定的 JSON 文件中提取操作数并进行统计，特别是针对各种类型的操作数，输出统计结果到一个文本文件中
mopdict = {'mop_z': 0, 'mop_r': 1, 'mop_n': 2, 'mop_str': 3, 'mop_d': 4, 'mop_S': 5, 'mop_v': 6, 'mop_b': 7,
            'mop_f': 8, 'mop_l': 9, 'mop_a': 10, 'mop_h': 11, 'mop_c': 12, 'mop_fn': 13, 'mop_p': 14, 'mop_sc': 15}

def add(staticdict, opstr):
    if opstr:
        if staticdict.__contains__(opstr):
            total_static[opstr] = total_static[opstr] + 1
        else:
            total_static[opstr] = 1

def add_token_to_dict(staticdict,oprand):
    mop = oprand[0]
    moptype = oprand[1]
    if mopdict[moptype] == 0:
        return
    elif mopdict[moptype] == 4:
        for subop in [mop[1],mop[2],mop[3]]:
            add_token_to_dict(staticdict,subop)
    elif mopdict[moptype] == 8:
        if mop[0] =='voidf':
            add(staticdict,mop[0])
        else:
            for subop in mop:
                try:
                    add_token_to_dict(staticdict,subop)
                except:
                    print('123')
    elif mopdict[moptype] == 14 :
        for i in mop:
            try:
                add_token_to_dict(staticdict,i)
            except:
                print('123')
    elif mopdict[moptype] == 12:
        for i in mop:
            add(staticdict,i)
    else:
        try:
            add(staticdict,mop)
        except:
            print('123')
    
    return staticdict





dataset_dir = os.path.join(os.path.dirname(__file__), 'datasets')

files = os.listdir(dataset_dir)

resf = open('static.txt', 'w')

total_static = dict()

for file in files:
    path = os.path.join(dataset_dir,file)
    with open(path, 'r') as jsonfp:
        for line in jsonfp:
            data = json.loads(line)
            for bdata in data['regu_mc']:
                for idata in bdata:
                    # if idata[0]:
                    #     if total_static.__contains__(idata[0]):
                    #         total_static[idata[0]] = total_static[idata[0]] + 1
                    #     else:
                    #         total_static[idata[0]] = 1

                    for oprand in [idata[1], idata[2], idata[3]]:
                        if oprand[0]:
                            total_static = add_token_to_dict(total_static,oprand)


                    # opcode = idata[0]
                    # (l, lt) = idata[1]
                    # (r, rt) = idata[2]
                    # (d, dt) = idata[3]
                    # for op in [opcode,l,r,d]:
                    #     if op:
                    #         if total_static.__contains__(op):
                    #             total_static[op] = total_static[op] + 1
                    #         else:
                    #             total_static[op] = 1


t = 0
for k,v in total_static.items():
    t = t + v
    # if v>1000:
    resf.write(k + ',' + str(v) + '\n')


resf.write('total: ' + str(t) + '\n')
resf.close()