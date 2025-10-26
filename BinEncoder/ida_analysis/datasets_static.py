import json
import os

fn=0
bb=0
In=0
SmallFN=0
MediumFN=0
LargeFN=0
complex_inst=0

# 统计复杂指令、函数、基本块数量

complexx=["mop_f","mop_c","mop_p","mop_d"]

# print('hello')
dir='H:\\jxd_mc\\mc_jsons\\ultimate_pretrain_json'
files = os.listdir(dir)
for file in files:
    with open(os.path.join(dir,file),'r') as fp:
        for line in fp:
            data = json.loads(line)
            func_data=data['regu_mc']
            fn+=1
            if len(func_data)<=10:
                SmallFN+=1
            elif len(func_data)<=100 and len(func_data)>10:
                MediumFN+=1
            else:
                LargeFN+=1
            for block_data in func_data:
                bb+=1
                for inst_data in block_data:
                    In+=1
                    for op in inst_data:
                        if type(op)==type('jxd'):
                            continue
                        elif op[1] in complexx:
                            complex_inst +=1
                            break
    print('fn:{}   bb:{}   In:{}   SmallFN:{}   MediumFN:{}   LargeFN:{}   complex_inst:{}'.format(fn,bb,In,SmallFN,MediumFN,LargeFN,complex_inst))



