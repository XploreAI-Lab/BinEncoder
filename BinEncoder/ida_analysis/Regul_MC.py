import os
import re
import json

TOKEN_MAGIN = 200
# 从特定目录中的 JSON 文件提取和规范化微操作，并生成指定的数据集和操作码列表。

def get_regul_mcoprand(oppair, static_dict):
    oprand = oppair[0]
    oprtype = oppair[1]
    regul_oprand = ''
    byte = oprand.split('.')[-1]
    if oprtype == 'mop_z':
        regul_oprand = 'mop_z'

    elif oprtype == 'mop_d':
        regul_oprand = 'mop_d'

    elif oprtype == 'mop_r':
        if oprand.startswith('kr'):
            regul_oprand = 'kr.' + byte
        elif oprand.startswith('xmm'):
            regul_oprand = 'xmm.' + byte
        elif '^' in oprand:
            regul_oprand = 'mopr_^.' + byte
        elif static_dict[oprand] < TOKEN_MAGIN:
            regul_oprand = 'mop_r.' + byte
        else:
            regul_oprand = oprand

    elif oprtype == 'mop_n':
        regul_oprand = 'mop_n.' + byte

    elif oprtype == 'mop_S':
        if oprand.startswith(r'%var'):
            regul_oprand = 'var.' + byte
        elif oprand.startswith(r'%0x'):
            regul_oprand = 'num_s.' + byte
        elif oprand.startswith(r'%arg_'):
            regul_oprand = 'arg.' + byte
        elif r'@' in oprand:
            regul_oprand = 'mop_s@.' + byte
        elif static_dict[oprand] < TOKEN_MAGIN:
            regul_oprand = 'mop_S.' + byte
        else:
            regul_oprand = oprand

    elif oprtype == 'mop_v':
        regul_oprand = 'mop_v'

    elif oprtype == 'mop_b':
        regul_oprand = 'mop_b'

    elif oprtype == 'mop_a':
        regul_oprand = 'mop_a.' + byte

    elif oprtype == 'mop_c':
        regul_oprand = 'mop_c'

    elif oprtype == 'mop_h':
        regul_oprand = 'mop_h'

    elif oprtype == 'mop_f':
        regul_oprand = 'mop_f'

    elif oprtype == 'mop_p':
        regul_oprand = 'mop_p'

    elif oprtype == 'mop_fn':
        regul_oprand = 'mop_fn.' + byte
    # 下面三种情况在libsqlite中没有出现
    # elif oprtype == 'mop_str':
    #     regul_oprand = 'mop_str'
    # elif oprtype == 'mop_l':
    #     regul_oprand = 'mop_l'
    # elif oprtype == 'mop_sc':
    #     regul_oprand = 'mop_sc'
    else:
        regul_oprand = oprtype
        print('unexpected optype situation!')
    return regul_oprand

    

if __name__ == '__main__':

    dataset_dir = 'datasets'
    regul_datesetdir = 'regul_datasets'

    files = os.listdir(dataset_dir)

    total_static = dict()
    add_vocal = set()
    

    # 统计token数量
    total_static = dict()
    for file in files:
        path = os.path.join(dataset_dir,file)
        with open(path, 'r') as jsonfp:
            for line in jsonfp:
                data = json.loads(line)
                for bdata in data['microcode']:
                    for idata in bdata:
                        for oprand in [idata[1], idata[2], idata[3]]:
                            if total_static.__contains__(oprand[0]):
                                total_static[oprand[0]] = total_static[oprand[0]] + 1
                            else:
                                total_static[oprand[0]] = 1

    for file in files:
        path = os.path.join(dataset_dir,file)
        print(path)
        regul_path = os.path.join(regul_datesetdir,file)
        with open(path, 'r') as jsonfp:
            save_f = open(regul_path, 'w')
            for line in jsonfp:
                data = json.loads(line)
                regul_mc = []
                for bdata in data['microcode']:
                    regul_bmc = []
                    for idata in bdata:
                        regul_imc = []
                        regul_imc.append(idata[0])
                        for oppair in [idata[1], idata[2], idata[3]]:
                            regul_op = get_regul_mcoprand(oppair,total_static)
                            regul_imc.append([regul_op,oppair[1]])
                            add_vocal.add(regul_op)
                        regul_bmc.append(regul_imc)
                    regul_mc.append(regul_bmc)
                # save new json
                data['reg_mc'] = regul_mc
                func_data = json.dumps(data)
                save_f.write(func_data + '\n')
            save_f.close()
    
    vocalfp = open('add_vocal.txt', 'w')

    # 写入操作码
    mopcode_list = ['nop', 'stx', 'ldx', 'ldc', 'mov', 'neg', 'lnot', 'bnot', 'xds', 'xdu', 'low', 'high', 'add', 'sub', 'mul', 'udiv', 'sdiv', 'umod',
            'smod', 'or', 'and', 'xor',  'shl', 'shr', 'sar', 'cfadd', 'ofadd', 'cfshl', 'cfshr', 'sets', 'seto', 'setp', 'setnz', 'setz', 'setae',
            'setb', 'seta', 'setbe', 'setg', 'setge', 'setl', 'setle', 'jcnd', 'jnz', 'jz', 'jae', 'jb', 'ja', 'jbe', 'jg', 'jge', 'jl', 'jle',
            'jtbl', 'ijmp', 'goto', 'call', 'icall', 'ret', 'push', 'pop', 'und', 'ext', 'f2i', 'f2u', 'i2f', 'u2f', 'f2f', 'fneg', 'fadd', 'fsub',
            'fmul', 'fdiv']
    for i in mopcode_list:
        add_vocal.add( '<' + i + '>')
    # 写入规范化后的操作数
    for i in add_vocal:
        vocalfp.write(str(i) + '\n')

    vocalfp.write( '<empty>'+ '\n') # 空的基本块
    vocalfp.close()



