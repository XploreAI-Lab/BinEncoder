import json
import os
from tqdm import tqdm

# 处理微代码（microcode）数据，并将其转换为一种特定的文本格式

mopdict = {'mop_z': 0, 'mop_r': 1, 'mop_n': 2, 'mop_str': 3, 'mop_d': 4, 'mop_S': 5, 'mop_v': 6, 'mop_b': 7,
            'mop_f': 8, 'mop_l': 9, 'mop_a': 10, 'mop_h': 11, 'mop_c': 12, 'mop_fn': 13, 'mop_p': 14, 'mop_sc': 15}

def GetProgramInfo(filename):
    fsplit = filename.split('_')
    compiler = fsplit[1]
    arch = fsplit[2]
    bite = fsplit[3]
    opt = fsplit[4]
    program_name = '{0}-{1}'.format(fsplit[0],fsplit[-1])
    return compiler, arch, bite, opt, program_name

def GetInfo(json_dir, similar_list, program_name, collect_list, tokenizer, bertmodel):
    func_set = set()
    choice = dict()
    for prog in tqdm(similar_list, desc='loading temp dict of {}'.format(program_name)):
        compiler, arch, bite, opt, _ = GetProgramInfo(prog)
        type = '{}_{}_{}_{}'.format(compiler, arch, bite, opt)
        subdict = dict()
        reading_path = os.path.join(json_dir, prog)
        with open(reading_path, 'r') as fp:
            for line in fp.readlines():
                data = json.loads(line)
                fname = data['fname']
                func_set.add(fname)
                mc = GetFuncMicrocode(data['regu_mc'])
                mc_inputs = tokenizer(
                    mc,
                    truncation=True,
                    max_length=512,
                    padding='max_length',
                    return_tensors='pt').to(device)
                vec = bertmodel(mc_inputs).detach().cpu()
                subdict[fname] = vec
            choice[type] = subdict

    for func in tqdm(func_set,desc='merging {} by index of function name'.format(program_name)):
        func_dict = dict()
        func_dict['prog'] = program_name
        func_dict['function_name'] = func
        for choice_key in choice.keys():
            if func in choice[choice_key].keys():
                func_dict[choice_key] = choice[choice_key][func]
        collect_list.append(func_dict)


def GetFuncMicrocode(regu_mc):
    func_regmc = ''
    n = 1
    for bdata in regu_mc:
        func_regmc = func_regmc + 'start@{} '.format(n)
        if bdata:
            for idata in bdata:
                opcode = idata[0]
                l = idata[1]
                r = idata[2]
                d = idata[3]
                rel = mop_prosess(l)
                rer = mop_prosess(r)
                red = mop_prosess(d)
                func_regmc = func_regmc + '<' + opcode + '>' + ' ' + rel + ' ' + rer + ' ' + red + ' '
            n += 1
    return func_regmc

def mop_prosess(oprand):
    if not oprand:
        return ''
    mop = oprand[0]
    moptype = oprand[1]
    if mopdict[moptype] == 0:
        return ''
    elif mopdict[moptype] == 4:
        sub_opcode = mop[0]
        # ret = '<{}>'.format(sub_opcode)
        rel = mop_prosess(mop[1])
        rer = mop_prosess(mop[2])
        red = mop_prosess(mop[3])
        # for subop in [mop[1], mop[2], mop[3]]:
        #     ret = ret + mop_prosess(subop)
        return '[' + '<{0}> {1} {2} {3}'.format(sub_opcode, rel, rer, red).strip() + ']'  # '[<sub_opcode> rel rer red]'
    elif mopdict[moptype] == 8:
        if mop[0] == 'voidf':
            # add(staticdict, mop[0])
            return 'voidf'
        else:
            ret = ''
            for subop in mop:
                # try:
                ret = ret + mop_prosess(subop) + ' '
                # except:
                #     print('123')
            return '[{}]'.format(ret.strip())

    elif mopdict[moptype] == 9:
        defblk = oprand[2]
        if 'argv' in mop:
            return mop.strip()
        else:
            if 'lvname' in mop:
                if defblk >= 200:
                    return 'lvname'
                else:
                    return 'lvname@{}'.format(defblk)
            elif 'resultlv' in mop:
                if defblk >= 200:
                    return 'resultlv'
                else:
                    return 'resultlv@{}'.format(defblk)
            elif 'lvar' in mop:
                if defblk >= 200:
                    return 'lvar'
                else:
                    return 'lvar@{}'.format(defblk)
            else:
                return 'unusedlv'

    elif mopdict[moptype] == 14:
        # ret = ''
        hop = mop_prosess(mop[0])
        lop = mop_prosess(mop[1])
        # for i in mop:
        #     # try:
        #     ret = ret + mop_prosess(i)
        # except:
        #     print('123')
        return '{0}:{1}'.format(hop, lop).strip()
    elif mopdict[moptype] == 12:
        ret = ''
        for i in mop:
            ret = ret + i + ' '
        return ret.strip()
    else:
        return mop.strip()



if __name__ == '__main__':

    mclist = [[["mov", [["call", ["mop_v", "mop_v"], ["", "mop_z"], [[["argv0", "mop_l", 0], ["0", "mop_n"]], "mop_f"]], "mop_d"], ["", "mop_z"], ["lvname3", "mop_l", 1]], ["jz", ["lvname3", "mop_l", 1], ["0", "mop_n"], ["JUMP@4", "mop_b"]]], [["jnz", [["ldx", ["cs.2", "mop_r"], [["add", ["lvname3", "mop_l", 1], ["constn.4", "mop_n"], ["", "mop_z"]], "mop_d"], ["", "mop_z"]], "mop_d"], ["constn.4", "mop_n"], ["JUMP@4", "mop_b"]]], [["ldx", ["cs.2", "mop_r"], [["add", [["ldx", ["cs.2", "mop_r"], [["add", ["lvname3", "mop_l", 1], ["constn.4", "mop_n"], ["", "mop_z"]], "mop_d"], ["", "mop_z"]], "mop_d"], ["constn.4", "mop_n"], ["", "mop_z"]], "mop_d"], ["lvar1", "mop_l", 3]], ["goto", ["JUMP@5", "mop_b"], ["", "mop_z"], ["", "mop_z"]]], [["mov", ["0", "mop_n"], ["", "mop_z"], ["lvar1", "mop_l", 3]]], [["mov", ["lvar1", "mop_l", 3], ["", "mop_z"], ["resultlv", "mop_l", 5]]]]
    mc = GetFuncMicrocode(mclist)
    with open('static.txt','w') as fp:
        fp.write(mc)











