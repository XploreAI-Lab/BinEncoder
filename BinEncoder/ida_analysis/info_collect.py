# -*- coding: UTF-8 -*-
import idc
import idaapi
import idautils
import networkx as nx
import json
import os
import re

# is_debug = True
is_debug = False
MC_OPT = idaapi.MMAT_LVARS
DEBUGNUM = 50

# 从 IDA Pro 中提取函数信息的 Python 脚本。主要用于获取二进制文件中的函数的汇编代码、中间表示（microcode）、调用关系等信息，并将结果保存为 JSON 格式，相比jtrans多了调用关系

def get_func_asm_code_list(allblock):
    func_asm_code_list = []
    block_num = 0
    for block in allblock:
        block_num = block_num + 1
        block_asm_code_list = []
        inst_addr = block.start_ea
        b_eea = block.end_ea
        while inst_addr < b_eea:
            inst_asm_code_list = []
            inst_asm_code_list.append(idc.print_insn_mnem(inst_addr))
            opcnt = 0
            while (idc.get_operand_type(inst_addr, opcnt)!=0):
                inst_asm_code_list.append(idc.print_operand(inst_addr,opcnt))
                opcnt = opcnt +1
            block_asm_code_list.append(inst_asm_code_list)
            inst_addr = idc.next_head(inst_addr)
        func_asm_code_list.append(block_asm_code_list)
    return func_asm_code_list,block_num


def get_norm_func_code_list(allblock,arch):
    func_norm_code_list = []
    for block in allblock:
        block_norm_code_list = []
        inst_addr = block.start_ea
        b_eea = block.end_ea
        while inst_addr < b_eea:
            inst_norm_code_list = []
            inst_norm_code_list.append(idc.print_insn_mnem(inst_addr))
            opcnt = 0
            while (idc.get_operand_type(inst_addr, opcnt) != idc.o_void):
                inst_norm_code_list.append(Fit_Normalize_Oprand(inst_addr,opcnt,arch))
                opcnt = opcnt +1
            block_norm_code_list.append(inst_norm_code_list)
            inst_addr = idc.next_head(inst_addr)
        func_norm_code_list.append(block_norm_code_list)
    return func_norm_code_list


def Fit_Normalize_Oprand(inst_addr,offset,arch):
    #最后对第五类情况的处理好像有点不对，然后把[rax+rdx]这种情况忽略了
    opType = idc.get_operand_type(inst_addr, offset)
    res = ''
    if opType == idc.o_far or opType == idc.o_near:  # 6 7
        res = 'FOO'
    elif opType == idc.o_reg:  # 1
        res = idc.print_operand(inst_addr, offset)  # x86, arm
    elif opType == idc.o_displ:  # 4
        if arch == 'ARM':
            tmp = idc.print_operand(inst_addr, offset).split(',')
            if len(tmp) == 1:
                tmp = tmp[0][1:-1]
            else:
                if tmp[0][-1] == ']':
                    tmp = tmp[0][1:-1]
                else:
                    tmp = tmp[0][1:]
            res = '[{}+0]'.format(tmp)
        elif arch == 'metapc':
            res = '[{}+0]'.format(idc.print_operand(inst_addr, offset).split('+')[0][1:])  # x86
        elif arch == 'mipsb':
            res = '[{}+0]'.format(idc.print_operand(inst_addr, offset).split('$')[-1][:-1])   # mips
    elif opType == idc.o_idpspec1:
        if arch == 'ARM':
            res = idc.print_operand(inst_addr, offset)
        else:
            pass
    else:
        strings, consts = getConst(inst_addr, offset)  # 5
        
        if strings and not consts:
            res = '<STR>'
        elif consts and not strings:
            res = '0'
        else:
            res = '<TAG>'
        breakpoint()
    return res


def getConst(ea, offset):
    strings = []
    consts = []
    optype1 = idc.get_operand_type(ea, offset)
    if optype1 == idc.o_imm:
        imm_value = idc.get_operand_value(ea, offset)
        if 0<= imm_value <= 10:
            consts.append(imm_value)
        else:
            if idaapi.is_loaded(imm_value) and idaapi.getseg(imm_value):
                str_value = idc.get_strlit_contents(imm_value)
                if str_value is None:
                    str_value = idc.get_strlit_contents(imm_value+0x40000)
                    if str_value is None:
                        consts.append(imm_value)
                    else:
                        
                        re = all(40 <= c < 128 for c in str_value)
                        if re:
                            breakpoint()
                            strings.append(str_value)
                        else:
                            consts.append(imm_value)
                else:
                    re = all(40 <= c < 128 for c in str_value)
                    if re:
                        strings.append(str_value)
                    else:
                        consts.append(imm_value)
            else:
                consts.append(imm_value)
    return strings, consts


def get_func_adj_table(allblock):
    cfg = nx.DiGraph()
    adj_list = []

    block_dict = dict()
    # 基本块按地址编号
    for i,block in enumerate(allblock):
        addr = hex(block.start_ea)
        block_dict[addr] = i
    # 生成邻接表
    for block in allblock:
        cfg.add_node(block_dict[hex(block.start_ea)])
        for succ_block in block.succs():
            cfg.add_edge(block_dict[hex(block.start_ea)],block_dict[hex(succ_block.start_ea)])
        for pred_block in block.preds():
            cfg.add_edge(block_dict[hex(pred_block.start_ea)],block_dict[hex(block.start_ea)])
    for n, succ in cfg.adjacency():
        adj_list.append(list(succ.keys()))

    return adj_list


def get_func_str(allblock):
    pass


def get_xref_func_name(sea):

    fname = idc.get_func_name(func)
    xref_set_from = set()       #该函数调用的函数
    xref_set_to = set()         #调用该函数的函数
    for item in idautils.FuncItems(sea):

        #获取调用关系
        xrefs_from = idautils.XrefsFrom(item)
        xrefs_to = idautils.XrefsTo(item)

        for xfrom in xrefs_from:
            if idaapi.get_func_name(xfrom.to):
                xref_set_from.add(idaapi.get_func_name(xfrom.to))
        
        for xto in xrefs_to:
            if idaapi.get_func_name(xto.frm):
                xref_set_to.add(idaapi.get_func_name(xto.frm))

    return ['<self>' if f == fname else f for f in list(xref_set_from)] , ['<self>' if f == fname else f for f in list(xref_set_to)]


mopcode_list = ['nop', 'stx', 'ldx', 'ldc', 'mov', 'neg', 'lnot', 'bnot', 'xds', 'xdu', 'low', 'high', 'add', 'sub',
                'mul', 'udiv', 'sdiv', 'umod',
                'smod', 'or', 'and', 'xor',  'shl', 'shr', 'sar', 'cfadd', 'ofadd', 'cfshl', 'cfshr', 'sets', 'seto',
                'setp', 'setnz', 'setz', 'setae',
                'setb', 'seta', 'setbe', 'setg', 'setge', 'setl', 'setle', 'jcnd', 'jnz', 'jz', 'jae', 'jb', 'ja',
                'jbe', 'jg', 'jge', 'jl', 'jle',
                'jtbl', 'ijmp', 'goto', 'call', 'icall', 'ret', 'push', 'pop', 'und', 'ext', 'f2i', 'f2u', 'i2f', 'u2f',
                'f2f', 'fneg', 'fadd', 'fsub',
                'fmul', 'fdiv']

moprand_list = ['mop_z', 'mop_r', 'mop_n', 'mop_str', 'mop_d', 'mop_S', 'mop_v', 'mop_b', 'mop_f',
                'mop_l', 'mop_a', 'mop_h', 'mop_c', 'mop_fn', 'mop_p', 'mop_sc']

'''
mop_z = cvar.mop_z
r"""
none
"""

mop_r = cvar.mop_r
r"""
register (they exist until MMAT_LVARS)
"""

mop_n = cvar.mop_n
r"""
immediate number constant
"""

mop_str = cvar.mop_str
r"""
immediate string constant
"""

mop_d = cvar.mop_d
r"""
result of another instruction
"""

mop_S = cvar.mop_S
r"""
local stack variable (they exist until MMAT_LVARS)
"""

mop_v = cvar.mop_v
r"""
global variable
"""

mop_b = cvar.mop_b
r"""
micro basic block (mblock_t)
"""

mop_f = cvar.mop_f
r"""
list of arguments
"""

mop_l = cvar.mop_l
r"""
local variable
"""

mop_a = cvar.mop_a

mop_h = cvar.mop_h
r"""
helper function
"""

mop_c = cvar.mop_c
r"""
mcases
"""

mop_fn = cvar.mop_fn
r"""
floating point constant
"""

mop_p = cvar.mop_p
r"""
operand pair
"""

mop_sc = cvar.mop_sc
r"""
scattered
"""
'''


def moprand_process(mop,call_list):
    # ea = insn.ea
    moptype = moprand_list[mop.t]
    mopstr = ''
    if mop.t == 0:      # mop_z , none
        return [mopstr, moptype]
    
    elif mop.t == 1:    # mop_r , register , eg: cs.2{12} , 把括号去掉
        match_s = re.search(r'({\d+?}$)',mop.dstr())
        if match_s:      #匹配括号
            return [mop.dstr().replace(match_s.group(),''), moptype]
        else:
            return [mop.dstr(), moptype]

    elif mop.t == 2:    # mop_n , immediate number constant
        # breakpoint()
        if mop.is01():
            return [str(mop.nnn.value) , moptype]
        else:
            return ['constn.{}'.format(mop.size), moptype]

    elif mop.t == 3:    # mop_str , immediate string constant
        return['mop_str.{}'.format(mop.size), moptype]

    elif mop.t == 4:    # mop_d , result of another instruction
        
        subins = mop.d
        sub_opcode = mopcode_list[subins.opcode]
        l = subins.l
        r = subins.r
        d = subins.d
        subinst_mc_info = [sub_opcode, moprand_process(l,call_list), moprand_process(r,call_list), moprand_process(d,call_list)]
        return [subinst_mc_info, moptype] # 返回的列表加入一维表示该操作数有子指令，可以递归

    elif mop.t == 5:    # mop_S ,  local stack variable
        breakpoint()
        # exit()

    elif mop.t == 6:    # mop_v , global variable
        # breakpoint()
        if mop.size==-1:
            return['mop_v', moptype]
        else:
            return['mop_v.{}'.format(mop.size), moptype]
        # breakpoint()
        # exit()

    elif mop.t == 7:    # mop_b , micro basic block
        return ['JUMP' + mop.dstr(), moptype]

    elif mop.t == 8:    # mop_f , list of arguments
        args = mop.f.args
        callee_addr = mop.f.callee                                               # ！！！！！！
        callee_name = idc.get_func_name(callee_addr)
        call_list.append(callee_name)
        mc_info = []
        for i in range(mop.f.solid_args):
            mc_info.append(moprand_process(args.at(i),call_list))
        if len(mc_info) ==0:
            mc_info.append('voidf')
        return[mc_info, moptype]

    elif mop.t == 9:    # mop_l , local variable
        # mba_vars = mop.l.mba.vars()
        var = mop.l.var()
        var_idx = mop.l.idx #mop在函数mba中的索引
        defblk = var.defblk
        if var.used:
            if var.is_arg_var:      #函数参数
                return ['argv{}'.format(var_idx), moptype, defblk]#加入一维表示定义该变量的基本块
            
            elif var.is_result_var: #结果
                return ['resultlv', moptype, defblk]
            
            elif var.has_nice_name:     # 有自己的变量名
                return ['lvname{}'.format(var_idx), moptype, defblk]
            
            else:
                return ['lvar{}'.format(var_idx), moptype, defblk]#ida分配变量名的变量
            
        else:
            return ['unusedlv{}'.format(var_idx), moptype, defblk]

    elif mop.t == 10:   # mop_a 
        
        return['mop_a.{}'.format(mop.size), moptype]

    elif mop.t == 11:   # mop_h  helper function

        return['mop_h', moptype]

    elif mop.t == 12:   # mop_c  switch case 
        # breakpoint()
        # exit()
        targets = mop.c.targets
        switch_blk = []
        for i in range(targets.size()):
            switch_blk.append('SWITCH@{}'.format(targets.at(i)))

        return[switch_blk, moptype]

    elif mop.t == 13:   # mop_fn  floating point constant
        
        return['floatnum.{}'.format(mop.size), moptype]

    elif mop.t == 14:   # mop_p  operand pair
        # breakpoint()
        # exit()
        hop = mop.pair.hop
        lop = mop.pair.lop
        oppair = [moprand_process(hop,call_list),moprand_process(lop,call_list)]
        return[oppair, moptype]

    elif mop.t == 15:   # mop_sc  scattered
        breakpoint()
        # exit()
        return['mop_sc', moptype]
    else:
        return ['oov','out_of_voc']


def get_mc(mba,mblock_num):
    regu_mc = []
    ori_mc = []
    func_call_list = []
    # microcode 基本块最前面和最后面的两个基本块分别为整个mba的入口和出口，里面没有语句，是虚拟的基本块，在这把它们排除
    for i in range(mblock_num - 2):
        mblock = mba.get_mblock(i+1) #从1开始排号，跳过第0个
        head = mblock.head
        regublock_mc_list = []
        oriblock_mc_list = []
        blockcall_list = []#调用信息在处理操作数时一并获取                                 #！！！！！！！！！！！！！！
        # if not head:
            # print('empty microblock !')
            # breakpoint()
            # break
        cur = head
        while(cur):
            opcode_num = cur.opcode 
            opcode_str = mopcode_list[opcode_num]
            l = cur.l
            r = cur.r
            d = cur.d
            oriinst_mc_info = [opcode_str, [l.dstr(), moprand_list[l.t]], [r.dstr(), moprand_list[r.t]], [d.dstr(), moprand_list[d.t]]]
            reguinst_mc_info = [opcode_str, moprand_process(l,blockcall_list), moprand_process(r,blockcall_list), moprand_process(d,blockcall_list)]
            oriblock_mc_list.append(oriinst_mc_info)
            regublock_mc_list.append(reguinst_mc_info)
            cur = cur.next
        ori_mc.append(oriblock_mc_list)
        regu_mc.append(regublock_mc_list)
        func_call_list.append(blockcall_list)
    return ori_mc , regu_mc, func_call_list


def GetCallString(nodecall):
    if not nodecall:
        return ''
    str = ' '.join(nodecall).strip()
    if len(str)>0:
        return str
    else:
        return ''

def GetInnerCallGraph(G,adj,call_list):
    del_list = []       # the node to delate
    ICG = []            # the graph been cut off
    ICGstr = []         # the calling function name of ICG
    for i in range(len(adj)):
        callstr = GetCallString(call_list[i])
        if not callstr:
            del_list.append(i)
        else:
            ICGstr.append(callstr)

    for delnode in del_list:
        succ = list(G.successors(delnode))
        pred = list(G.predecessors(delnode))
        for p in pred:
            for s in succ:
                G.add_edge(p,s)
        G.remove_node(delnode)

    left_nodes = list(G.nodes())
    cut_node_dict = dict()      # key: original node number   ,    value: new node number
    cnt = 0
    for i in range(len(adj)):   # reset a new number of nodes
        if i in left_nodes:
            cut_node_dict[i] = cnt
            cnt += 1

    for node in left_nodes:
        succ = list(G.successors(node))
        newsucc = []
        for s in succ:
            newsucc.append(cut_node_dict[s])
        ICG.append(newsucc)

    return ICG,ICGstr

def get_mc_graph_adjtable(graph, func_call_list):

    g = nx.DiGraph()
    adj_list = []
    # if(graph.nsucc(0)>1):
    #     print('yes')
    for i in range(graph.size()-2):
        g.add_node(i)
        nsucc = graph.nsucc(i+1)
        for n in range(nsucc):# n表示第n个后续节点
            succ = graph.succ(i+1, n)
            if succ != graph.size()-1:
                g.add_edge(i, succ-1)

    for i in range(len(g.nodes)):
        adj_list.append(list(g.successors(i)))

    ICG, ICGstr = GetInnerCallGraph(G=g,adj=adj_list,call_list=func_call_list)
    # for n,succ in g.adjacency():
    #     adj_list.append(list(succ.keys()))
    return adj_list, ICG, ICGstr


def get_func_microcode_info(func , Opt_Level):

    funcaddr = idaapi.get_func(func)
    hf = idaapi.hexrays_failure_t()
    mc_graph_adjtable = []
    mc_vars = []
    nice_name = []
    ori_mc = []
    regu_mc = []
    func_call_list = []
    ICG = []
    ICGstr = []
    try:
        mbr = idaapi.mba_ranges_t(funcaddr)
        mba = idaapi.gen_microcode(mbr, hf, None, idaapi.DECOMP_WARNINGS, Opt_Level)
        graph = mba.get_graph()
        mblock_num = graph.size()
        mc_vars, nice_name = get_mc_vars(mba)
        ori_mc, regu_mc ,func_call_list = get_mc(mba,mblock_num)
        mc_graph_adjtable, ICG, ICGstr = get_mc_graph_adjtable(graph,func_call_list)        #用不上图了
    except:
        return ori_mc, regu_mc, func_call_list, mc_graph_adjtable, ICG, ICGstr, mc_vars, nice_name
    
    if is_debug:
        print('vars:' , mc_vars)
        print('nice_name:' , nice_name)

    return ori_mc, regu_mc, func_call_list, mc_graph_adjtable, ICG, ICGstr, mc_vars, nice_name

def get_mc_vars(mba):
    #获取局部变量名
    vars = []
    nice_name = []
    for i in range(mba.vars.size()):
        v = mba.vars.at(i)
        vars.append(v.name)
        if(v.used and v.has_nice_name):
            nice_name.append(v.name)
    return vars, nice_name


def McIsError(mc):
    if len(mc['regu_mc']) == 0:
        return True
    for bdata in mc['regu_mc']:
        if len(bdata)==0:
            return True
    
    return False


if __name__ == '__main__':

    idaapi.auto_wait()
    
    #获取当前架构信息
    arch = idaapi.get_inf_structure().procname
    print('arch:', arch)

    # 转到.text段
    seg_sea = 0
    seg_eea = 0
    for seg in idautils.Segments():
        if(idc.get_segm_name(seg) == '.text'):
            seg_sea = idc.get_segm_start(seg)
            seg_eea = idc.get_segm_end(seg)
            break

    #存入文件
    bin_name = idc.get_input_file_path().split(os.sep)[-1]
    saving_dir = 'E:\\项目\\ida_analysis\\call_graph_new_json'
    # saving_path = saving_dir + '\\' + bin_name + '.json'
    if is_debug:
        saving_path = os.path.join(saving_dir, bin_name + '_debug.json')
    else:
        saving_path = os.path.join(saving_dir, bin_name + '.json')
    if os.path.exists(saving_path):
        idc.qexit(0)

    with open(saving_path,'w') as save_f:
        print('processing:',saving_path)
        # 分函数处理.text段数据
        func_cnt = 0
        for func in idautils.Functions(seg_sea,seg_eea):
        
            # Ignore Library Code
            flag = idc.get_func_flags(func)
            if flag & idc.FUNC_LIB:
                continue

            if func_cnt%100 ==0:
                print('{} function loaded.'.format(func_cnt))
            # 保存类型：字典
            fdict = dict()

            if is_debug and func_cnt > DEBUGNUM:        #调试
                break
               
            # func name
            func_name = idc.get_func_name(func)
            # print('cur:',func_name)

            fdict['fname'] = func_name

            '''
            # get all basic block of current function
            # allblock = idaapi.FlowChart(idaapi.get_func(func))


            # 初始汇编代码 & 基本块数量
            # fdict['ori_asmcode'] , fdict['block_num'] = get_func_asm_code_list(allblock)
        
            # 经过规范化的代码
            # func_reg_code_list = []
            # for block in allblock:
            #     block_reg_code_list = []
            #     inst_addr = block.start_ea
            #     b_eea = block.end_ea
            #     while inst_addr < b_eea:
            #         block_reg_code_list.append(Fit_Normalization(inst_addr,arch))
            #         inst_addr = idc.next_head(inst_addr)
            #     func_reg_code_list.append(block_reg_code_list)
            # fdict['re_asmcode'] = get_norm_func_code_list(allblock,arch)

            # 邻接表
            # fdict['adj_table'] = get_func_adj_table(allblock)


            # 使用的字符串
            # fdict['func_str'] = get_func_str(allblock)
            '''
            # return ori_mc, regu_mc, func_call_list, mc_graph_adjtable, ICG, ICGstr, mc_vars, nice_name
            # 中间表示
            # fdict['microcode'], fdict['micrograph'] ,fdict['mc_vars']= get_func_microcode_info(func, MC_OPT)
            fdict['ori_mc'], fdict['regu_mc'], fdict['call_list'], fdict['mc_graph_adj'],fdict['ICG'],fdict['ICGstr'],\
            fdict['mc_vars'], fdict['nice_name'] = get_func_microcode_info(func, MC_OPT)

            if McIsError(fdict):
                continue

            # 函数调用,被调
            fdict['callee'], fdict['caller'] = get_xref_func_name(func)

            func_cnt = func_cnt + 1

            func_data = json.dumps(fdict)

            save_f.write(func_data + '\n')

        print('{} data collected!'.format(func_cnt))
    idc.qexit(0)

