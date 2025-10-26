# -*- coding: UTF-8 -*-
import idc
import idaapi
import idautils
import networkx as nx
import json
import os
import re

MC_OPT = idaapi.MMAT_LVARS









if __name__ == '__main__':

    idaapi.auto_wait()
    ea = idaapi.get_screen_ea()
    funcaddr = idaapi.get_func(ea)
    breakpoint()
    hf = idaapi.hexrays_failure_t()

    mbr = idaapi.mba_ranges_t(funcaddr)
    mba = idaapi.gen_microcode(mbr, hf, None, idaapi.DECOMP_WARNINGS, MC_OPT)
    # graph = mba.get_graph()
    # mblock_num = graph.size()
    for i in range(3):
        mblock = mba.get_mblock(i+1) #从1开始排号，跳过第0个
        head = mblock.head
        breakpoint()



