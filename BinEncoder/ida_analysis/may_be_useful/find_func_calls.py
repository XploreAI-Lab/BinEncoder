import idc
import idautils
import idaapi
import ida_kernwin

cur = ida_kernwin.get_screen_ea()
func = idaapi.get_func(cur)

sea = func.start_ea
eea = func.end_ea
called_func_name = set()
for item in idautils.FuncItems(sea):

    insn = idaapi.insn_t()
    idaapi.decode_insn(insn,item)

    called_func_eas = insn.ops
    for called_ea in called_func_eas:
        addr = called_ea.addr
        if idaapi.get_func_name(addr):
            called_func_name.add(idaapi.get_func_name(addr))




xref_set_from = set()
xref_set_to = set()
for item in idautils.FuncItems(sea):
    # breakpoint()
    # target = idc.get_operand_value(item,0)
    xrefs_from = idautils.XrefsFrom(item)
    xrefs_to = idautils.XrefsTo(item)
    
    for xfrom in xrefs_from:
        if idaapi.get_func_name(xfrom.to):
            xref_set_from.add(idaapi.get_func_name(xfrom.to))
    
    for xto in xrefs_to:
        if idaapi.get_func_name(xto.frm):
            xref_set_to.add(idaapi.get_func_name(xto.frm))


print(xref_set_from)
print(xref_set_to)

print(hex(cur))