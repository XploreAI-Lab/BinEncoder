"""
summary: generates microcode for selection

description:
  Generates microcode for selection and dumps it to the output window.
"""

import ida_bytes
import ida_range
import ida_kernwin
import ida_hexrays
import idc
import idaapi
import idautils

# idaapi.install_microcode_filter

if ida_hexrays.init_hexrays_plugin():
    ea = ida_kernwin.get_screen_ea()
    func = idaapi.get_func(ea)
    sel, sea, eea = 1, func.start_ea , func.end_ea
    w = ida_kernwin.warning
    if sel:
        F = ida_bytes.get_flags(sea)
        if ida_bytes.is_code(F):
            hf = ida_hexrays.hexrays_failure_t()
            mbr = ida_hexrays.mba_ranges_t()
            mbr.ranges.push_back(ida_range.range_t(sea, eea))
            mba = ida_hexrays.gen_microcode(mbr, hf, None, ida_hexrays.DECOMP_WARNINGS)
            if mba:
                breakpoint()
                # idaapi.FlowChart(idaapi.get_func(func))
                print("Successfully generated microcode for 0x%08x..0x%08x\n" % (sea, eea))
                vp = ida_hexrays.vd_printer_t()
                mba._print(vp)
            else:
                w("0x%08x: %s" % (hf.errea, hf.str))
        else:
            w("The selected range must start with an instruction")
    else:
        w("Please select a range of addresses to analyze")
else:
    print('vds13: Hex-rays is not available.')

