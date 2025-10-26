import os
import json
import idaapi
import idautils
import idc
import ida_funcs
import ida_loader
import ida_ua
import ida_bytes
import ida_nalt
import time

# 全局变量
OUTPUT_DIR = None


def get_operand_info(ea, n):
    """
    获取指令的操作数信息

    Args:
        ea (int): 指令地址
        n (int): 操作数索引

    Returns:
        list: 操作数信息，格式为 [值, 类型]
    """
    op_type = idc.get_operand_type(ea, n)
    op_value = idc.get_operand_value(ea, n)
    op_text = idc.print_operand(ea, n)

    if op_type == idc.o_void:
        return ["", "mop_z"]
    elif op_type == idc.o_reg:
        return [f"mop_r", "mop_r"]
    elif op_type == idc.o_mem or op_type == idc.o_phrase or op_type == idc.o_displ:
        if op_text.startswith("$"):
            return [op_text, "mop_v"]
        else:
            return ["mop_v", "mop_v"]
    elif op_type == idc.o_imm:
        return [str(op_value), "mop_n"]
    else:
        return [op_text, "mop_a"]


def get_instruction_operands(ea):
    """
    获取指令的所有操作数

    Args:
        ea (int): 指令地址

    Returns:
        list: 操作数列表
    """
    operands = []
    for i in range(3):  # 最多处理3个操作数
        if idc.get_operand_type(ea, i) != idc.o_void:
            operands.append(get_operand_info(ea, i))
        else:
            break
    return operands


def process_function(func_ea):
    """
    处理单个函数，提取所需信息

    Args:
        func_ea (int): 函数地址

    Returns:
        dict: 函数信息，格式符合要求
    """
    func_name = idaapi.get_func_name(func_ea)
    func = idaapi.get_func(func_ea)

    if not func:
        return None

    # 初始化数据结构
    result = {
        "fname": func_name,
        "ori_mc": [],
        "regu_mc": [],
        "mc_vars": [],
        "nice_name": [],
        "callee": [],
        "caller": ["<self>"]
    }

    # 处理指令
    ori_block = []
    regu_block = []
    vars_set = set()
    callee_set = set(["<self>"])

    # 处理函数体
    curr_ea = func.start_ea
    while curr_ea < func.end_ea:
        mnem = idc.print_insn_mnem(curr_ea)

        # 跳过无效指令
        if mnem == "":
            curr_ea = idc.next_head(curr_ea)
            continue

        # 特别处理调用指令
        if "call" in mnem.lower():
            # 提取原始格式
            op0 = get_operand_info(curr_ea, 0)
            op1 = ["", "mop_z"]  # 第二个操作数通常为空

            # 对于函数调用的第三个操作数（函数参数）
            call_target = idc.get_operand_value(curr_ea, 0)
            target_name = idaapi.get_func_name(call_target) if call_target != 0 else ""

            if target_name:
                callee_set.add(target_name)
                # 生成函数参数描述
                op2 = [f"<fast:\"int (__fastcall *main)(int, char **, char **)\" &(${target_name}).8>.0", "mop_f"]
            else:
                # 间接调用或外部调用
                if op0[0].startswith("$"):
                    called_func_name = op0[0].strip('$".')
                    if called_func_name.endswith('"'):
                        called_func_name = called_func_name[:-1]
                    callee_set.add(called_func_name)
                op2 = ["<>.0", "mop_f"]

            ori_instruction = ["call", op0, op1, op2]
            ori_block.append(ori_instruction)

            # 创建规范化格式
            regu_op0 = ["mop_v", "mop_v"]
            regu_op1 = ["", "mop_z"]

            # 为规范化格式创建参数占位符
            if target_name or (op0[1] == "mop_v" and op0[0].startswith("$")):
                # 为已知函数创建更详细的结构
                arg_list = []
                # 添加常见参数
                arg_names = ["rtld_fini", "argc", "ubp_av", "x4_0", "x5_0", "x6_0", "x7_0"]
                for arg_name in arg_names:
                    vars_set.add(arg_name)
                    if arg_name == "argc":
                        arg_list.append(["argv8", "mop_l", 0])
                    elif arg_name == "ubp_av":
                        arg_list.append(["argv0", "mop_l", 0])
                    else:
                        arg_list.append(["mop_a.8", "mop_a"])

                regu_op2 = [[arg_list], "mop_f"]
            else:
                regu_op2 = [["voidf"], "mop_f"]

            regu_instruction = ["call", regu_op0, regu_op1, regu_op2]
            regu_block.append(regu_instruction)

            # 从函数调用中提取变量名
            for i in range(3):
                op_text = idc.print_operand(curr_ea, i)
                if op_text:
                    possible_vars = [v for v in op_text.split() if v.isalnum() and not v.isdigit()]
                    for var in possible_vars:
                        if len(var) > 2 and var not in ["call", "jmp", "mov", "push", "pop"]:
                            vars_set.add(var)

        curr_ea = idc.next_head(curr_ea)

    # 如果有指令，将块添加到结果中
    if ori_block:
        result["ori_mc"].append(ori_block)
        result["regu_mc"].append(regu_block)

    # 添加特殊变量和函数引用
    for var in ["rtld_fini", "argc", "ubp_av", "x1_0", "x2_0", "x3_0", "x4_0", "x5_0", "x6_0", "x7_0"]:
        vars_set.add(var)

    result["mc_vars"] = list(vars_set)

    # 添加预定义的nice_name
    for var in result["mc_vars"]:
        if var in ["argc", "argv", "rtld_fini", "ubp_av"]:
            result["nice_name"].append(var)

    # 添加特殊调用关系
    if func_name == "_start":
        callee_set.update(["main", ".__libc_start_main", "__libc_csu_init", "__libc_csu_fini", ".abort"])

    result["callee"] = list(callee_set)

    return result


def extract_all_functions():
    """
    提取所有函数信息并保存到JSON文件
    """
    global OUTPUT_DIR

    try:
        # 获取当前数据库文件名
        input_file = ida_nalt.get_root_filename()
        base_name = os.path.splitext(input_file)[0]

        print(f"[D] 处理文件: {input_file}")

        # 创建输出文件路径
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        output_path = os.path.join(OUTPUT_DIR, f"{base_name}.json")

        # 处理所有函数
        func_data = {}
        func_count = 0

        for func_ea in idautils.Functions():
            result = process_function(func_ea)
            if result:
                func_name = result["fname"]
                func_data[func_name] = result
                func_count += 1

        # 写入结果到JSON文件
        with open(output_path, 'w') as f:
            json.dump(func_data, f, indent=2)

        print(f"[D] 已处理 {func_count} 个函数")
        print(f"[D] 输出已保存到: {output_path}")

        return True

    except Exception as e:
        print(f"[!] 处理文件时出错: {e}")
        return False


def wait_for_analysis_completion():
    """
    等待IDA分析完成的函数，适用于各种IDA版本
    """
    print("[D] 等待IDA分析完成...")

    # 尝试不同的方法来等待分析完成
    try:
        # 方法1: 使用新版IDA API
        import ida_auto
        ida_auto.auto_wait()
        print("[D] 分析完成 (使用ida_auto.auto_wait)")
        return
    except (ImportError, AttributeError):
        print("[D] ida_auto.auto_wait不可用，尝试其他方法")

    try:
        # 方法2: 使用idaapi.autoWait (老版本)
        if hasattr(idaapi, 'autoWait'):
            idaapi.autoWait()
            print("[D] 分析完成 (使用idaapi.autoWait)")
            return
    except (AttributeError, Exception) as e:
        print(f"[D] idaapi.autoWait不可用: {str(e)}")

    # 方法3: 手动等待和刷新
    print("[D] 使用手动等待...")

    # 等待一段时间，确保IDA有足够时间完成分析
    time.sleep(10)

    # 尝试刷新视图
    try:
        idaapi.refresh_idaview_anyway()
    except Exception:
        pass

    print("[D] 分析完成 (使用手动等待)")


def main():
    """
    主函数 - IDA脚本入口点
    """
    global OUTPUT_DIR

    try:
        print("[D] 脚本开始执行...")

        # 获取脚本参数 - 关键修复点
        # 处理 IDA Pro 命令行传递参数的方式
        # IDA脚本参数通常在 ARGV 中，但是格式可能会有所不同
        script_args = None
        for i in range(len(idc.ARGV)):
            arg = idc.ARGV[i]
            if arg.startswith("-Oextractor:"):
                script_args = arg
                break

        if not script_args:
            print("[!] 错误: 缺少必要参数")
            print("    用法: -Oextractor:<output_dir>")
            idc.qexit(1)
            return

        # 解析参数
        args = script_args.split(":")
        if len(args) < 2:
            print("[!] 错误: 参数格式不正确")
            print("    用法: -Oextractor:<output_dir>")
            idc.qexit(1)
            return

        OUTPUT_DIR = args[1]
        print(f"[D] 输出目录: {OUTPUT_DIR}")

        # 等待IDA分析完成
        wait_for_analysis_completion()

        # 提取函数信息
        if extract_all_functions():
            print("[D] 处理完成")
            idc.qexit(0)
        else:
            print("[!] 处理失败")
            idc.qexit(1)

    except Exception as e:
        print(f"[!] 脚本执行出错: {e}")
        idc.qexit(1)


if __name__ == "__main__":
    main()