import subprocess
import os
import time
from os import walk
from os.path import abspath, dirname, isfile, join, exists

# 设置IDA路径和插件脚本路径
IDA_PATH = os.getenv("IDA_PATH", "D:\\ida\\idat.exe")  # 使用32位IDA
IDA_SCRIPT = join(dirname(abspath(__file__)), 'idb_to_json.py')
LOG_PATH = "ida_function_extractor_log.txt"
input_dir = r"E:\HermesSim-main\dbs\Dataset-RTOS-525\dbs"
output_dir = r"E:\HermesSim-main\dbs\RTOS-525-json"


def main(input_dir, output_dir):
    """
    批量处理.idb文件，提取函数信息并生成JSON文件
    """
    try:
        # 检查IDA路径是否有效
        if not isfile(IDA_PATH):
            print(f"[!] 错误: IDA_PATH:{IDA_PATH} 无效")
            print("请使用 'export IDA_PATH=/path/to/idat' 设置正确的路径")
            return

        # 检查输入目录是否存在
        if not exists(input_dir):
            print(f"[!] 错误: 输入目录 {input_dir} 不存在")
            return

        # 创建输出目录（如果不存在）
        if not exists(output_dir):
            os.makedirs(output_dir)
            print(f"[D] 已创建输出目录: {output_dir}")

        print(f"[D] 输入目录: {input_dir}")
        print(f"[D] 输出目录: {output_dir}")

        # 获取所有.idb文件
        ida_files = []
        for root, _, files in walk(input_dir):
            for file in files:
                if file.endswith('.idb'):
                    ida_files.append(join(root, file))

        if not ida_files:
            print(f"[!] 错误: 在 {input_dir} 中未找到.idb文件")
            return

        print(f"[D] 找到 {len(ida_files)} 个IDA数据库文件需要处理")

        # 处理每个文件
        success_cnt, error_cnt = 0, 0
        start_time = time.time()

        for ida_path in ida_files:
            print(f"\n[D] 处理文件: {ida_path}")

            # 确保脚本路径存在
            if not exists(IDA_SCRIPT):
                print(f"[!] 错误: 脚本文件 {IDA_SCRIPT} 不存在")
                print(f"请将IDA脚本放在: {IDA_SCRIPT}")
                return

            # 使用绝对路径
            script_path = abspath(IDA_SCRIPT)
            output_dir_abs = abspath(output_dir)

            # 修改命令构建方式 - 关键修改点
            cmd = [
                IDA_PATH,
                '-A',
                f'-L{LOG_PATH}',
                # 关键修改：确保脚本路径和参数之间没有空格
                f'-S"{script_path}" -Oextractor:{output_dir_abs}',
                f'"{ida_path}"'
            ]

            # 转为单个命令字符串
            cmd_str = " ".join(cmd)
            print(f"[D] 执行命令: {cmd_str}")

            # 执行命令
            try:
                proc = subprocess.Popen(
                    cmd_str, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                stdout, stderr = proc.communicate(timeout=600)

                stdout_text = stdout.decode('utf-8', errors='ignore') if stdout else ""
                stderr_text = stderr.decode('utf-8', errors='ignore') if stderr else ""

                # 输出处理结果
                if stdout_text:
                    print(f"[D] 标准输出: {stdout_text}")
                if stderr_text:
                    print(f"[D] 标准错误: {stderr_text}")

                if proc.returncode == 0:
                    print(f"[D] {ida_path}: 处理成功")
                    success_cnt += 1
                else:
                    print(f"[!] {ida_path} 处理失败 (返回码={proc.returncode})")
                    error_cnt += 1

            except subprocess.TimeoutExpired:
                print(f"[!] {ida_path} 处理超时")
                error_cnt += 1
                continue

        # 统计和记录
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"\n[D] 总耗时: {elapsed_time:.2f} 秒")
        print(f"[D] 成功处理的文件数: {success_cnt}")
        print(f"[D] 处理失败的文件数: {error_cnt}")

        with open(LOG_PATH, "a+") as f_out:
            f_out.write(f"总耗时: {elapsed_time:.2f} 秒\n")
            f_out.write(f"成功处理: {success_cnt}\n")
            f_out.write(f"处理失败: {error_cnt}\n")

    except Exception as e:
        print(f"[!] 执行过程中发生异常:\n{e}")


if __name__ == '__main__':
    main(input_dir, output_dir)