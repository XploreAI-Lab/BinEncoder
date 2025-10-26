import os
import subprocess
import time
from os.path import exists, join, abspath, dirname, splitext

# 设置路径
IDA_PATH = os.getenv("IDA_PATH", "D:\\ida\\idat64.exe")
IDA_SCRIPT = join(dirname(abspath(__file__)), 'idb_to_json.py')
LOG_PATH = "ida_function_extractor_log.txt"
input_dir = r"E:\HermesSim-main\dbs\Dataset-RTOS-525\dbs"
output_dir = r"E:\HermesSim-main\dbs\RTOS-525-json"


def preprocess_databases():
    """预处理：将所有.idb文件转换为.i64格式"""
    print("[D] 开始预处理数据库文件...")

    # 获取所有.idb文件
    idb_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.idb'):
                idb_path = join(root, file)
                i64_path = splitext(idb_path)[0] + '.i64'

                # 检查对应的.i64文件是否存在
                if exists(i64_path):
                    # 如果已存在，则删除旧的.i64文件
                    print(f"[D] 删除已存在的.i64文件: {i64_path}")
                    try:
                        os.remove(i64_path)
                    except Exception as e:
                        print(f"[!] 无法删除文件 {i64_path}: {e}")
                        continue

                idb_files.append(idb_path)

    if not idb_files:
        print("[D] 未找到需要转换的.idb文件")
        return

    print(f"[D] 找到 {len(idb_files)} 个需要转换的.idb文件")

    # 转换每个.idb文件
    for idb_path in idb_files:
        print(f"[D] 转换数据库: {idb_path}")

        # 构建命令 - 仅打开文件，让IDA自动执行转换
        cmd = [
            IDA_PATH,
            '-A',  # 自动模式
            f'-L{LOG_PATH}',  # 日志文件
            f'"{idb_path}"'  # 要处理的文件
        ]

        cmd_str = " ".join(cmd)
        print(f"[D] 执行命令: {cmd_str}")

        try:
            # 执行命令
            proc = subprocess.Popen(
                cmd_str, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            stdout, stderr = proc.communicate(timeout=300)  # 5分钟超时

            if proc.returncode == 0:
                print(f"[D] 成功转换: {idb_path}")
            else:
                print(f"[!] 转换失败: {idb_path}")

        except subprocess.TimeoutExpired:
            print(f"[!] 转换超时: {idb_path}")
            continue

        # 等待一段时间确保文件已完全写入
        time.sleep(2)

    print("[D] 数据库预处理完成")


def process_databases():
    """处理所有.i64文件，提取函数信息"""
    print("[D] 开始处理数据库文件...")

    # 获取所有.i64文件
    i64_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.i64'):
                i64_files.append(join(root, file))

    if not i64_files:
        print("[!] 未找到.i64文件，请先转换数据库")
        return

    print(f"[D] 找到 {len(i64_files)} 个.i64文件需要处理")

    # 创建输出目录
    if not exists(output_dir):
        os.makedirs(output_dir)

    # 处理每个文件
    success_cnt, error_cnt = 0, 0
    start_time = time.time()

    for i64_path in i64_files:
        print(f"[D] 处理文件: {i64_path}")

        # 构建命令
        cmd = [
            IDA_PATH,
            '-A',  # 自动模式
            f'-L{LOG_PATH}',  # 日志文件
            f'-S"{IDA_SCRIPT}"',  # IDA脚本
            f'-Oextractor:{output_dir}',  # 参数
            f'"{i64_path}"'  # 要处理的文件
        ]

        cmd_str = " ".join(cmd)
        print(f"[D] 执行命令: {cmd_str}")

        try:
            # 执行命令
            proc = subprocess.Popen(
                cmd_str, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            stdout, stderr = proc.communicate(timeout=600)  # 10分钟超时

            stdout_text = stdout.decode('utf-8', errors='ignore') if stdout else ""
            stderr_text = stderr.decode('utf-8', errors='ignore') if stderr else ""

            if stdout_text:
                print(f"[D] 标准输出: {stdout_text}")
            if stderr_text:
                print(f"[D] 标准错误: {stderr_text}")

            if proc.returncode == 0:
                print(f"[D] {i64_path}: 处理成功")
                success_cnt += 1
            else:
                print(f"[!] {i64_path} 处理失败 (返回码={proc.returncode})")
                error_cnt += 1

        except subprocess.TimeoutExpired:
            print(f"[!] {i64_path} 处理超时")
            error_cnt += 1
            continue

    # 统计和记录
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"[D] 总耗时: {elapsed_time:.2f} 秒")
    print(f"[D] 成功处理的文件数: {success_cnt}")
    print(f"[D] 处理失败的文件数: {error_cnt}")

    with open(LOG_PATH, "a+") as f_out:
        f_out.write(f"总耗时: {elapsed_time:.2f} 秒\n")
        f_out.write(f"成功处理: {success_cnt}\n")
        f_out.write(f"处理失败: {error_cnt}\n")


def main():
    """主函数"""
    try:
        # 检查IDA路径
        if not exists(IDA_PATH):
            print(f"[!] 错误: IDA路径不存在: {IDA_PATH}")
            return

        # 检查脚本路径
        if not exists(IDA_SCRIPT):
            print(f"[!] 错误: 脚本文件不存在: {IDA_SCRIPT}")
            return

        # 1. 预处理：转换数据库
        preprocess_databases()

        # 2. 处理：提取函数信息
        process_databases()

    except Exception as e:
        print(f"[!] 执行过程中发生异常: {e}")


if __name__ == "__main__":
    main()