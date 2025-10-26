# import config
import subprocess
# import glob
import os
# import config

ida32_path = "D:\\ida_pro\\IDA_Pro_7.7\\idat.exe"
ida64_path = "D:\\ida_pro\\IDA_Pro_7.7\\idat64.exe"

# 用 IDA Pro 对指定目录下的 ELF 文件进行分析，并生成相应的 JSON 文件

# binary_path = './ori_bin_files/gnu-no-inline/cpio-cpio/cpio-2.12_gcc-8.2.0_x86_32_O0_cpio.elf'
binary_dir = '.\\ori_bin_files\\add_bin_0807'
script_path = '.\\info_collect.py'
log_path = '.\\ida_log\\mylog.log'
json_dir = 'call_graph_new_json'
done = os.listdir(json_dir)

for file in os.listdir(binary_dir):
    binary_path = os.path.join(binary_dir, file)
    if not binary_path[-4:] == '.elf':
        os.remove(binary_path)
# subprocess.call([ida_path,'-B',binary_path])
#  delete_list = ['.id0', '.id1', '.id2', '.nam', '.til', '.i64']
for file in os.listdir(binary_dir):
    binary_path = os.path.join(binary_dir, file)
    if binary_path[-4:] == '.elf':
        if file+'.json' in done:
            continue
        print('processing: ',binary_path)
        bite = file.split('_')[3]
        if bite == '32':
            cmd_str = '{} -L{} -c -A -S{} {}'.format(ida32_path, log_path, script_path, binary_path)
        else:
            cmd_str = '{} -L{} -c -A -S{} {}'.format(ida64_path, log_path, script_path, binary_path)
        p = subprocess.Popen(cmd_str, shell=True)
        p.wait()

for file in os.listdir(binary_dir):
    binary_path = os.path.join(binary_dir, file)
    if not binary_path[-4:] == '.elf':
        os.remove(binary_path)