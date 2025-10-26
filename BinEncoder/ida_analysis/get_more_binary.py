import os
import shutil

# 筛选目录下二进制文件，并将其复制到指定目录

gnu_dir = 'E:\资料\gnu_debug_noinline'
store_dir = 'ori_bin_files\\add_bin_0807'
check_dir1 = 'ori_bin_files\\ultimate-pretrain-binaries'
check_dir2 = 'ori_bin_files\\all_binaries'

check1 = os.listdir(check_dir1)
check2 = os.listdir(check_dir2)

for proj in os.listdir(gnu_dir):
    projdir = os.path.join(gnu_dir, proj)
    for file in os.listdir(projdir):
        if 'clang-7.0' in file or 'gcc-8.2.0' in file:
            if not 'mipseb' in file:
                if not file in check1 and not file in check2:
                    src_path = os.path.join(projdir,file)
                    dst_path = os.path.join(store_dir,file)
                    shutil.copy(src_path,dst_path)
