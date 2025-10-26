import idaapi
import idautils
import idc
import sys
import os
import ida_hexrays

# ida_hexrays.gen_microcode()



def stdout_to_file(output_file_name, output_dir=None):
	if not output_dir:
		output_dir = os.path.dirname(os.path.realpath(__file__))
	output_file_path = os.path.join(output_dir, output_file_name)
	print (output_file_path)

	# save original stdout descriptor
	orig_stdout = sys.stdout
	# create output file
	f = open(output_file_path, "w")
	print ("original output start")
	# set stdout to output file descriptor
	sys.stdout = f
	return f, orig_stdout



def main():
	idc.auto_wait()  # 待 IDA 分析完程序后执行
	print('123456')
	print('12345678')

if __name__=='__main__':
	f, orig_stdout = stdout_to_file("output.txt")
	main()
	sys.stdout = orig_stdout #recover the output to the console window
	f.close()

	idc.qexit(0)




#
# import idc
# import idaapi
# import idautils
# idaapi.auto_wait()
# count = 0
# for func in idautils.Functions():
#  # Ignore Library Code
# 	flags = idc.get_func_attr(func, FUNCATTR_FLAGS)
# 	if flags & FUNC_LIB:
# 		continue
# 	for instru in idautils.FuncItems(func):
# 		count += 1
# f = open("instru_count.txt", 'w')
# print_me = "Instruction Count is %d" % (count)
# f.write(print_me)
# f.close()
# idc.qexit(0)
