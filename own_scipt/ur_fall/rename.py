import os
import sys

def rename(path):
	file_list = os.listdir(path)
	for file_name in file_list:
		old_name = os.path.join(path, file_name)
		if os.path.isdir(old_name):
			rename(old_name)
		else:
			#new_name = os.path.join(path, file_name.split('.')[0].split('-')[-1])
			new_name = os.path.join(path, str(int(file_name)))
			os.rename(old_name,new_name)
			print(old_name, new_name)


def main(argv):
	for dirs in argv[1:]:
		rename(dirs)


if __name__=='__main__':
	argv = sys.argv
	main(argv)







