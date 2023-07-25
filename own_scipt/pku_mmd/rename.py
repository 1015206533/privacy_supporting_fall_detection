import os
import sys

def rename(path):
	file_list = os.listdir(path)
	file_min_num = 1000000
	old_file_list = []
	for file_name in file_list:
		old_name = os.path.join(path, file_name)
		if os.path.isdir(old_name):
			rename(old_name)
		else:
			old_file_list.append(file_name)
			file_min_num = min(file_min_num, int(file_name.split('.')[0]))
	for file in old_file_list:
		new_name = os.path.join(path, 'img_{:04}.png'.format(int(file.split('.')[0])-file_min_num+1))
		old_name = os.path.join(path, file)
		os.rename(old_name,new_name)
		#print(old_name, new_name)


def main(argv):
	for dirs in argv[1:]:
		rename(dirs)


if __name__=='__main__':
	argv = sys.argv
	main(argv)







