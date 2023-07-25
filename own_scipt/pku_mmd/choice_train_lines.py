import sys 

choice_num = [3, 8, 11, 16, 18, 22, 23, 24, 27, 33, 37, 41, 42, 44, 46, 48, 57, 69, 71, 85, 93, 108, 112, 115, 116, 128, 129, 130, 131, 134, 140, 146, 149, 151, 157, 159, 160, 163, 173, 182, 184, 188, 191, 193, 200, 202, 211, 214, 215, 224]


def process(file):
    train_file = []
    test_file = []
    with open(file, 'r') as f:
        i = 0
        for line in f:
            if i in choice_num:
                test_file.append(line.strip())
            else:
                train_file.append(line.strip())
            i += 1
    print('-----------------------train file-----------------------')
    for line in train_file:
       print(line)
    print('--------------------------------------------------------')
    print('--------------------------------------------------------')
    print('-----------------------test file------------------------')
    for line in test_file:
       print(line)


def main(argv):
    for file in argv[1:]:
        process(file)

if __name__=='__main__':
    argv = sys.argv
    main(argv)
