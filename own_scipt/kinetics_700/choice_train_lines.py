import sys 

choice_num = [14, 16, 21, 27, 30, 35, 38, 42, 57, 59, 63, 66, 80, 83, 86, 89, 90, 99, 115, 128, 134, 138, 144, 146, 156, 170, 182, 191, 194, 195, 204, 228, 232, 233, 234, 238, 247, 248, 250, 262, 263, 288, 302, 304, 307, 312, 313, 330, 334, 350, 364, 369, 376, 380, 381, 390, 400, 402, 404, 409, 410, 414, 421, 423, 424, 434, 465, 472, 473, 489, 494, 500, 524, 525, 532, 533, 546, 551, 555, 556, 566, 580, 581, 588, 591, 598, 609, 612, 620, 626, 643, 646, 647, 650, 654, 662, 669, 673, 683, 700]


def process(file, train_file_name, test_file_name):
    train_file = []
    test_file = []
    print(file, train_file_name, test_file_name)
    train_f = open(train_file_name, 'w')
    test_f = open(test_file_name, 'w')
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
        train_f.write(line+'\n')
        print(line)
    print('--------------------------------------------------------')
    print('--------------------------------------------------------')
    print('-----------------------test file------------------------')
    for line in test_file:
        test_f.write(line+'\n')
        print(line)
    train_f.close()
    test_f.close()


def main(argv):
    process(argv[1], argv[2], argv[3])

if __name__=='__main__':
    argv = sys.argv
    main(argv)
