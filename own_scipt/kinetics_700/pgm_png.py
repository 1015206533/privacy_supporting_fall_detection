from PIL import Image
import sys
 
def convert(filename):
    im = Image.open(filename)       # open ppm file
    print(im.size)
    im.show()
 
    # newname = filename[:-4] + '.png'# new name for png file
    # im.save(newname)                # save picture to new file
 
if "__main__" == __name__:
    filename= sys.argv[1]
    convert(filename)