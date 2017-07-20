import sys
import typing
#import re
from PIL import Image


HEIGHT = 100
# chars = "   ...',;:clodxkO0KXNWMMMM"
# chars = "  01"

def pic2ascii(filename: str):
    output = ''
    image = Image.open(filename)#打开一个jpg图像文件，注意是当前路径
    size = getsize(image)#将图像的高度和宽度进行换算
    image = image.resize(size)#以换算后的结果来调整图片
    image = image.convert('L')#将图像转换为灰度图像
    pixs = image.load()#将图像的数据载入pixs中

    for y in range(size[1]):#从上到下遍历（高度）
        for x in range(size[0]):#从左到右遍历（宽度）
            # output += chars[-int(pixs[x, y] / 10)]#依据像素不同灰度，比较暗的地方用空格或者点号
            #                         #比较亮的地方用比较大的字符，在标准输出打印出相应的ASCII码。
            if int(pixs[x, y] / 10) >= 13:
                output += '0'
            else:
                output += '1'
        output += '\n'
    print(output)

    a=filename.rfind(".")#提取文件名中.jpg之前的部分，作为.txt的文件名
    #b=re.findall(r"(.+?)\.",a)  使用正则表达式提取
    with open("%s.txt"%filename[:a],'w') as f:
            f.write(output)


def getsize(image: Image.Image) -> typing.Tuple[int, int]:
    '''Calculate the target picture size
    '''
    s_width = image.size[0]#获取此图像的宽度（以像素为单位）。
    s_height = image.size[1]#获取此图像的高度（以像素为单位）。
    # t_height = HEIGHT
    #
    # #将图片的高度与宽度进行适当的
    # t_width = (t_height*s_width)/s_height
    # t_width = int(t_width * 2.3)
    t_width = 32
    t_height = 32
    t_size = (t_width, t_height)
    return t_size


if __name__ == '__main__':
    # if len(sys.argv) < 2: #len(sys.argv)获取命令行参数的个数
    #     print("Useage: pic2ascii.py filename")#若命令行参数个数少于2，则提示正确格式
    #     sys.exit(1)#sys.exit(n) 退出程序引发SystemExit异常, 可以捕获异常执行些清理工作.
    #                #n默认值为0, 表示正常退出. 其他都是非正常退出.
    #                #还可以sys.exit("sorry, goodbye!"); 一般主程序中使用此退出.
    # filename = sys.argv[1]
    filename = 'my_test_data/9_1.png'
    pic2ascii(filename)