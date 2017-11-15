# -*- coding: utf-8 -*-

from queue import Queue
import cv2
from PIL import Image

def cfs(im,x_fd,y_fd):

    xaxis=[]
    yaxis=[]
    pix=im.load()
    visited =set()
    q=Queue()
    q.put((x_fd, y_fd))
    visited.add((x_fd, y_fd))
    offsets=[(1, 0), (0, 1), (-1, 0), (0, -1)]#四邻域

    while not q.empty():
        x,y=q.get()

        for xoffset,yoffset in offsets:
            x_neighbor,y_neighbor=x+xoffset,y+yoffset

            if (x_neighbor,y_neighbor) in (visited):
                continue  # 已经访问过了

            visited.add((x_neighbor, y_neighbor))

            try:
                if pix[x_neighbor, y_neighbor]==(0,0,0):
                    xaxis.append(x_neighbor)
                    yaxis.append(y_neighbor)
                    q.put((x_neighbor,y_neighbor))

            except IndexError:
                pass   

    xmax=max(xaxis)
    xmin=min(xaxis)
    #ymin,ymax=sort(yaxis)
    return xmax,xmin

def detectFgPix(im,xmax):

    l,s,r,x=im.getbbox()
    pix=im.load()
    for x_fd in range(xmax+1,r):
        for y_fd in range(x):
            if pix[x_fd,y_fd]==(0,0,0):
                return x_fd,y_fd
            
def CFS(im):

    zoneL=[]#各区块长度L列表
    zoneBE=[]#各区块的[起始，终点]列表
    zoneBegins=[]#各区块起点列表

    xmax=0#上一区块结束黑点横坐标,这里是初始化
    for i in range(1):
        try:
            x_fd,y_fd=detectFgPix(im,xmax)
            xmax,xmin=cfs(im,x_fd,y_fd)
            L=xmax-xmin
            zoneL.append(L)
            zoneBE.append([xmin,xmax])
            zoneBegins.append(xmin)

        except TypeError:
            return zoneL,zoneBE,zoneBegins

    return zoneL,zoneBE,zoneBegins

def zonexCutLines(zoneL,zoneBegins):

    Dmax=20  #最大字符长度，人工统计后填入
#    Dmin=0  #最小字符长度,人工统计后填入
    Dmean=14  #平均字符长度，人工统计后填入

    zonexCutLines=[]

    for i in range(len(zoneL)):
        xCutLines=[]     
        if zoneL[i]>Dmax:

            num=round(float(zoneL[i])/float(Dmean))
            num_int=int(num)

            if num_int==1:
                continue

            for j in range(num_int-1):
                xCutLine=zoneBegins[i]+Dmean*(j+1)
                xCutLines.append(xCutLine)
            zonexCutLines.append(xCutLines)

        else:
            continue

    return zonexCutLines

def yVectors_sorted(zoneBE,VerticalProjection):

    yVectors_dict={}
    yVectors_sorted=[]
    for zoneBegin,zoneEnd in zoneBE:
        L=zoneEnd-zoneBegin
        Dmean=10   #基于人工统计的平均字符长度值
        num=round(float(L)/float(Dmean))#区块长度L除以平均字符长度Dmean四舍五入可得本区块字符数量
        num_int=int(num)

        if num_int>1:#当本区块字符数量>1时候，可以认为出现字符粘连，是需要切割的区块

            for i in range(zoneBegin,zoneEnd+1):

                i=str(i)
                yVectors_dict[i]=VerticalProjection[i]#扣取需要切割的区块对应的垂直投影直方图的部分
            #对扣取部分进行重排并放入yVectors_sorted列表中   
            yVectors_sorted.append(sorted(yVectors_dict.iteritems(),key=lambda d:d[1],reverse=False))

    return yVectors_sorted

def get_dropsPoints(zoneL,zonexCutLines,yVectors_sorted):

    Dmax=0
    Dmean=0
    drops=[]
    h=-1

    for j in range(len(zoneL)):
        yVectors_sorted_=[]

        if zoneL[j]>Dmax:

            num=round(float(zoneL[j])/float(Dmean))
            num_int=int(num)

            #容错处理
            if num_int==1:
                continue

            h+=1
            yVectors_sorted_=yVectors_sorted[h]
            xCutLines=zonexCutLines[h]

            #分离
            yVectors_sorted_x=[]
            yVectors_sorted_vector=[]
            for x,vector in yVectors_sorted_:
                yVectors_sorted_x.append(x)
                yVectors_sorted_vector.append(vector)

            for i in range(num_int-1):

                for x in yVectors_sorted_x:

                    x_int=int(x)
                    #d表示由Dmean得出的切割线和垂直投影距离的最小点之间的距离
                    d=abs(xCutLines[i]-x_int)

                    #d和Dmean一样也需要人工设置
                    if d<4:
                        drops.append(x_int)#x是str格式的 
                        break 

        else:

            #print '本区块只有一个字符'
            continue

    return drops

def get_Wi(im,Xi,Yi):
    pix = im.load()
    n1 = pix[Xi-1,Yi-1]
    n2 = pix[Xi,Yi+1]
    n3 = pix[Xi+1,Yi]
    n4 = pix[Xi+1,Yi]
    n5 = pix[Xi-1,Yi]
    
    if n1 == 255:
        n1 = 1
    if n2 == 255:
        n2 = 1
    if n3 == 255:
        n3 = 1
    if n4 == 255:
        n4 = 1
    if n5 == 255:
        n5 = 1
        
    S = 5*n1 + 4*n2 + 3*n3 + 2*n4 + n5
    
    if S==0 or S==15:
        Wi = 4
    else:
        Wi = max(5*n1, 4*n2, 3*n3, 2*n4, n5)
    return Wi

def situations(Xi,Yi,Wi):
    switcher = {
            1:lambda:(Xi-1,Yi),
            2:lambda:(Xi+1,Yi),
            3:lambda:(Xi+1,Yi+1),
            4:lambda:(Xi,Yi+1),
            5:lambda:(Xi-1,Yi+1)
            }
    func = switcher.get(Wi,lambda:switcher[4]())
    return func()

def dropPath(im,drops):
    l,s,r,x = im.getbbox()
    path = []
    zonePath = []
    for drop in drops:
        Xi = drop
        Yi = 0
        limit_left = drop-4
        limit_right = drop+4
        
        while Yi != x-1:
            Wi = get_Wi(im,Xi,Yi)
            Xi,Yi = situations(Xi,Yi,Wi)
            
            if Xi == limit_left or Xi == limit_right:
                Xi,Yi = path[-1]
                
            if Yi>2:
                if path[-2] == (Xi,Yi) or path[-1] == (Xi,Yi):
                    Xi,Yi = situations(Xi,Yi,4)
                    
            path.append((Xi,Yi))
        
        return zonePath
    

def Drops(im):
    
    zoneL,zoneBE,zoneBegins = CFS(im)
    zonexCutLines_ = zonexCutLines(zoneL,zoneBegins)
    print('zonexCutLines_:',zonexCutLines_)
    yVectors_sorted_ = yVectors_sorted(zoneBE,zoneBegins)
    drops = get_dropsPoints(zoneL,zonexCutLines_,yVectors_sorted_)
    return drops

def DropCut(im):
    
    pix = im.load()    
    drops = Drops(im)
    print('drops:',drops)
    zonePath=dropPath(im,drops)
    print('zonePath:',zonePath)
    for path in zonePath:
        for x,y in path:
            pix[x,y]=255
    return im

if __name__ == '__main__':
    path = 'E:/pest1/nacao/12.png'
    im = Image.open(path)
    im = DropCut(im)
    print (im)