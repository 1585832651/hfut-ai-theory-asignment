def GJ(this,k):#估价函数计算 h(n) = M + C - K * B
    return this[0] + this[1] - k * this[2]

def creat(array,M,C,B,N):#判断生成节点是否符合规则、判断是否重复
    P = array[:]
    if M == N :#左岸传教士数量等于总数
        if  C >=0  and  C <= N :
            P.insert(0,[M,C,1-B])
            for i in open:
                if P[0] == i[0]:
                    return False
            for i in closed:
                if P[0] == i[0]:
                    return False
            open.append(P)
            return True
        else:
            return False
    elif M > 0 :#左岸传教士数量基于0到N之间时
        if  C >=0 and M >= C and M <= N and C <= N and N-M >= N-C:
            P.insert(0,[M,C,1-B])
            for i in open:
                if P[0] == i[0]:
                    return False
            for i in closed:
                if P[0] == i[0]:
                    return False
            open.append(P)
            return True
        else:
            return False
    elif M == 0:#左岸传教士为0
        if  C >= 0 and C <= N:
            P.insert(0, [M, C, 1 - B])
            for i in open:
                if P[0] == i[0]:
                    return False
            for i in closed:
                if P[0] == i[0]:
                    return False
            open.append(P)
            return True
        else:
            return False
    else:
        return False

if  __name__ == '__main__':
    N = int(input("传教士和野人的人数（默认相同）："))
    K =int(input("船的最大容量："))
    open = []  #创建open表
    closed = [] #创建closed表
    sample = [N,N,1] #初始状态
    goal = [0,0,0]#目标状态
    open.append([sample])
    creatpoint = searchpoint = 0
    while(1):
        if sample == goal:
            print("初始状态为目标状态！")
            break
        if len(open) == 0:
            print("未搜索到解！")
            break
        else:
            this = open.pop(0)
            closed.append(this)
            if this[0] == goal:
                print("搜索成功！")
                print('共生成节点数：{}，共搜索节点数:{}'.format(creatpoint,searchpoint + 1))
                print('过河方案如下：')
                print('      [M, C, B]')
                for i in this[::-1]:
                    print('---->',i)
                exit()
            #扩展节点
            searchpoint += 1
            if this[0][2] == 1 :#船在左岸时
                for i in range(1,K+1):#只
                    if creat(this,this[0][0]-i,this[0][1],this[0][2],N):
                        creatpoint += 1
                for i in range(1,K+1):
                    if creat(this,this[0][0],this[0][1]-i,this[0][2],N):
                        creatpoint += 1
                for i in range(1,K):
                    for r in range(1,K-i+1):
                        if creat(this,this[0][0] - i,this[0][1] - r, this[0][2],N):
                            creatpoint += 1
            else:#船在右岸时
                for i in range(1,K+1):
                    if creat(this,this[0][0]+i,this[0][1],this[0][2],N):
                        creatpoint += 1
                for i in range(1,K+1):
                    if creat(this,this[0][0],this[0][1]+ i,this[0][2],N):
                        creatpoint += 1
                for i in range(1,K):
                    for r in range(1,K-i+1):
                        if creat(this,this[0][0] + i,this[0][1] + r, this[0][2],N):
                            creatpoint += 1

            #计算估计函数h(n) = M + C - K * B 重排open表
            for x in range(0,len(open)-1):
                m = x
                for y in range(x+1,len(open)):
                    if  GJ(open[x][0],K) >  GJ(open[y][0],K):
                        m = y
                if m != x:
                    open[x],open[m] = open[m],open[x]

