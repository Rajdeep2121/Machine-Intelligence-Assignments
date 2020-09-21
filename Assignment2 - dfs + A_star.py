from queue import PriorityQueue
import copy

def A_star_Traversal(
    cost, heuristic, start_point, goals
):
    l = []
    heu_len = len(heuristic)
    q = PriorityQueue()
    v = []
    for i in range(heu_len):
        v.append(0)
    
    path = [start_point]
    q.put((heuristic[start_point],start_point,path))
    
    while not q.empty():
        state = q.get()
        
        path = state[2]
        if v[state[1]] == 0:
            v[state[1]] = state[0]
        if state[1] in goals:
            l = path
            break
        traverser = state[1]
        for i in range(0,heu_len): 
            if(cost[traverser][i] > 0):
                est_heu_sub = state[0] - heuristic[traverser]
                est_cost = est_heu_sub + cost[traverser][i] + heuristic[i]
                path.append(i)
                temp = []
                for num in path:
                    temp.append(num)
                if v[i] == 0:
                    q.put((est_cost,i,temp))
                    #print(est_cost,i,path)
                elif v[i] >= est_cost and est_cost > 0:
                    q.put((est_cost,i,temp))
                    v[i] = est_cost
                path.pop()
    return l

def UCS_Traversal(cost, start_point, goals):
    l = []

    l.append([0,[start_point]])     
    visited = set()
    visited.add(start_point)
    while(l!=[]):
        l.sort(key=lambda x:x[0])   
        if(l[0][1][-1] in goals):
            return l[0][1]
        children = findChildren(l,cost,visited)
        path = l[0][1]
        min_cost = l[0][0]
        del l[0]
        for i in reversed(children):
            final_cost = min_cost + cost[path[-1]][i]
            final_path = copy.deepcopy(path)
            final_path.append(i)
            l.insert(0,[final_cost, final_path])

    return l

def findChildren(l,cost,visited):
    x = []
    lastNode = l[0][1][-1]
    for i in range(1,len(cost[lastNode])):
        if(cost[lastNode][i]!=0 and cost[lastNode][i]!=-1 and i not in visited):
            x.append(i)
    return x

def DFS_Traversal(cost, start_point, goals):
    stack =[]
    path=[]
    visited= [0 for i in range(0,len(cost[0]))]
    visited[start_point]=1
    stack.append(start_point)
    i= start_point
    while(i not in goals and stack):       
        i= stack.pop()
        visited[i]=1
        path.append(i)
        j= len(cost[i])-1
        while j>0:
            if visited[j]==1:
                pass 
            elif cost[i][j]<=0:
                pass
            else:
                stack.append(j)
            j=j-1
    if i == start_point:
        path.append(start_point)   
    if path[-1] not in goals:
        path = []     
    return path


'''
Function tri_traversal - performs DFS, UCS and A* traversals and returns the path for each of these traversals 

n - Number of nodes in the graph
m - Number of goals ( Can be more than 1)
1<=m<=n
Cost - Cost matrix for the graph of size (n+1)x(n+1)
IMP : The 0th row and 0th column is not considered as the starting index is from 1 and not 0. 
Refer the sample test case to understand this better

Heuristic - Heuristic list for the graph of size 'n+1' 
IMP : Ignore 0th index as nodes start from index value of 1
Refer the sample test case to understand this better

start_point - single start node
goals - list of size 'm' containing 'm' goals to reach from start_point

Return : A list containing a list of all traversals [[],[],[]]
1<=m<=n
cost[n][n] , heuristic[n][n], start_point, goals[m]

NOTE : you are allowed to write other helper functions that you can call in the given fucntion
'''

def tri_traversal(cost, heuristic, start_point, goals):
    l = []
    t1 = DFS_Traversal(cost, start_point, goals)
    t2 = UCS_Traversal(cost, start_point, goals)
    t3 = A_star_Traversal(cost, heuristic, start_point, goals)

    l.append(t1)
    l.append(t2)
    l.append(t3)
    return l
