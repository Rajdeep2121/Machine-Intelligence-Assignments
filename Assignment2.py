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
        #print(q.queue)
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
    visited = []
    visited.append(start_point)
    while(l!=[]):
        l.sort(key=lambda y:y[0]) 
        if(l[0][1][-1] in goals):
            dup = 0
            min_cost_dup = l[0][0] 
            for i in range(1,len(l)):
                if(l[i][0]==min_cost_dup):
                    dup+=1
            if(dup>0):
                ll = [i for i in l if i[0]==min_cost_dup]
                ll.sort(key=lambda r: r[1])
                for i in ll:
                    if(i[1][-1] in goals):
                        return i[1]
                return ll[0][1]
            else:
                return l[0][1]
        path = l[0][1]
        min_cost = l[0][0]
        children = findChildren(l,cost,visited)
        del l[0]
        for i in reversed(children):
            final_cost = min_cost + cost[path[-1]][i]
            final_path = copy.deepcopy(path)
            final_path.append(i)
            l.insert(0,[final_cost, final_path])
            visited.append(i)

    return l

def findChildren(l,cost,visited):
    x = []
    lastNode = l[0][1][-1]
    for i in range(1,len(cost[lastNode])):
        local_visited = l[0][1]
        if(cost[lastNode][i]!=0 and cost[lastNode][i]!=-1 and i not in local_visited):
            x.append(i)
    return x

def DFS_Traversal(cost, start_point, goals):
    stack =[]
    path=[]
    visited= [0 for i in range(0,len(cost[0]))]
    stack.append((start_point,[start_point]))
    if start_point in goals:
        return [start_point]
    i= start_point
    while((i not in goals) and len(stack)):  
        #print(stack)     
        i,arr= stack.pop()
        visited[i] = 1
        path = arr
        j= len(cost[i]) - 1

        while j>=0:
            if visited[j] == 0 and cost[i][j] > 0:
                stack.append((j,arr+[j]))
            j=j-1
   
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

