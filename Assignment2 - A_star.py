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
        print(state)
        path = state[2]
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
                if v[i] == 0:
                    temp = []
                    for num in path:
                        temp.append(num)
                    q.put((est_cost,i,temp))
                    
                    #print(est_cost,i,path)
                elif v[i] > est_cost and est_cost > 0:
                    v[i] = est_cost
                path.pop()
    """
    l = []
    path = []
    traverser = start_point
    path.append(start_point)
    sum_cost = 0
    while(traverser not in goals):
        intermediate_F = []
        for i in range(0,len(heuristic)): 
            if(cost[traverser][i] > 0):
                intermediate_F.append(sum_cost+cost[traverser][i]+heuristic[i])
                #print(sum_cost,cost[traverser][i],heuristic[i],sum_cost+cost[traverser][i]+heuristic[i])
            else:
                intermediate_F.append(99999)
        #print(intermediate_F)
        min_F = min(intermediate_F)
        #print(min_F)
        index_F = intermediate_F.index(min_F)
        #print(index_F)
        sum_cost += cost[traverser][index_F]
        traverser = index_F
        path.append(traverser)
    print(path)
    l.append(path)
    """
    print(l)
    return l

def UCS_Traversal(
    #add your parameters 
):
    l = []

    return l

def DFS_Traversal(
    #add your parameters 
):
    l = []

    return l


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

    t1 = DFS_Traversal(
)
    t2 = UCS_Traversal(
    #send whatever parameters you require 
)
    t3 = A_star_Traversal(
    cost, heuristic, start_point, goals
)

    l.append(t1)
    l.append(t2)
    l.append(t3)
    return l

