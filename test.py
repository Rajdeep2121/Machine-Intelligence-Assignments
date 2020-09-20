
def A_star_Traversal(
    #add your parameters 
):
    l = []

    return l

def UCS_Traversal(
    #add your parameters 
):
    l = []

    return l

def DFS_Traversal(cost, heuristic, start_point, goals):
    l = []
    visited = [0 for i in range(len(cost)-1)]
    visited[start_point]= 1
    i= start_point
    j= start_point
    l.append(start_point)
    while(i not in goals):
        j= start_point
        while(cost[i][j]==0 or cost[i][j]==-1):
            j=j+1
        while(visited[j]==1 or cost[i][j]==0 or cost[i][j]==-1):
            j=j+1
        if visited[j]!=1:
            i=j
            visited[j]=1
            l.append(j)
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

    t1 = DFS_Traversal(cost, heuristic, start_point, goals)
    t2 = UCS_Traversal(
    #send whatever parameters you require 
)
    t3 = A_star_Traversal(
    #send whatever parameters you require 
)

    l.append(t1)
    l.append(t2)
    l.append(t3)
    return l