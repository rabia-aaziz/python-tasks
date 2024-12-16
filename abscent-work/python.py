
# ////////////////topics ... matplotlib, plotting rountine, scattered plot ,bar chart, 
# histogram,  pie chart//////////////  
# list
# x = [1,2,3,4,5]
# y = [6,2,7,8,9] 
# # numpy array
# x1 =([1,2,3,4,5])
# y1 = ([6,7,8,3,9])
# # pandas DataFrame
# data = {
#     'x':[1,2,3,4,5],
#     'y':[1,6,7,8,9]
# }
# df  = pd.DataFrame(data)
# plt.plot(x,y,label='list data',marker='^' ,ls='--'),
# plt.plot(x1,y1,label='numpy data',marker='o', c='#FF0000'),
# plt.plot(df['x'],df['y'],label='DataFrame data',marker='*',linewidth='3'),
# plt.title('listing chart')
# plt.xlabel('x axis')
# plt.ylabel('y axis')
# plt.legend()
# plt.grid('true')
# plt.show()
# import matplotlib.pyplot as plt 
#///////////.......... scattered plot.
# x = ([1,2,3,4,5])
# y = ([6,2,7,8,9]) 

# plt.scatter(x,y,label='list data',marker='^'),
# plt.title('scatter chart')
# plt.xlabel('x axis')
# plt.ylabel('y axis')
# plt.legend()
# plt.grid('true')
# plt.savefig('scattered_png',dpi=300)
# plt.show()

# # /////////////////// bar chart./////////

# var = ([1,2,3,4,5])
# myvar= ([6,2,7,8,9]) 
# plt.bar(var,myvar,label='list data'),
# plt.title('bar chart')
# plt.xlabel('x axis')
# plt.ylabel('y axis')
# plt.legend()
# plt.grid('true')
# plt.show()
# # ///////////////////////.......... histogram.................

# data = ([1,2,2,3,4,4,4,5,5,5,5,6,6,6,6])

# plt.hist(data, bins=6,label='list data'),
# plt.title('histogram chart')
# plt.xlabel('x axis')
# plt.ylabel('y axis')
# plt.legend()
# plt.grid('true')
# plt.show()
# # //////////////////////////// pie chart///////////////////////
# import matplotlib.pyplot as plt
# import pandas as pd 

# data={
#     'department':['HR','IT','markiting','Sales','finance'],
#     'employes':[10,30,45,10,5]
# }
# df=pd.DataFrame(data)
# plt.pie(df['employes'],labels = df['department'],autopct='%1.1f%%', startangle=90,colors=['black','blue','pink','orange','white'])
# plt.title("pie chart")
# plt.xlabel('x axis')
# plt.ylabel('y axis')
# plt.legend()
# plt.grid('true')
# plt.savefig('pie_png', dpi = 400)
# plt.show()

# /////////////..........homework............./////////

# months=['jan','feb','march','april','may','june','july','aug','sep','nov','dec']
# sale_2023=[230,345,456,678,900,222,456,678,567,879,567,457]
# sale_2024=[790,679,790,700,865,780,356,552,643,867,900,543]

# plt.plot(months,sale_2023, lable='sale of 2023',marker='o',ls='--')
# plt.plot(months,sale_2024,lable='sale of 2024',marker='o',ls='--')

# plt.plot(sale_2023, lable='sales')
# plt.title('2 years sale')
# plt.xlabel('months')
# plt.ylabel('sales')
# plt.legend()
# plt.grid('true')
# plt.savefig('line_png',dpi=400)
# plt.show()

# months = ['jan', 'feb', 'march', 'april', 'may', 'june', 'july', 'aug','sep', 'oct' ,'nov','dec']
# sales_2023 = [2300,3400,4560,6780,9000,2220,4560,6780,5670,8790,5670,4570]
# sales_2024 = [7900,6790,7900,7000,8650,7800,3560,5520,6430,8670,9000,5430]
# plt.bar(months,sales_2023,label= 'Sales 2023',color = '#FF0000')
# plt.bar(months,sales_2024,label= 'Sales 2024', color = '#d88860')
# plt.xlabel('Months'), plt.ylabel('Sales')
# plt.title('2 years sale')
# plt.legend()
# plt.savefig('bar_png',dpi=400)
# plt.show()

# //////////............. 25-11-24.................///////////////


# task.. analyze an visualize the impact of missing data.on insight////
# ... Dataset.. a  csv file product slaes data ,containing missing 
# value in sales column for certain month...
# steps.. load the datasets and visualize the original data with a line chart..
# . clean datasets by filling missing values with the mean ..
# highlight cleaned areas on the plot.
# save and compare the plots before and after cleaning...



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# data={
#     'months':['jan','feb','march','april','may','june','july','aug','sep','oct','nov','dec'],
#     'sales':[200,np.nan,590,345,246,np.nan,354,432,123,234,np.nan,132]
# }
# df=pd.DataFrame(data)
# plt.plot(df['months'],df['sales'], label='original data',marker='o',color='blue',)
# plt.title('original dataset(missing values)')
# plt.xlabel('months')
# plt.ylabel('sales')
# plt.grid(alpha=0.5)
# plt.legend()
# plt.show()

# # fill missing values  with the mean values/..
# mean_sales=df['sales'].mean()
# df['sales_cleaned']=df['sales'].fillna(mean_sales)

# print(':Mean_sales',mean_sales)
# print("Dataset after cleaning:/n",df)

# # plot clean data////
# plt.plot(df['months'],df['sales_cleaned'],ls='--', label='cleaned data (filled)',marker='o',color='green',)
# plt.plot(df['months'],df['sales'], label='original data',marker='o',color='blue',)
# plt.title('original dataset(missing values)')
# plt.xlabel('months')
# plt.ylabel('sales_cleaned')
# plt.grid(alpha=0.5)
# plt.legend()
# plt.show()

# # save the plot to file///
# plt.plot(df['months'],df['sales'], label='original data',marker='o',color='blue',)
# plt.title('original dataset(missing values)')
# plt.xlabel('months')
# plt.ylabel('sales')
# plt.grid(alpha=0.5) 
# plt.legend()
# plt.savefig('original_sales_plot_png')

# # ..cleaned data save////
# plt.plot(df['months'],df['sales_cleaned'], label='cleaned data (filled)',marker='o',color='green',)
# plt.title('original dataset(missing values)')
# plt.xlabel('months')
# plt.ylabel('sales_cleaned')
# plt.grid(alpha=0.5)
# plt.legend()
# plt.savefig('cleaned_data_plot_png')



# ///////comparing data/Categories with different charts...........//

# data={
#     'region':['north','south','east','west','north','east','south','north'],
#     'products':['p1','p2','p3','p4','p5','p6','p7','p8'],
#     'sales':[180,150,300,210,190,280,333,210]
# }
# df=pd.DataFrame(data)

# region_sales=df.groupby('region')['sales'].sum()

# plt.pie(region_sales,labels=region_sales.index, autopct='%1.1f%%',startangle=90, colors=['black','blue','white'])
# plt.title("pie chart")
# plt.xlabel('region axis')
# plt.ylabel('sales axis')
# plt.legend()
# # plt.grid('true')
# # plt.savefig('pie_png', dpi = 400)
# plt.show()


# data={
#     'region':['north','south','east','west','north','east','south','north'],
#     'products':['p1','p2','p3','p4','p5','p6','p7','p8'],
#     'sales':[180,150,300,210,190,280,333,210]
# }
# df=pd.DataFrame(data)

# region_sales=df.groupby('region')['sales'].sum()

# plt.bar(region_sales.index,region_sales, color='green')
# plt.title("bar chart")
# plt.xlabel('region axis')
# plt.ylabel('sales axis')
# plt.legend()
# plt.show()


#  ///state space search......../

# 1)precise  2) analyze...
# states = start , goal(intermidiate)
# actions=left, right, up , down/...
     
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
maze=[
  [0,1,0,0],
  [0,0,0,1],
  [0,0,0,1],
  [1,0,0,0]
]
start=(0, 0)
goal=(3, 3)
def plot_maze(maze,path= None):
  maze_array= np.array(maze)
  plt.imshow(maze_array,cmap='Accent')
  plt.xticks(range(len(maze[0])))
  plt.xticks(range(len(maze)))
  plt.text(start[1],start[0],"start",ha="center",va="center",color="red",fontsize=15,fontweight="bold")
  plt.text(goal[1],goal[0],"Goal",ha="center",va="center",color="red",fontsize=15,fontweight="bold")

  if path:
    for (x, y) in path:
      plt.scatter(y,x ,c='blue', s=100,  alpha=0.6)
    plt.title('scatter maze')  
    plt.show()
def get_neighbour(state):
  x,y = state
  #      right, down, left,   up
  moves=[(0,1),(1,0),(0,-1),(-1,0)]
  neighbours=[]
  for dx , dy in moves:
    nx,ny = x + dx , y + dy
    if nx  >= 0  and ny >=0 and nx< len(maze[0]) and maze[nx][ny] == 0 :
      neighbours.append((nx,ny))
      return neighbours
    
# print("neighbours of state",next(start))

def solve_maze(start,goal):
  queue = deque([(start,[start])])
  visited = set()

  while queue:
   current_state,path = queue.popleft()
   if current_state == goal:
     return path
  
   if current_state not in visited:
    visited.add(current_state)
    for neighbour in get_neighbour(current_state):
        queue.append((neighbour , path + [neighbour]))

  return None 
path = solve_maze(start,goal)
if path:
  print("path from start to goal:",path)
  plot_maze(maze,path)
else:
  print("path not found.")

#   #  list of matrix//
# graph={
#    'A':['B','C'],
#    'B':['C','D'],
#    'c':['a','c','E'],
#    'd':['b','c','f'],
#    'e':['c','f']
#    }
# # .......NOT CONNECTED.......
# graph_list ={
#   'G':['A','B','C','D','E','F'],
#   'A':[0,1,1,0,0,0],
#   'B':[1,0,0,1,0,0],
#   'C':[1,0,0,1,1,0],
#   'D':[0,1,1,0,0,1],
#   'E':[0,0,1,0,0,1],
#   'F':[0,0,0,1,0,1]
# }
# graph={
#   "A":['B','C'],
#   "B":['A','D'],
#   "C":['A','D'],
#   "D":['B','C']
# }
# def  bfs(graph,start):
#   visited = set()
#   queue = deque([start])
#   result =[]
#   while queue:
#         node = queue.popleft()
#         if node not in visited:
#           visited.add(node)
#           result.append(node)

#         for neighbour in graph[node]:
#             if neighbour  not in visited:
#               queue.append(neighbour)
#   return result
        
# start_node = 'A'
# print('bfs treversal:',bfs(graph,start_node))

# what is bfs
# Breadth-First Search (BFS) is a graph traversal algorithm that explores nodes level by level, 
# starting from a given source node. It is commonly used to find the shortest path in an unweighted graph or
# to explore all reachable nodes in a graph.

#  how its worked........................///
# Depth-First Search (DFS) is a graph traversal algorithm that explores as far as possible along each 
# branch before backtracking. It is often used to explore a graph
# or tree by going deeper into the graph before visiting its neighbors.

#    ............homework...........

# write a programe to :
# represent a graph using an adjancey list:
# print all neighbours of a given node. 
# 2) examples : input :A , output : neighbours of A :['B','C']

# graph={
#   'A':['B','C'],
#    'B':['C','D'],
#    'c':['a','c','E'],
#    'd':['b','c','f'],
#    'e':['c','f']   
# }
# def find_adjancy (neighbour):
#     if neighbour in graph:
#         print(f'Neighbour of {neighbour}:{graph[neighbour]}')
#     else:
#         print('{neighbour} is not a node in the graph')

# neighbour = input('Enter a number to get a Neighbour :')
# find_adjancy(neighbour)

# # ................... using dfs,............
# graph={
#     'A':['B','C'],
#     'B':['A','D'],
#     'C':['A','D'],
#     'D':['B','C']
# }
# def dfs (graph,start):
#     visited = set()
#     stack  = [start]
#     result =[]
#     while stack:
#         node = stack.pop()
#         if node not in visited:
#           visited.add(node)
#           result.append(node)

#         for neighbour in graph[node]:
#             if neighbour  not in visited:
#               stack.append(neidghbour)
#     return result
        
# start_node = 'A'
# print('dfs treversal:',dfs(graph,start_node))
#  weighted........././////////////
# wgraph={
#    'A':[('B,2'),('F,6')],
#    'B':[('A,2'),('C,4')],
#    'C':[('B,4'),('D,1')],
#    'D':[('C,1'),('E,3')],
#    'E':[('D,3'),('F,1')],
#    'F':[('E,1'),('A,6')]
# }  
# # ///directed......./////////////////
# dgraph={
#    'A':['B'],
#    'B':['C','D'],
#    'C':['A'],
#    'D':[],
#    'E':['C','D']
# }
# # ............bfs...............
# def  bfs(dgraph,start):
#   visited = set()
#   queue = deque([start])
#   result =[]
#   while queue:
#       node = queue.popleft()
#       if node not in visited:
#           visited.add(node)
#           result.append(node)

#       for neighbour in dgraph[node]:
#           if neighbour  not in visited:
#               queue.append(neighbour)
#   return result
        
# start_node = 'A'
# print('bfs treversal:',bfs(dgraph,start_node))
# # ///////////..........dfs...................
# def dfs (dgraph,start):
#     visited = set()
#     stack  = [start]
#     result =[]
#     while stack:
#         node = stack.pop()
#         if node not in visited:
#           visited.add(node)
#           result.append(node)

#         for neighbour in dgraph[node]:
#             if neighbour  not in visited:
#               stack.append(neighbour)
#     return result
        
# start_node = 'B'
# print('dfs treversal:',dfs(dgraph,start_node))


# graph = {
#   'alice' : ['bob','coral'],
#   'bob'   : ['alica','david'],
#   'coral' : ['alice','david'],
#   'david' : ['bob','coral']
# }
# person_1 = 'bob'
# person_2 = 'coral'

# def mutual_friends(graph,person_1,person_2):
#     if person_1 not in graph or person_2 not in graph:
      
#       return f"{person_1} or {person_2} does not exist in graph"
    
#     friend_1 = set(graph[person_1])
#     friend_2 = set(graph[person_2])
    
#     mutual_friends = friend_1.intersection(friend_2)
    
#     return list(mutual_friends)
   
# result = mutual_friends(graph,person_1,person_2)
    
# print(f'mutual friends of {person_1} or {person_2} is {result}')
# # /...................///////////////
# graph = {
#   'alice' : ['bob','coral'],
#   'bob'   : ['alica','david'],
#   'coral' : ['alice','david'],
#   'david' : ['bob','coral']
# }
# person_1 = 'bob'
# person_2 = 'coral'

# def mutual_friends(graph,person_1,person_2):
#     if person_1 not in graph or person_2 not in graph:
      
#       return f"{person_1} or {person_2} does not exist in graph"
    
#     friend_1 = set(graph[person_1])
#     friend_2 = set(graph[person_2])
    
#     mutual_friends = friend_1.union(friend_2)
    
#     return list(mutual_friends)
   
# result = mutual_friends(graph,person_1,person_2)
    
# print(f'mutual friends of {person_1} or {person_2} is {result}')


# # //////////////////////...............
# graph ={
#   'A' : [('B',9),('J',1)],
#   'B' : [('A',9),('C',2)],
#   'C' : [('B',2),('D',3)],
#   'D' : [('C',3),('E',8)],
#   'E' : [('D',8),('F',4)],
#   'F' : [('E',4),('G',10)],
#   'G' : [('F',10),('H',7)],
#   'H' : [('G',7),('I',6)],
#   'I' : [('H',6),('J',5)],
#   'J' : [('I',5),('A',1)],
# }    
# person_1 = 'F'
# person_2 = 'H'

# def mutual_friends(graph,person_1,person_2):
#     if person_1 not in graph or person_2 not in graph:
      
#       return f"{person_1} or {person_2} does not exist in graph"
    
#     friend_1 = set(graph[person_1])
#     friend_2 = set(graph[person_2])
#     mutual_friends = friend_1.difference(friend_2)

#     return list(mutual_friends)   
# person_1=input("enter your first name")
# person_2=input("enter your second name")
# result = mutual_friends(graph,person_1,person_2)   
# print(f'mutual friends of {person_1} or {person_2} is {result}')

# ................
# from collections import deque
# maze = [
#   [1,1,1,1,1],
#   [1,0,0,0,1],
#   [1,0,1,0,1],
#   [1,0,0,0,1],
#   [1,1,1,1,1]
# ]
# def dfs_maze(maze, x ,y, visited= None ,path = None):
#     rows ,colm =len(maze) ,len(maze[0])
#     if visited is None:

#       visited = set()
#     if path is None:
#          path = []
#     if (x,y)== (rows -2, colm -2):
#         path.append((x,y))
#         print("goal reached!")
#         return True
#     visited.add((x,y))
#     path.append((x,y))
#     maze[x][y]=2
#     moves=[(0,1),(1,0),(0,-1),(-1,0)]

#     for dx, dy in moves:
#          nx,ny = x + dx , y + dy
#          if (0 <= nx <rows and 0<= ny <colm and maze [nx][ny]==0 and (nx,ny) not in visited):
#             if dfs_maze (maze,nx,ny,visited,path):
#                 return True
#     path.pop()
#     return False

# def maze_print(maze):
#    for row in maze:
#       print(" ".join(map(str,row)))

# path =[]
# if dfs_maze(maze,1,1,path=path):
#       print('path found')
#       print('path' ,path)
# else:                            
#       print("no path exist")

# print('maze with path(mark as 2 ):')
# maze_print(maze)

# # ............................... bfs method.......................

# from collections import deque
# maze = [
#     [1,1,1,1,1],
#     [1,0,0,0,1],
#     [1,0,1,0,1],
#     [1,0,0,0,1],
#     [1,1,1,1,1],
# ]
# start=(1, 1) 
# goal=(3, 3) 

# def maze_dfs(state): 
#   x,y = state 
#   moves=[(0,1),(1,0),(0,-1),(-1,0)] 
#   path=[] 
#   for dx , dy in moves: 
#     nx,ny = x + dx , y + dy 


#     if 0 <= nx < len(maze) and 0 <= ny < len(maze[0]) and maze[nx][ny] == 0 :
#       path.append((nx,ny)) 
#       maze[nx][ny] = 2
#       return path
    
# def solve_maze(start,goal): 
#   queue = deque([(start, [start])])
#   visited = set() 
#   while queue:
#     current_state, path = queue.pop() 
#     if current_state == goal:
#       return path

#     if current_state not in visited:
#       visited.add(current_state) 
#       for  dfs in maze_dfs(current_state):
#         queue.append((dfs , path + [dfs]))

#   return None

# path = solve_maze(start , goal)

# print(maze)
# if path:
#   print("path of goal: ", path)    
# else:
#   print("No path found!")
