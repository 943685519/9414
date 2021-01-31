import sys
from cspProblem import CSP, Constraint
from cspConsistency import Search_with_AC_from_CSP
from searchGeneric import AStarSearcher

class soft_CSP(CSP):
    def __init__(self,domains,constraints,soft_constraints,soft_cost):
        super().__init__(domains,constraints)
        self.soft_constraints = soft_constraints
        self.soft_cost = soft_cost
        
class Search_with_AC_from_Cost_CSP(Search_with_AC_from_CSP):
    def __init__(self,csp):
        super().__init__ (csp)
        self.cost = []
        self.soft_cons = csp.soft_constraints
        self.soft_cost = soft_cost
        
    def heuristic(self,node):
        cost = 0
        cost_list = []
        for task in node:
            if task in self.soft_cons:
                temp = []
                expect_time = self.soft_cons[task]
                for value in node[task]:
                    actual_time = value[1]
                    if actual_time > expect_time:
                        delay = (actual_time//10- expect_time//10)*24 + ((actual_time%10) - (expect_time%10))
                        temp.append(self.soft_cost[task] * delay)
                    else:
                        temp.append(0)
                if len(temp)!=0:
                    cost_list.append(min(temp))
        cost = sum(cost_list)
        return cost
    
        
def before (t1,t2):
    return t1[1]<=t2[0]

def after (t1,t2):
    return t1[0]>=t2[1]
       
def same_day (t1,t2):
    return t1[0]//10 == t2[0]//10        

def starts_at (t1,t2):
    return t1[0] == t2[1]

def start_day(day):
    startday = lambda x: x[0]//10 ==day
    return startday

def start_time(time):
    def starttime(val):
        return val[0] % 10 == time
    return starttime

def starts_before(t_time):
    startbefore= lambda x: x[0] <= t_time
    return startbefore

def starts_after(t_time):
    startbefore= lambda x: x[0] >= t_time
    return startbefore

def ends_before(t_time):
    startbefore= lambda x: x[1] <= t_time
    return startbefore

def ends_after(t_time):
    startbefore= lambda x: x[1] <= t_time
    return startbefore


def clearLine(filename):
    file1 = open(filename, 'r', encoding='utf-8')
    file2 = open('change_input.txt', 'w', encoding='utf-8')
    try:
        for line in file1.readlines():
            if line.startswith('#'):
               line = "\n"
               line = line.strip("\n")
            file2.write(line)
    finally:
        file1.close()
        file2.close()
        
filename = sys.argv[1]
clearLine(filename)
    
with open("change_input.txt",'r') as f:
    task={}
    varies=[]
    soft_cost = {}
    soft_constraint = {}
    task_domain = {}
    hard_domain = []
    dayn = {'mon':1,'tue':2,'wed':3,'thu':4,'fri':5}
    timen = {'9am':1,'10am':2,'11am':3,'12pm':4,'1pm':5,'2pm':6,'3pm':7,'4pm':8,'5pm':9}
    for line in f:
            line = line.strip()
            line = line.replace(',','')
            line = line.split(' ')
  
            if 'task' in line:
                varies.append(line[1])
                task[line[1]]= int(line[2])
                
            domain = set()
            for i in range(11,60):
                if not i%10 ==0: 
                    domain.add(i)
            for t in task:
                time = []
                for i in domain:
                    if i%10 + int(task[t]) <= 9:
                        time.append((i,i+int(task[t])))
                task_domain[t] = sorted(time)
                
            if len(line) ==4 and 'constraint' in line:
                    l1 = line[1]
                    l2 = line[3]
                    if line[2]=='before':
                        hard_domain.append(Constraint((l1,l2),before))
                    elif line[2]=='after':
                        hard_domain.append(Constraint((l1,l2),after))
                    elif line[2]=='same-day':
                        hard_domain.append(Constraint((l1,l2),same_day))
                    elif line[2]=='starts-at':
                        hard_domain.append(Constraint((l1,l2),starts_at))                                        
     
                                                         
            if 'domain' in line:           
                    if len(line) == 3:
                        t = line[1]
                        s = line[2]
                        if s in dayn:
                            dd = dayn[s]
                            hard_domain.append(Constraint((t,),start_day(dd)))
                        if s in timen:  
                            tt = timen[s]
                            hard_domain.append(Constraint((t,),start_time(tt)))
                    elif len(line) == 5:            
                        if line[3] in dayn and line[4] in timen:
                            t =line[1]
                            if line[2] == 'starts-before':
                                t_time = int(dayn[line[3]]) * 10 + int(timen[line[4]])
                                hard_domain.append(Constraint((t,),starts_before))
                            if line[2] == 'starts-after':
                                t_time = int(dayn[line[3]]) * 10 + int(timen[line[4]])
                                hard_domain.append(Constraint((t,),starts_after))
                            if line[2] == 'ends-before':
                                t_time = int(dayn[line[3]]) * 10 + int(timen[line[4]])
                                hard_domain.append(Constraint((t,),ends_before))
                            if line[2] == 'ends-after':
                                t_time = int(dayn[line[3]]) * 10 + int(timen[line[4]])
                                hard_domain.append(Constraint((t,),ends_after))
                    elif len(line) == 6:
                        if line[2] == 'starts-in':
                            t =line[1]
                            ttt = line[4].split('-')
                            ttt1 = timen[ttt[0]]
                            ttt2 = dayn[ttt[1]]
                            t11 = dayn[line[3]]
                            t22 = timen[line[5]]
                            t1 = t11 * 10 + ttt1
                            t2 = ttt2* 10 + t22
                            hard_domain.append(Constraint((t,),lambda x: x[0] <= t2 and t1 <= x[0]))
                        elif line[2] == 'ends-in':
                            t =line[1]
                            ttt = line[4].split('-')
                            ttt1 = timen[ttt[0]]
                            ttt2 = dayn[ttt[1]]
                            t11 = dayn[line[3]]
                            t22 = timen[line[5]]
                            t1 = t11 * 10 + ttt1
                            t2 = ttt2* 10 + t22
                            hard_domain.append(Constraint((t,),lambda x: t1 <= x[1] <= t2))
                        elif line[2] == 'ends-by':
                            tasks = line[1]
                            day = dayn[line[3]]
                            time = timen[line[4]]
                            soft_cost[tasks] = int(line[5])
                            soft_constraint[tasks] = day*10 + time
                            
                    elif len(line) == 4:            
                        if line[3] in timen:
                            t =line[1]
                            t_time = timen[line[3]]
                            if line[2] == 'starts-before':
                                hard_domain.append(Constraint((t,),lambda x: x[0]%10 <= t_time))
                            if line[2] == 'ends-before':
                                hard_domain.append(Constraint((t,),lambda x: x[1]%10 <= t_time))
                            if line[2] == 'starts-after':
                                hard_domain.append(Constraint((t,),lambda x: x[0]%10 >= t_time))
                            if line[2] == 'ends-after':
                                hard_domain.append(Constraint((t,),lambda x: x[1]%10 >= t_time))                  
    line = f.readline()
f.close()                
            



csp = soft_CSP(task_domain,hard_domain,soft_constraint,soft_cost)
problem = Search_with_AC_from_Cost_CSP(csp)
solution = AStarSearcher(problem).search()


if solution:
    solution = solution.end()
    for task in solution:
        solution[task] = sorted(solution[task])
        for i in dayn:
            if dayn[i] ==list(solution[task])[0][0]//10:
                day = i
        for m in timen:
            if timen[m] == list(solution[task])[0][0]%10:
                time = m
        print(f'{task}:{day} {time}')
    print(f'cost:{problem.heuristic(solution)}')
else:
    print('No solution')
