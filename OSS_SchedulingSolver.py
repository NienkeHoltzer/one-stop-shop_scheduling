# Import necessary packages
import gurobipy as gp
import numpy as np
import pandas as pd # for testing purposes

def list_to_operations(seq, RO_tasks, RTT_tasks, AutoSeg, AutoQA):
  
  RO  = []
  RTT = []
  Seg = []
  QA  = []
  tmp = []
  
  for p in seq:
    tmp.append(p)
    cnt = tmp.count(p)
    if cnt in RO_tasks:
      RO.append((int(p),cnt))
    elif cnt in RTT_tasks:
      RTT.append((int(p),cnt))
    elif cnt in AutoSeg:
      Seg.append((int(p),cnt))
    elif cnt in AutoQA:
      QA.append((int(p),cnt))
  
  return RO, RTT, Seg, QA

def precendence_constraint(operationset):
  
  relations = []
  for i, opA in enumerate(operationset):
    for opB in operationset[i+1:]:
      relations.append((opA,opB))
  
  return relations

def input_solver(p, RO_tasks, RTT_tasks, AutoSeg, AutoQA):
  
  sequences = {}
  for i, seq in enumerate(p):
    RO, RTT, Seg, QA = list_to_operations(seq, RO_tasks, RTT_tasks, AutoSeg, AutoQA)
    
    Precedence_for_RO  = precendence_constraint(RO)
    Precedence_for_RTT = precendence_constraint(RTT)
    Precedence_for_AutoSeg = precendence_constraint(Seg)
    Precedence_for_AutoQA  = precendence_constraint(QA)
    
    prec = {'RO1' : Precedence_for_RO,
        'RO2'  : Precedence_for_RO,
        'RTT1' : Precedence_for_RTT,
        'RTT2' : Precedence_for_RTT,
        'AutoSeg' : Precedence_for_AutoSeg,
        'AutoQA' : Precedence_for_AutoQA
        }  
    
    sequences[i] = {'RO' : RO,
            'RTT': RTT,
            'AutoSeg' : Seg,
            'AutoQA' : QA,
            'PrecendenceRelations' : prec
            } 
  
  return sequences

def solver_Sets(nrPatients, nrOperations):
  
  patients    = range(1, nrPatients + 1)        # set of all patients
  executers   = ['RO1', 'RO2', 'RTT1', 'RTT2', 'AutoSeg', 'AutoQA']        # set of executers
  
  set_of_all_combinations = []
  for e in executers:
    for p in patients:
      for t in range(1,nrOperations+1):
        set_of_all_combinations.append((e,p,t))
  
  tasks = range(1,nrOperations+1)             # set of tasks
  task_2RTT  = [2]                  # tasks that need to be done by 2 RTT together
  task_diffRTT = [(7,8),(12,13)]            # tasks that need to be done by 2 different RTTs
  
  return {'patients' : patients,
      'executers' : executers,
      'all combo' : set_of_all_combinations,
      'tasks' : tasks,
      'task 2 RTT' : task_2RTT,
      'task different RTTs' : task_diffRTT
       }

def solver_Parameters(nrOperations):
  
  r = {1  : 31,     # RO op1:  Patient consult
       2  : 26,     # RTT op1: CT scan acquisition
       3  : 4,      # AutoSeg: Automatic segmentation
       4  : 22,     # RO op2:  Target volume delineation
       5  : 29,     # RO op3:  Peer consult + patient chart administration
       6  : 22,     # RTT op2: Treatment planning preparation
       7  : 15,     # RTT op3: Treatment planning
       8  : 5,      # RTT op4: Treatment planning check
       9  : 5,      # RO op4:  Approve treatment planning
       10 : 15,     # RTT op5: Post-processing treatment planning
       11 : 1,      # AutoQA:  Automatic Physics Quality Assurance
       12 : 10,     # RTT op6: Manual finish of import in R&V
       13 : 5       # RTT op7: Check R&V
       }
  
  
  M      = 10**8      # large number
  t_first  = 1        # first task
  t_final  = nrOperations       # last task
  break_RO   = 15     # break length for ROs
  break_RTT  = 45     # break length for RTTs 
  break_start   = 270          # start of break time window
  break_end     = 330          # end of break time window
  end_of_day = 540    # end of day in minutes
  
  return {'processingtimes' : r,
      'M' : M,
      'first task' : t_first,
      'last task' : t_final,
      'break length RO' : break_RO,
      'break length RTT' : break_RTT,
      'break start' : break_start,
      'break end' : break_end,
      'end of day' : end_of_day
       }
     
def Model(nrROs, RO_tasks, RTT_tasks, AutoSeg, AutoQA, sets, parameters, prec, cnt):

  r = parameters['processingtimes']
  M = parameters['M']
  ROs = [ i for i in sets['executers'] if i.startswith('RO')][:nrROs]
  ROsAll = [ i for i in sets['executers'] if i.startswith('RO')]
  RO_combos = sorted(set([i[0:2] for i in sets['all combo'] if i[0] in ROs]))
  RTTs = [ i for i in sets['executers'] if i.startswith('RTT')]
  RTT_combos = [i for i in sets['all combo'] if i[0] in RTTs]
  RTT_auto_combos = [i for i in sets['all combo'] if i[0] not in ROsAll]
  all_combos = [i for i in sets['all combo'] if i[0] in ROs] + RTT_auto_combos
  t_first = parameters['first task']
  t_final = parameters['last task']
  break_RO = parameters['break length RO']
  break_RTT = parameters['break length RTT']
  end_of_day = parameters['end of day']
  break_start = parameters['break start']
  break_end = parameters['break end']

  # # Model
  m = gp.Model("OneStopShop")

  # # mute console output
  m.setParam('OutputFlag', 0)

  # # Variables
  # time that executor e starts working on operation t on patient p
  s = m.addVars(all_combos , lb=0, name="s")

  # decision variable: 1 if RO e performs task t on patient p, zero otherwise
  u = m.addVars(RO_combos, vtype=gp.GRB.BINARY, name="u")

  # decision variable: 1 if RTT e performs task t on patient p, zero otherwise
  a = m.addVars(RTT_auto_combos, vtype=gp.GRB.BINARY, name="a")

  # break (binary)
  b = m.addVars(all_combos ,vtype=gp.GRB.BINARY, name='b')

  # starttime of break
  bt = m.addVars(ROs + RTTs, name='bt')

  # flow time of patient p
  F = m.addVars(sets['patients'] , lb=sum(r.values()), ub=end_of_day, name="F")

  #Constraints
  #constraints 2:
  m.addConstrs( s[e,p1,t1] + r[t1] - M*(1 - u[e,p1]) <= s[e,p2,t2] + M*(1 - u[e,p2]) for e in ROs for (p1,t1),(p2,t2) in prec[e] )

  #constraints 3:
  m.addConstrs( s[e,p1,t1] + r[t1] - M*(1 - a[e,p1,t1]) <= s[e,p2,t2] + M*(1 - a[e,p2,t2]) for e in RTTs + ['AutoSeg', 'AutoQA'] for (p1,t1),(p2,t2) in prec[e] )

  # constraints 4: 
  m.addConstrs( s[e1,p,t-1] + r[t-1] - M*(1 - a[e1,p,t-1]) <= s[e2,p,t] + M*(1 - a[e2,p,t]) for t in sets['tasks'][1:] for p in sets['patients'] for e1 in RTTs + ['AutoSeg', 'AutoQA'] for e2 in RTTs + ['AutoSeg', 'AutoQA'] )

  # constraints 5:
  m.addConstrs( s[e1,p,t-1] + r[t-1] - M*(1 - a[e1,p,t-1]) <= s[e2,p,t] + M*(1 - u[e2,p]) for t in sets['tasks'][1:] for p in sets['patients'] for e1 in RTTs + ['AutoSeg', 'AutoQA'] for e2 in ROs )

  # constraints 6:
  m.addConstrs( s[e1,p,t-1] + r[t-1] - M*(1 - u[e1,p]) <= s[e2,p,t] + M*(1 - a[e2,p,t]) for t in sets['tasks'][1:] for p in sets['patients'] for e1 in ROs for e2 in RTTs + ['AutoSeg', 'AutoQA'] )

  # constraints 7:
  m.addConstrs( s[e1,p,t-1] + r[t-1] - M*(1 - u[e1,p]) <= s[e2,p,t] + M*(1 - u[e2,p]) for t in sets['tasks'][1:] for p in sets['patients'] for e1 in ROs for e2 in ROs )

  # constraints 8:
  m.addConstrs( s['RTT1',p,t] == s['RTT2',p,t] for p in sets['patients'] for t in sets['task 2 RTT'] )

  # constraints 9:
  m.addConstrs( F[p] >= s[e1,p,t_final] + r[t_final] - s[e2,p,t_first] for p in sets['patients'] for e1 in RTTs for e2 in ROs )
  
  # constraints 10:
  m.addConstrs( sum( a[e,p,t] for e in RTTs ) == 1 for p in sets['patients'] for t in [x for x in RTT_tasks if x != sets['task 2 RTT'][0]] )
  
  # constraints 11:
  m.addConstrs( a['AutoSeg',p,t]  == 1 for p in sets['patients'] for t in AutoSeg )
  
  # constraints 12:
  m.addConstrs( a['AutoQA',p,t]  == 1 for p in sets['patients'] for t in AutoQA )
  
  # constraints 13:
  m.addConstrs( sum( u[e,p] for e in ROs ) == 1 for p in sets['patients'] )
  
  # constraints 14:
  m.addConstrs( s[e,p,t] + r[t] <= end_of_day for e in ROs + RTTs + ['AutoSeg', 'AutoQA'] for p in sets['patients'] for t in sets['tasks'] )  
  
  # constraints 15:
  m.addConstrs( a[e,p,t] == 0 for e in ['AutoSeg', 'AutoQA'] for p in sets['patients'] for t in RTT_tasks )

  # constraints 16:
  m.addConstrs( a[e,p,t] == 0 for e in RTTs + ['AutoQA'] for p in sets['patients'] for t in AutoSeg )
  
  # constraints 17:
  m.addConstrs( a[e,p,t] == 0 for e in RTTs + ['AutoSeg'] for p in sets['patients'] for t in AutoQA )
  
  # constraints 18:
  m.addConstrs( sum( a[e,p,t] for e in RTTs ) == 2 for p in sets['patients'] for t in sets['task 2 RTT'] )

  # constraints 19:
  m.addConstrs( a[e,p,t1] == 1 - a[e,p,t2] for e in RTTs for p in sets['patients'] for t1,t2 in sets['task different RTTs'] )
  
  # constraints 20:
  m.addConstrs( b[e,p1,t1] >= b[e,p2,t2] for e in ROs + RTTs for (p1,t1),(p2,t2) in prec[e] )

  # constraints 21:
  m.addConstrs( bt[e] >= break_start for e in ROs + RTTs )
  
  # constraints 22:
  m.addConstrs( bt[e] + break_RO <= break_end for e in ROs )
  
  # constraints 23:
  m.addConstrs( bt[e] + break_RTT <= break_end for e in RTTs )

  # constraints 24:
  m.addConstrs( bt[e] >= s[e,p,t] + r[t] - M*(1 - b[e,p,t]) - M*(1 - a[e,p,t]) for e in RTTs for p in sets['patients'] for t in RTT_tasks )
  
  # constraints 25:
  m.addConstrs( bt[e] >= s[e,p,t] + r[t] - M*(1 - b[e,p,t]) - M*(1 - u[e,p]) for e in ROs for p in sets['patients'] for t in RO_tasks )
  
  # constraints 26:
  m.addConstrs( bt[e] + break_RTT <= s[e,p,t] + M*b[e,p,t] + M*(1 - a[e,p,t]) for e in RTTs for p in sets['patients'] for t in RTT_tasks )
  
  # constraints 27:
  m.addConstrs( bt[e] + break_RO  <= s[e,p,t] + M*b[e,p,t] + M*(1 - u[e,p]) for e in ROs for p in sets['patients'] for t in RO_tasks )
  
  # constraint 28:
  m.addConstr( s[ROs[0],1,1] == 0 )
  
  # Objective function
  m.setObjective(F.sum()/max(sets['patients']), gp.GRB.MINIMIZE)

  # # Optimization
  m.optimize()
  
  if (m.status != gp.GRB.OPTIMAL):
    averageFt = 999
    totalTime = 999
    robustness = 2
  else:
    nr_t_final = [ t for e in RTTs for p in sets['patients'] for t in RTT_tasks if round(a[e,p,t].X) == 1 and t == t_final ]
    if len(nr_t_final) != len(sets['patients']):
      averageFt = 999
      totalTime = 999
      robustness = 2
      print('Check sequence ' + str(cnt))
    else:
      averageFt = m.getObjective().getValue()
      totalTime = max([round(max(s[e,p,t_final].X for p in sets['patients'])) + r[t_final] for e in ['RTT1', 'RTT2']])
      
      df_s_r = pd.DataFrame([[e, p, t, round(s[e,p,t].X), r[t]] for e,p,t in all_combos], columns=["e", "p", "t", "s", "r"]).drop_duplicates()
      df_a = pd.DataFrame([[e, p, t, round(a[e,p,t].X)] for e in RTTs + ['AutoSeg', 'AutoQA'] for p in sets['patients'] for t in sets['tasks']], columns=["e", "p", "t", "a"]).drop_duplicates()
      df_u = pd.DataFrame([[e, p, round(u[e,p].X)] for e in ROs for p in sets['patients'] for t in sets['tasks']], columns=["e", "p", "u"]).drop_duplicates()
      df_b_bt = pd.DataFrame([[e, p, t, round(b[e,p,t].X), round(bt[e].X)] for e in ROs + RTTs for  p in sets['patients'] for t in sets['tasks']], columns=["e", "p", "t", "b", "bt"]).drop_duplicates()
      
      df = pd.merge(df_s_r, df_a, on=['e', 'p', 't'], how='outer').sort_values(by=['e', 'p', 't']).reset_index(drop=True)
      df = pd.merge(df, df_u, on=['e', 'p'], how='outer').sort_values(by=['e', 'p', 't']).reset_index(drop=True)
      df = pd.merge(df, df_b_bt, on=['e', 'p', 't'], how='outer').sort_values(by=['e', 'p', 't']).reset_index(drop=True)
      
      u_tmp = df[(df['u']==1) & (~df['t'].isin(RO_tasks))].index
      df.loc[u_tmp, 'u'] = 0
      
      df = df[(df['a'] == 1) | (df['u'] == 1)].reset_index(drop=True)
      
      robustness = calcRobustness(df, break_RO, break_RTT, end_of_day)

  return averageFt, totalTime, robustness

def calcRobustness(df, break_RO, break_RTT, end_of_day):
  setPatients = sorted(df['p'].unique())
  nrPatients = len(setPatients)
  nrOperations = len(df['t'].unique())
  lastTask = max(df['t'].unique())
  all_executers = df['e'].unique()
  
  # get start times
  start = df.groupby(['p'])['s'].min().values
  
  # get seq
  df = df.sort_values(by='s')
  
  t_list = []
  for e in all_executers:
    t_list.append(list(zip(df[df['e']==e]['p'],df[df['e']==e]['t'])))
  
  seq = dict(zip(all_executers, t_list)) 
  
  set_of_processingtimes = np.load('ProcessingTimes.npy')
  set_of_processingtimes = set_of_processingtimes[:nrPatients,:,:]
  
  df = df.sort_values(['p','t']).reset_index(drop=True)
  b_idx = df[df['b']==1].groupby('p')['s'].idxmax().values
  b_list = df.iloc[b_idx][['p','t']].values
  b_list = list(zip(df.iloc[b_idx]['e'],b_list))
  
  for pt in b_list:
    break_len = break_RO if np.char.startswith(pt[0], 'RO') else break_RTT
	
    if pt[1][1] is not lastTask:
      set_of_processingtimes[pt[1][0]-1,:,pt[1][1]-1] = set_of_processingtimes[pt[1][0]-1,:,pt[1][1]-1] + float(break_len)
      
  # find precedence relations
  Precedences_for_patients = []
  for p in range(1,nrPatients+1):
      for j in range(1,nrOperations):
          precedence_relation = ( (p, j) , (p, j+1) )
          Precedences_for_patients.append(precedence_relation)
  
  Precedences_for_machines = []
  for name in seq.keys():
      orders = seq[name]
      for j in range(len(orders)-1):
          precedence_relation = ( (orders[j][0], orders[j][1]), (orders[j+1][0], orders[j+1][1]))
          Precedences_for_machines.append(precedence_relation)
  
  prec = Precedences_for_patients + Precedences_for_machines
  
  set_p_t = []
  for p in range(1,nrPatients+1):
      for t in range(1,nrOperations+1):
          set_p_t.append((p,t))
  
  t_first = 1
  t_final = nrOperations
  
  end_time = []
  nr_scenarios = 250
  
  for i in range(nr_scenarios):
    # set the model
    m = gp.Model("OneStopShop-Robustness")
    
    # mute console output
    m.setParam('OutputFlag', 0)  
    
    # select a sample
    r = set_of_processingtimes[:,i,:] 
    
    # time of operation t on patient p
    s = m.addVars(set_p_t, lb=0, name="s")
    
    # flow time of patient p
    F = m.addVars(range(1,nrPatients+1), lb=sum(r[0]), name="F") # ub changed from 560 to end_time feb 8th
    
    #constraint 1:
    m.addConstrs( s[p1,t1] + r[p1-1,t1-1] <= s[p2,t2] for (p1,t1),(p2,t2) in prec )
    
    # constraint 2:
    m.addConstrs( F[p] >= s[p,t_final] + r[p-1,t_final-1] - s[p,t_first] for p in setPatients )
    
    # constraint 3:
    m.addConstrs( s[p,1] >= start[p-1] for p in setPatients )
    
    # Objective function
    m.setObjective(F.sum()/nrPatients, gp.GRB.MINIMIZE)
    
    m.optimize()
      
    if m.status != gp.GRB.OPTIMAL:
      end_time.append(999)
    else:
      startTimes = []
      for p in setPatients:
      
        tmp_s = np.array([i.X for i in s.values()])
        tmp_i = np.where( np.array([i for i in s])[:,0] == p)
        startTimes.append(tmp_s[tmp_i][0])
      
      flowTimes = np.array([i.X for i in F.values()])
      end_time.append(np.max(startTimes + flowTimes))
  
  robustness = float(np.count_nonzero(np.array(end_time) > end_of_day)) / nr_scenarios
  
  return robustness

def runSolver(sequences, nrROs, RO_tasks, RTT_tasks, AutoSeg, AutoQA, sets, parameters):
  
  flowTimes = []
  totalDuration = []
  robustness = []
  
  for i in range(len(sequences)):
    prec = sequences[i]['PrecendenceRelations']
    averageFt, totalTime, tmpRobustness = Model(nrROs, RO_tasks, RTT_tasks, AutoSeg, AutoQA, sets, parameters, prec, i)
    flowTimes.append(averageFt)
    totalDuration.append(totalTime)
    robustness.append(tmpRobustness)

  return flowTimes, totalDuration, robustness