import OSS_SchedulingSolver as SS
import os
import numpy as np
from datetime import datetime

def save_results(p, flowTimes, totalDuration, robustness, i, path):
  np.save(path + str(i) + '_population.npy', p)
  np.save(path + str(i) + '_flowTimes.npy', flowTimes)
  np.save(path + str(i) + '_total_duration.npy', totalDuration)
  np.save(path + str(i) + '_robustness.npy', robustness)

def create_population(p, nrPatients, nrOperations, p_size, cutoff, elitists_ind, norm_prob_dist):
  
  # (5a) Select parents from mating pool
  parents_ind = np.random.choice(elitists_ind, size=(int(p_size/4),2), replace=True, p=norm_prob_dist)
  
  len_parents = nrPatients*nrOperations
  pairs = p[parents_ind]
  cp = cutoff * np.ones(int(p_size/4)) if np.any(cutoff) else np.round(np.random.random(int(p_size/4)), 1)
  
  part1 = [pairs[i,:,:int(cp[i]*len_parents)] for i in range(int(p_size/4))]
  
  indices = list(map(int, cp * len_parents))
  
  # (5b) Create offspring through crossover
  new_parents = []
  
  for i,children in enumerate(part1):
    
    s = indices[i]
    
    children = children.tolist()
    
    if parents_ind[i][0] == parents_ind[i][1]:
      new_parents.append(pairs[i][0])
      new_parents.append(pairs[i][1])
    else:
      while (len(children[0]) < len_parents) | (len(children[1]) < len_parents):
        
        s = int(s)
        
        if np.count_nonzero(np.array(children[0]) == pairs[i][1][s]) < nrOperations:
          children[0] = children[0] + [pairs[i][1][s]]
        
        if np.count_nonzero(np.array(children[1]) == pairs[i][0][s]) < nrOperations:
          children[1] = children[1] + [pairs[i][0][s]]
        
        s = (s+1) % len_parents
      
      new_parents.append(children[0])
      new_parents.append(children[1])
  
  return np.array(new_parents)

def mutation(p_new, max_nr_mutations):
    
  for i, child in enumerate(p_new):
    nr_mutations = int(max_nr_mutations * np.random.random() + 1)
    
    for m in range(nr_mutations):
      rand_ind = np.random.choice(range(1,len(child)), 2, replace=False)
      p_new[i][rand_ind[0]], p_new[i][rand_ind[1]] = p_new[i][rand_ind[1]], p_new[i][rand_ind[0]]

  return p_new

def remove_duplicates(p_size, p_old, p_new=None):
  
  if p_new is None:
    p = p_old
  else:
    p = np.vstack([p_old, p_new])
  
  x = []
  unique_children = [[i, x.append(tuple(r))] for i, r in enumerate(p) if tuple(r) not in x]
  
  while len(unique_children) < len(p):
    
    ind_unique_children = [i[0] for i in unique_children]
    ind_doub_children = [i for i in range(int(len(p))) if i not in ind_unique_children]
    
    for i in ind_doub_children:
      rand_ind = np.random.choice(range(1,len(p[i])), 2, replace=False)
      p[i][rand_ind[0]], p[i][rand_ind[1]] = p[i][rand_ind[1]], p[i][rand_ind[0]]
    
    x = []
    unique_children = [[i, x.append(tuple(r))] for i, r in enumerate(p) if tuple(r) not in x]
  
  if p_new is None:
    return p
  else:
    return p[int(p_size*0.5):]

def is_pareto_efficient(costs, return_mask = True):
  """
  From stackoverflow: 
  https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
  """
  
  is_efficient = np.arange(costs.shape[0])
  n_points = costs.shape[0]
  next_point_index = 0
  
  while next_point_index < len(costs):
  
    nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
    nondominated_point_mask[next_point_index] = True
    
    is_efficient = is_efficient[nondominated_point_mask]
    costs = costs[nondominated_point_mask]
    next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
  
  if return_mask:
  
    is_efficient_mask = np.zeros(n_points, dtype = bool)
    is_efficient_mask[is_efficient] = True
    
    return is_efficient_mask
  else:
    return is_efficient

def score_pareto_index(flowTimes, robustness, p_size):
  
  ft = flowTimes.copy()
  r = robustness.copy()
  
  par_idx = []
  F = []
  i = 0
  
  # Pareto front scoring
  while len(par_idx) < int(p_size*0.5):
    i = i + 1
    
    costs = np.array(list(zip(ft, r)))
    is_efficient = is_pareto_efficient(costs, return_mask = True)
    
    idx = np.where(is_efficient)[0]
    ft[idx] = 999
    r[idx] = 2
    
    par_idx.extend(idx)
    F.append([i, list(idx)])
  
  # Crowded comparison
  if len(par_idx) > int(p_size*0.5):
    F_rest_idx = F[-1][1]
    par_idx = par_idx[:-len(F[-1][1])]
    F = F[:-1]
    
    r_rest = robustness[F_rest_idx]
    ft_rest = flowTimes[F_rest_idx]
    
    order_ft = np.lexsort((r_rest, ft_rest))
    
    r_rest = r_rest[order_ft]
    ft_rest = ft_rest[order_ft]
    
    F_rest_idx = np.array(F_rest_idx)
    F_rest_idx = list(F_rest_idx[order_ft])
    
    dist = []
    
    for i in range(len(ft_rest)-2):
      i = i+1
      r_norm_dist = (r_rest[i-1]-r_rest[i+1])/(np.max(r_rest)-np.min(r_rest))
      ft_norm_dist = (ft_rest[i+1]-ft_rest[i-1])/(np.max(ft_rest)-np.min(ft_rest))
      dist.append(r_norm_dist + ft_norm_dist)
    
    order_dist = np.argsort(-np.array(dist))
    
    idx_dist = []
    idx_dist.extend(order_dist+1)
    idx_dist.extend([0])
    idx_dist.extend([len(ft_rest)-1])
    
    d = int(p_size*0.5) - len(par_idx)
    
    F_rest_idx = np.array(F_rest_idx)[idx_dist]
    F.append([F[-1][0]+1, list(F_rest_idx[:d])])
    par_idx.extend(F_rest_idx[:d])
    
  return F

def test_stopping_criterion(F0, X):
  F0_diff = []
  
  for i in range(len(F0)-1):
    F0_diff.append(len(set(F0[i+1]) - set(F0[i])))
  
  v = np.lib.stride_tricks.sliding_window_view(F0_diff,X)
  delta_F0 = v.sum(axis=-1)
  
  delta_F0_0 = np.where(delta_F0.astype(int)<=0)[0]
  
  if any(delta_F0_0):
    metStoppingCriterion = True
  else:
    metStoppingCriterion = False
  
  return metStoppingCriterion
  
def runAlgo(nrPatients, nrROs, RO_tasks, RTT_tasks, AutoSeg, AutoQA, p_size, max_nr_mutations, X, cp, nbIterations, prevGen, prevGenPath, break_RO, break_RTT, break_start, break_end, end_of_day): 
  
  nrOperations = len(RO_tasks + RTT_tasks + AutoSeg + AutoQA)
  
  # (1) Initialize population of size P
  if np.any(prevGen):
    path = prevGenPath
    p = np.load(path + str(prevGen) + '_population.npy')
    
    i = 1 + prevGen
    nbIterations = nbIterations + prevGen
  else:
    path = r'./run_' + str(nrPatients) + '_pats_' + str(nrROs) + '_ROs_' + datetime.now().strftime('%y%m%d_%H%M') + '/'
    os.mkdir(path)    
    
    seq0 = np.array([np.arange(nrPatients) for i in range(nrOperations)]).ravel()
    p = np.vstack([np.random.choice(seq0[1:], nrPatients*nrOperations-1, replace=False) for i in range(p_size)])
    p = np.concatenate([np.zeros((p_size,1)),p], axis=1) + 1
    p = remove_duplicates(p_size, p)
    
    i = 1
  
  # (2) Calculate Fmean and RoO for each individual sequence
  sequences = SS.input_solver(p, RO_tasks, RTT_tasks, AutoSeg, AutoQA)
  sets = SS.solver_Sets(nrPatients, nrOperations)
  parameters = SS.solver_Parameters(nrOperations)
  
  parameters['break length RO'] = break_RO
  parameters['break lenght RTT'] = break_RTT
  parameters['break start'] = break_start
  parameters['break end'] = break_end
  parameters['end of day'] = end_of_day
  
  flowTimes, totalDuration, robustness = SS.runSolver(sequences, nrROs, RO_tasks, RTT_tasks, AutoSeg, AutoQA, sets, parameters)
  
  flowTimes = np.array(flowTimes)
  totalDuration = np.array(totalDuration)
  robustness = np.array(robustness)
  
  save_results(p, flowTimes, totalDuration, robustness, i, path)
  
  nr_unique_p = len(np.unique(p, axis=0))
  print(i, nr_unique_p, flowTimes[np.where(flowTimes < 999)].mean(), robustness[np.where(flowTimes < 999)].mean())
  
  F0 = []
  metStoppingCriterion = False
  
  while metStoppingCriterion is False:
    
    # (3) Rank sequences by Pareto front index
    F = score_pareto_index(flowTimes, robustness, p_size)
    
    if i == 1:
      F0.append(list(zip(flowTimes[F[0][1]].astype(int), robustness[F[0][1]])))
  
    i = i + 1
    
    # (4) Create mating pool
    elitists_ind = np.hstack([f[1] for f in F])
    par_score = np.hstack([np.ones(len(f[1]))*f[0] for f in F])
    
    probability_dist = (par_score.max()-par_score)/(par_score.max()-par_score.min())
    probability_dist[np.where(probability_dist == 0)] = 1.0/len(F)
    norm_prob_dist = probability_dist/probability_dist.sum()
    
    # (5) Create offspring through crossover
    p_new = create_population(p, nrPatients, nrOperations, p_size, cp, elitists_ind, norm_prob_dist)
    
    # (5c) Apply random mutations
    p_new = mutation(p_new, max_nr_mutations)
    
    # remove duplicates
    p_new = remove_duplicates(p_size, p[elitists_ind], p_new)
	
    # (6) Calculate Fmean and RoO for each new individual sequence
    sequences = SS.input_solver(p_new, RO_tasks, RTT_tasks, AutoSeg, AutoQA)
    new_flowTimes, new_totalDuration, new_robustness = SS.runSolver(sequences, nrROs, RO_tasks, RTT_tasks, AutoSeg, AutoQA, sets, parameters)
    new_flowTimes = np.array(new_flowTimes)
    new_totalDuration = np.array(new_totalDuration)
    new_robustness = np.array(new_robustness)
    
    # (7) Merge mating pool and new population
    p = np.vstack([p[elitists_ind], p_new])
    flowTimes = np.hstack([flowTimes[elitists_ind], new_flowTimes])
    totalDuration = np.hstack([totalDuration[elitists_ind], new_totalDuration])
    robustness = np.hstack([robustness[elitists_ind], new_robustness])
    
    save_results(p, flowTimes, totalDuration, robustness, i, path)
    
    nr_unique_p = len(np.unique(p, axis=0))
    print(i, nr_unique_p, new_flowTimes[np.where(new_flowTimes < 999)].mean(), new_robustness[np.where(new_flowTimes < 999)].mean())
    
    # (8) Rank sequences by Pareto front index
    
    F = score_pareto_index(flowTimes, robustness, p_size)
    F0.append(list(zip(flowTimes[F[0][1]].astype(int), robustness[F[0][1]])))
    
    if i > X:
      metStoppingCriterion = test_stopping_criterion(F0, X)
  
  return