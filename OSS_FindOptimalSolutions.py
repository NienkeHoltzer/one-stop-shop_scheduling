import OSS_GeneticAlgorithm as GA

# init a random population
nrPatients = 4
nrROs = 1
RO_tasks  = [1,4,5,9]
RTT_tasks = [2,6,7,8,10,12,13]
AutoSeg = [3] 
AutoQA = [11]
pop_size = 200 # population size, even number
max_nr_mutations = 5 # changed from 5 at 18 okt 2021
X = 50 # stopping criterion threshold 
cp = [] # cutoff ratio: if empty than random crossing point per pair
nbIterations = 750
prevGen = [] # []
prevGenPath = r'./run_5_pats_2_ROs_220120_1040/' # not used if prevGen is empty
break_RO = 15 # break length RO
break_RTT = 45 # break length RTT
break_start = 270 # start of break time window
break_end = 330 # end of break time window
end_of_day = 540 # length of working shift

GA.runAlgo(nrPatients,
           nrROs,
           RO_tasks,
           RTT_tasks,
           AutoSeg,
           AutoQA,
           pop_size,
           max_nr_mutations,
           X,
           cp,
           nbIterations,
           prevGen,
           prevGenPath,
           break_RO,
           break_RTT,
           break_start,
           break_end,
           end_of_day)