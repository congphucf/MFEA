import random
import copy
from chromosome import Chromosome

a = input().split()
route = int(a[0])
vehical = int(a[1])
max_weight = int(a[2])

weight = list(map(float, input().split()))

distance = []
for i in range(route+1):
    row = list(map(float, input().split()))
    distance.append(row)

city = int(input())
cost = []
for i in range(city):
    row = list(map(float, input().split()))
    cost.append(row)



def initialize_population(number_of_population, size):
    population = [Chromosome(size, 2) for x in range(number_of_population)]

    for y in range(number_of_population):
        sorted_indices = [0]+sorted(range(1,min(route+vehical,len(population[y].adn))), key=lambda k: population[y].adn[k])+[0]
        for x in range(1, len(sorted_indices)):
            if sorted_indices[x]>route:
                sorted_indices[x]=0

        w = 0
        score=0
        for x in sorted_indices:
            if(x==0):
                w = 0
            else:
                w += weight[x-1]

            if w > max_weight:
                score = max(score, w-max_weight)
            
        if score!=0:
            population[y].set_factory_cost_with_factor(0, score*500*route)
        
        else:
            d = 0
            for x in range(2, len(sorted_indices)):
                d += distance[sorted_indices[x-1]][sorted_indices[x]]
            population[y].set_factory_cost_with_factor(0, d)
        
        sorted_indices = sorted(range(1,len(population[y].adn)), key=lambda k: population[y].adn[k])
        path = []
        for x in sorted_indices:
            if x>0 and x<=city:
                path.append(int(x))
        path = [1]+path+[1]

        d=0
        for x in range(1, len(path)):
            d+=cost[path[x-1]-1][path[x]-1]
        
        population[y].set_factory_cost_with_factor(1,d)

    indexed_arr = [(value, index) for index, value in enumerate(population)]

    sorted_arr = sorted(indexed_arr, key=lambda k: k[0].factory_cost[0])

    for new_rank, (value, index) in enumerate(sorted_arr):
        population[index].set_factory_rank_with_factor(0, new_rank)

    sorted_arr = sorted(indexed_arr, key=lambda k: k[0].factory_cost[1])

    for new_rank, (value, index) in enumerate(sorted_arr):
        population[index].set_factory_rank_with_factor(1, new_rank)

    for x in range(number_of_population):
        population[x].set_skill_fator(1 if population[x].factory_rank[1] < population[x].factory_rank[0] else 0)
        population[x].set_scalar_fintness()
    
    return population

def fitness(chromosome: Chromosome):
    if (chromosome.skill_factor == 0):
        sorted_indices = [0]+sorted(range(1,min(route+vehical,len(chromosome.adn))), key=lambda k: chromosome.adn[k])+[0]
        for x in range(1, len(sorted_indices)):
            if sorted_indices[x]>route:
                sorted_indices[x]=0

        w = 0
        score=0
        for x in sorted_indices:
            if(x==0):
                w = 0
            else:
                w += weight[x-1]

            if w > max_weight:
                score = max(score, w-max_weight)
            
        if score!=0:
            return score*route*500
        
        d = 0
        for x in range(1, len(sorted_indices)):
            d += distance[sorted_indices[x-1]][sorted_indices[x]]
        return d
    
    else:
        sorted_indices = sorted(range(2,len(chromosome.adn)), key=lambda k: chromosome.adn[k])
        path = []
        for x in sorted_indices:
            if x>0 and x<=city:
                path.append(x)
        path = [1]+path+[1]

        d=0
        for x in range(1, len(path)):
            d+=cost[path[x-1]-1][path[x]-1]
        
        return d
    
def cross_over(chromosome1: Chromosome, chromosome2: Chromosome):

    son1 = Chromosome(len(chromosome1.adn), 2)
    son2 = Chromosome(len(chromosome1.adn), 2)

    adn1 = copy.copy(chromosome1.adn)
    adn2 = copy.copy(chromosome2.adn)

    k = random.randint(1, min(city, route))
    for x in range(k, len(adn1)):
        adn1[x], adn2[x] = adn2[x],adn1[x]
    
    son1.set_adn(adn1)
    son2.set_adn(adn2)

    rmp1 = random.uniform(0,1)
    if rmp1>0.5:
        son1.set_skill_fator(chromosome1.skill_factor)
    else:
        son1.set_skill_fator(chromosome2.skill_factor)
    
    rmp2 = random.uniform(0,1)
    if rmp2>0.5:
        son2.set_skill_fator(chromosome1.skill_factor)
    else:
        son2.set_skill_fator(chromosome2.skill_factor)

    son1.set_factory_cost(fitness(son1))
    son2.set_factory_cost(fitness(son2))
    
    return son1, son2

def mutate(chromosome: Chromosome):

    son = Chromosome(len(chromosome.adn), 2)
    adn = copy.copy(chromosome.adn)
    rmp=random.uniform(0,1)
    x = random.randint(1,10)
    for k in range(x):
        if rmp > 0.5:
            k1 = random.randint(1, len(chromosome.adn)-1)
            k2 = random.randint(1, len(chromosome.adn)-1)

            adn[k1], adn[k2] = adn[k2], adn[k1]
        else:
            k = random.randint(1, len(chromosome.adn)-1)
            adn[k]=1-adn[k]

    son.set_adn(adn)
    son.set_skill_fator(chromosome.skill_factor)    
    son.set_factory_cost(fitness(son))
    return son

def MEAF():
    population =  initialize_population(800, max(route+vehical, city+1))

    population.sort(key= lambda k: (k.skill_factor, k.factory_cost[k.skill_factor]))

    rank = 0
    check = True
    for k in range(len(population)):
        if check and population[k].skill_factor ==1:
            check = False
            rank = 0
        population[k].set_factory_rank(rank)
        population[k].set_scalar_fintness()
        rank+=1

    population = sorted(population, key = lambda k: k.scalar_fintness, reverse=1)[0:800]

    for x in range(500):
        tmp = []
        for y in range(500):
            population = sorted(population, key = lambda k: k.scalar_fintness, reverse=1)[0:800]
            rmp =  random.uniform(0,1)

            if(rmp > 0.1):
                k1 = random.randint(0, len(population)-1)
                k2= random.randint(0, len(population)-1)
                son1, son2 = cross_over(copy.copy(population[k1]), copy.copy(population[k2]))
                tmp.append(son1)
                tmp.append(son2)

            else:
                k = random.randint(0, len(population)-1)
                son = mutate(copy.copy(population[k]))
                tmp.append(son)
        
        population+=tmp
        population.sort(key= lambda k: (k.skill_factor, k.factory_cost[k.skill_factor]))
        rank = 0
        check = True
        for k in range(len(population)):
            if check and population[k].skill_factor ==1:
                check = False
                rank = 0
            population[k].set_factory_rank(rank)
            population[k].set_scalar_fintness()
            rank+=1
    
        population = sorted(population, key = lambda k: k.scalar_fintness, reverse=1)[0:800]
        for chromosome in population[0:2]:
            sorted_indices = [0]+sorted(range(1,len(chromosome.adn)), key=lambda k: chromosome.adn[k])+[0]
            for x in range(1, len(sorted_indices)):
                if sorted_indices[x]>route:
                    sorted_indices[x]=0
            print(chromosome.factory_cost[chromosome.skill_factor])

            if chromosome.skill_factor==1:
                print("TSP: ", chromosome.factory_cost[1])
                sorted_indices = sorted(range(2,len(chromosome.adn)), key=lambda k: chromosome.adn[k])
                path = []
                for x in sorted_indices:
                    if x>0 and x<=city:
                        path.append(x)
                path = [1]+path+[1]
                print(path)

            else:
                print("CVRP: ", chromosome.factory_cost[0] )
                sorted_indices = [0]+sorted(range(1,min(route+vehical,len(chromosome.adn))), key=lambda k: chromosome.adn[k])+[0]
                for x in range(1, len(sorted_indices)):
                    if sorted_indices[x]>route:
                        sorted_indices[x]=0
                print(sorted_indices)
    return population[0], population[1]


MEAF()



