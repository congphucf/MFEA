import random
import copy

def check_valid(chromosome, route, weights, max_weight):
    sorted_indices = [0]+sorted(range(1,len(chromosome)), key=lambda k: chromosome[k])
    for x in range(1, len(sorted_indices)):
        if sorted_indices[x]>route:
            sorted_indices[x]=0

    weight = 0
    for x in sorted_indices:
        if(x==0):
            weight = 0
        else:
            weight += weights[x-1]

        if weight > max_weight:
            return False

    return True

def caculate(chromosome, route, distance, weight, max_weight):
    sorted_indices = [0]+sorted(range(1,len(chromosome)), key=lambda k: chromosome[k])+[0]
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

def cross_over(chromosome1, chromosome2):

    k = random.randint(1, len(chromosome1))

    for x in range(k, len(chromosome1)):
        chromosome1[x], chromosome2[x] = chromosome2[x], chromosome1[x]

    return chromosome1, chromosome2

def mutate(chromosome):

    rmp=random.uniform(0,1)
     
    if rmp > 0.5:
        k1 = random.randint(1, len(chromosome)-1)
        k2 = random.randint(1, len(chromosome)-1)

        chromosome[k1], chromosome[k2] = chromosome[k2], chromosome[k1]
    else:
        k = random.randint(1, len(chromosome)-1)
        chromosome[k]=1-chromosome[k]

    return chromosome

def initialize_population(size, routes, vehicals, weight, max_weight):

    population = []
    num=0
    while(num<size):
        chromosome = [random.uniform(0,1) for x in range(routes+vehicals)]
        
        # if(check_valid(chromosome, routes, weight, max_weight)):
        population.append(chromosome)
        num+=1

    return population

def fitness(chromosome, route, distance):
    return 1/caculate(chromosome, route, distance)

def genetic_algorithm(rotue, vehical, distance, weight, max_weight):

    population = initialize_population(800, rotue, vehical, weight, max_weight)
    
    for k in range(8000):

        population = sorted(population, key = lambda x: caculate(x, rotue, distance, weight, max_weight))[0:800]
        rmp =  random.uniform(0,1)

        if(rmp > 0.2):
            k1 = random.randint(0, len(population)-1)
            k2 = random.randint(0, len(population)-1)

            son1, son2 = cross_over(copy.copy(population[k1]), copy.copy(population[k2]))
            # if(check_valid(son1,route, weight, max_weight)):
            population.append(son1)

            # if(check_valid(son2, route, weight, max_weight)):
            population.append(son2)

        else:
            k = random.randint(0, len(population)-1)

            son = mutate(copy.copy(population[k]))
            # if(check_valid(son,route, weight, max_weight)):
            population.append(son)
    
    for chromosome in population:
        sorted_indices = [0]+sorted(range(1,len(chromosome)), key=lambda k: chromosome[k])+[0]
        for x in range(1, len(sorted_indices)):
            if sorted_indices[x]>route:
                sorted_indices[x]=0
        print(sorted_indices)
    return caculate(population[0], route, distance, weight, max_weight)

a = input().split()
route = int(a[0])
vehical = int(a[1])
max_weight = int(a[2])

w = list(map(float, input().split()))
print(w)

distance = []
for i in range(route+1):
    row = list(map(float, input().split()))
    distance.append(row)

print(genetic_algorithm(route, vehical, distance, w, max_weight))



