import random
import copy

str = input().split()
n=int(str[0])
m=int(str[1])
k=int(str[2])

str = input().split()
a = int(str[0])
b = int(str[1])
c = int(str[2])
d = int(str[3])
e = int(str[4])
f = int(str[5])

s = []
for i in range(n):
    row = list(map(int, input().split()))
    s.append(row)

g = []
for i in range(n):
    row = list(map(int, input().split()))
    g.append(row)

t = list(map(int, input().split()))

def initalize(size, num_of_pop):
    population = [[random.randint(0,k-1) for x in range(size)] for y in range(num_of_pop)]

    return population

def fitness(chronosome):
    thesis = chronosome[0:n]
    teacher = chronosome[n:n+m]

    hd_thesis = [[] for x in range(k)]
    hd_teacher = [[] for x in range(k)]

    for x in range(len(thesis)):
        hd_thesis[thesis[x]].append(x)
    for x in range(len(teacher)):
        hd_teacher[teacher[x]].append(x)

    scorce = 0
    for values in hd_thesis:
        if len(values)<a :
            scorce-=1000000*(a-len(values))
        elif len(values)>b :
            scorce-=1000000*(len(values)-b)

    for values in hd_teacher:
        if len(values)<c :
            scorce-=1000000*(c-len(values))
        elif len(values)>d :
            scorce-=1000000*(len(values)-d)

    for values in hd_thesis:
        tmp = 0
        for x in values:
            for y in values:
                tmp+=s[x][y]
        if tmp<e:
            scorce-=1000000*(e-tmp)

        scorce+=tmp/2
    
    for x in range(k):
        tmp=0
        for i in hd_thesis[x]:
            for j in hd_teacher[x]:
                tmp+=g[i][j]
                if t[i]-1==j:
                    scorce-=10000000
        if tmp<f:
            scorce-=1000000*(f-tmp)

        scorce+=tmp
    return scorce

def cross_over(chronosome1, chronosome2):
    son1 = copy.copy(chronosome1)
    son2 = copy.copy(chronosome2)

    p = random.randint(0,len(son1))
    for x in range(p, len(son1)):
        son1[x], son2[x]=son2[x], son1[x]
    
    return son1, son2

def mutate(chronosome):
    son = copy.copy(chronosome)

    p = random.randint(0,n)
    for i in range(1):
        x = random.randint(0,len(son)-1)
        son[x] = random.randint(0,k-1)
    return son

def genetic_algorithm():
    population = initalize(n+m, 200)
    population.sort(key=lambda x: -fitness(x))
    for x in range(200):
        tmp = []
        for y in range(100):
            rmp = random.uniform(0,1)
            if(rmp>0.2):
                k1 = random.randint(0, len(population)-1)
                k2 = random.randint(0, len(population)-1)
                son1, son2 = cross_over(copy.copy(population[k1]), copy.copy(population[k2]))

                tmp.append(son1)
                tmp.append(son2)
            else:
                k1 = random.randint(0, len(population)-1)
                son = mutate(copy.copy(population[k1]))

                tmp.append(son)
        population += tmp
        population = sorted(population, key=lambda x: -fitness(x))[0:100]
    
    return population[0]

res = genetic_algorithm()
print(n)
thesis = res[0:n]
for i in range(n):
    print(thesis[i]+1, end=" ")
print()
print(m)
teacher = res[n:m+n]
for i in range(m):
    print(teacher[i]+1, end=" ")

