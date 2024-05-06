import random
import numpy as np
import copy

def sphere(x, shift=None):
    if shift is not None:
        x = x - shift
    return np.sum(x**2)

def weierstrass(x, shift=None, a=0.5, b=3, kmax=20):
    if shift is not None:
        x = x - shift
    D = len(x)
    result = 0
    for i in range(D):
        result += np.sum([a**k * np.cos(2*np.pi*b**k*(x[i]+0.5)) for k in range(kmax)])
    return result

def rosenbrock(x, shift=None):
    if shift is not None:
        x = x - shift
    return np.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def schwefel(x, shift=None):
    if shift is not None:
        x = x - shift
    return np.sum(-x * np.sin(np.sqrt(np.abs(x))))

def griewank(x, shift=None):
    if shift is not None:
        x = x - shift
    part1 = np.sum(x**2)/4000
    part2 = np.prod(np.cos(x/np.sqrt(np.arange(1, len(x)+1))))
    return 1 + part1 - part2

def rastrigin(x, shift=None):
    if shift is not None:
        x = x - shift
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)

def ackley(x, shift=None):
    if shift is not None:
        x = x - shift
    n = len(x)
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n)) - np.exp(np.sum(np.cos(2 * np.pi * x) / n)) + 20 + np.exp(1)

class Subpopulation:
    def __init__(self, low, high, label) -> None:
        self.max_pop = 200
        self.limit = [low, high]
        self.generation = [[random.uniform(low, high) for x in range(50)] for y in range(50)]
        self.archive = copy.copy(self.generation)
        self.ur = 0.9
        self.label = label
    
    @property
    def covariance(self):
        return np.cov(np.array(copy.copy(self.archive)).T)

    @property
    def expectation(self):
        return np.mean(np.array(copy.copy(self.archive)), axis=0)

    def add_pop(self, next_generation):
        if len(self.archive)< self.max_pop:
            self.archive += next_generation
        else:
            if random.uniform(0,1)<self.ur:
                for k in range(len(next_generation)):
                    p=random.randint(0,len(self.archive)-1)
                    self.archive.pop(p)
                self.archive+=next_generation

    def defireantial_evolution(self):
        next_generation = copy.deepcopy(self.generation)
        for k in range( len(next_generation)):
            x = random.randint(0, len(self.generation)-1)
            for i in range(len(next_generation[k])):
                rmp = random.uniform(0,1)
                if rmp>0.5:
                    next_generation[k][i] = next_generation[k][i] + 0.5*(next_generation[x][i]-next_generation[k][i])
           
        self.add_pop(next_generation)
        self.generation+=next_generation
        if self.label == 0:
            self.generation.sort(key=lambda x: sphere(np.array(x),np.array([0 for y in range(len(x))])))
        if self.label == 1:
            self.generation.sort(key=lambda x: sphere(np.array(x),np.array([80 for y in range(len(x))])))
        if self.label == 2:
            self.generation.sort(key=lambda x: sphere(np.array(x),np.array([-80 for y in range(len(x))])))
        if self.label == 3:
            self.generation.sort(key = lambda x: weierstrass(np.array(x),np.array([-0.4 for y in range(len(x))])))
        if self.label == 4:
            self.generation.sort(key= lambda x: rosenbrock(np.array(x), np.array([0 for y in range(len(x))])))
        if self.label == 5:
            self.generation.sort(key=lambda x: ackley(np.array(x), np.array([40 for y in range(len(x))])))
        if self.label == 6:
            self.generation.sort(key = lambda x: weierstrass(np.array(x),np.array([-0.4 for y in range(len(x))])))
        if self.label == 7:
            self.generation.sort(key= lambda x: schwefel(np.array(x), np.array([420.9678 for y in range(len(x))]) ))
        if self.label == 8:
            self.generation.sort(key = lambda x: griewank(np.array(x),np.array([-80 for y in range(25)]+[80 for y in range(25)])))
        if self.label == 9:
            self.generation.sort(key = lambda x: rastrigin(np.array(x),np.array( [-40 for y in range(25)]+[40 for y in range(25)] )))

        self.generation = self.generation[:50]
        

    def sort_generation(self):
        if self.label == 0:
            self.generation.sort(key=lambda x: sphere(np.array(x),np.array([0 for y in range(len(x))])))
        if self.label == 1:
            self.generation.sort(key=lambda x: sphere(np.array(x),np.array([80 for y in range(len(x))])))
        if self.label == 2:
            self.generation.sort(key=lambda x: sphere(np.array(x),np.array([-80 for y in range(len(x))])))
        if self.label == 3:
            self.generation.sort(key = lambda x: weierstrass(np.array(x),np.array([-0.4 for y in range(len(x))])))
        if self.label == 4:
            self.generation.sort(key= lambda x: rosenbrock(np.array(x), np.array([0 for y in range(len(x))])))
        if self.label == 5:
            self.generation.sort(key=lambda x: ackley(np.array(x), np.array([40 for y in range(len(x))])))
        if self.label == 6:
            self.generation.sort(key = lambda x: weierstrass(np.array(x),np.array([-0.4 for y in range(len(x))])))
        if self.label == 7:
            self.generation.sort(key= lambda x: schwefel(np.array(x), np.array([420.9678 for y in range(len(x))]) ))
        if self.label == 8:
            self.generation.sort(key = lambda x: griewank(np.array(x),np.array([-80 for y in range(25)]+[80 for y in range(25)])))
        if self.label == 9:
            self.generation.sort(key = lambda x: rastrigin(np.array(x),np.array( [-40 for y in range(25)]+[40 for y in range(25)] )))
    
    def fitness(self):
        if self.label == 0:
            return sphere(np.array(self.generation[0]),np.array([0 for y in range(len(self.generation[0]))]))
        if self.label == 1:
            return sphere(np.array(self.generation[0]),np.array([80 for y in range(len(self.generation[0]))]))
        if self.label == 2:
            return sphere(np.array(self.generation[0]),np.array([-80 for y in range(len(self.generation[0]))]))
        if self.label == 3:
            return weierstrass(np.array(self.generation[0]),np.array([-0.4 for y in range(len(self.generation[0]))]))
        if self.label == 4:
           return rosenbrock(np.array(self.generation[0]), np.array([0 for y in range(len(self.generation[0]))]))
        if self.label == 5:
            return ackley(np.array(self.generation[0]), np.array([40 for y in range(len(self.generation[0]))]))
        if self.label == 6:
            return weierstrass(np.array(self.generation[0]),np.array([-0.4 for y in range(len(self.generation[0]))]))
        if self.label == 7:
            return  schwefel(np.array(self.generation[0]), np.array([420.9678 for y in range(len(self.generation[0]))]) )
        if self.label == 8:
            return griewank(np.array(self.generation[0]),np.array([-80 for y in range(25)]+[80 for y in range(25)]))
        if self.label == 9:
            return rastrigin(np.array(self.generation[0]),np.array( [-40 for y in range(25)]+[40 for y in range(25)] ))
    



class FrameWork:
    def __init__(self) -> None:
        self.mutil_sub = []
        self.mutil_sub.append(Subpopulation(-100, 100, 0))
        self.mutil_sub.append(Subpopulation(-100, 100, 1))
        self.mutil_sub.append(Subpopulation(-100, 100, 2))
        self.mutil_sub.append(Subpopulation(-0.5, 0.5, 3))
        self.mutil_sub.append(Subpopulation(-50, 50, 4))
        self.mutil_sub.append(Subpopulation(-50, 50, 5))
        self.mutil_sub.append(Subpopulation(-0.5, 0.5, 6))
        self.mutil_sub.append(Subpopulation(-500, 500, 7))
        self.mutil_sub.append(Subpopulation(-100, 100, 8))
        self.mutil_sub.append(Subpopulation(-50, 50, 9))
        self.reward = [[1 for x in range(10)] for y in range(10)]
        self.score = [[1 for x in range(10)] for y in range(10)]
        self.ld = 2

    def KLD(self, archive1: Subpopulation, archive2: Subpopulation):
        tmp1 = (archive1.expectation-archive2.expectation).T@np.linalg.inv(archive1.covariance)@(archive1.expectation-archive2.expectation)
        tmp2 = np.log(abs(np.linalg.det(archive1.covariance)/np.linalg.det(archive2.covariance)))
        return abs(np.trace(np.linalg.inv(archive1.covariance) @ archive2.covariance) + tmp1 + tmp2 - 50)

    def sim(self, archive1, archive2):
        return 1/2*(self.KLD(archive1, archive2)+self.KLD(archive2, archive1))
    
    def update_reward(self, i, j):
        self.reward[i][j]*=self.ld
        
    def selection(self, i):
        for j in range(10):
            self.score[i][j] = 0.5*self.score[i][j] + self.reward[i][j]/np.log(self.sim(self.mutil_sub[i], self.mutil_sub[j]))
        
        total = np.sum(self.score[i])
        probabilities = [fit / total for fit in self.score[i]]
        selected_index = random.choices(range(10), probabilities)[0]
        return selected_index
    
    def knowledge_tranfer(self, target):
        assited = self.selection(target)
        tmp = copy.deepcopy(self.mutil_sub[target].generation)
        t = copy.deepcopy(self.mutil_sub[target].generation[0])
        for k in range(len(self.mutil_sub[target].generation)):
            i = random.randint(0, len(self.mutil_sub[assited].generation)-1)
            for x in range(len(tmp[k])):
                rmp = random.uniform(0,1)
                if(rmp>0.5):
                    tmp[k][x] = self.mutil_sub[assited].generation[i][x]

        self.mutil_sub[target].generation += tmp
        self.mutil_sub[target].sort_generation()
        self.mutil_sub[target].generation = self.mutil_sub[target].generation[0:50]
        
        if (t == self.mutil_sub[target].generation[0]):
            self.reward[target][assited]*=self.ld
        else:
             self.reward[target][assited]/=self.ld
        
        # print(len(tmp), len(tmp[0]))
        self.mutil_sub[target].add_pop(tmp)
        # 
    
f = FrameWork()
for k in range(200):
    for i in range(10):
        rmp = random.uniform(0,1)
        if(rmp < 0.3):
            f.mutil_sub[i]. defireantial_evolution()
        else:
            f.knowledge_tranfer(i)

    print(sphere(np.array(f.mutil_sub[0].generation[0])))

        