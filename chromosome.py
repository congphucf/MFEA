import random
import copy

class Chromosome:
    def __init__(self, size, skill_fator) -> None:

        self.adn = [random.uniform(0,1) for x in range(size)]
        self.factory_rank = [0 for x in range(skill_fator)]
        self.skill_factor :int  = 1
        self.scalar_fintness = 0
        self.factory_cost = [0 for x in range(skill_fator)]
    
    def set_adn(self, adn):
        self.adn = adn

    def set_skill_fator(self, skill_factor):
        self.skill_factor = skill_factor

    def set_factory_cost(self, f):
        self.factory_cost[self.skill_factor] = f
    
    def set_factory_rank(self, r):
        self.factory_rank[self.skill_factor] = r
    
    def set_factory_cost_with_factor(self, factor, f):
        self.factory_cost[factor] = f

    def set_factory_rank_with_factor(self, factor, r):
        self.factory_rank[factor] = r

    def set_scalar_fintness(self):
        self.scalar_fintness = 1/(self.factory_rank[self.skill_factor]+1)