from numpy.random import PCG64
import numpy as np
import sys, pygame, math
import random

class Spacecraft():
    def __init__(self, position_i, chromosome_list, radius):
        self.position = np.array(position_i)
        self.rd = np.array([0.0, 0.0])
        self.genome = chromosome_list
        self.encode_genome()
        self.size = radius
        self.fitness = 0
        
    def encode_genome(self):
        for index, chromosome in enumerate(self.genome):
            self.rd[index] = float(str(-1*int(chromosome[0])) 
                + str(int(chromosome[1:4], 2)) + '.' 
                + str(int(chromosome[4:], 2)))
        
class Planet():
    def __init__(self, x, y, mew, radius):
        self.position = np.array([x, y])
        self.mew = mew
        self.radius = radius
        
class Universe():
    def __init__(self, xy_scale, time_scale, loc_d, dur):
        self.scale = xy_scale #km/pixel
        self.t_scale = time_scale #s/ms
        self.sc = []
        self.p = []
        self.goal = loc_d
        self.sim_duration = dur
        
    def screen_init(self, display_size, frame_rate):
        pygame.init()
        self.screen = pygame.display.set_mode(display_size)
        self.clock = pygame.time.Clock()
        self.frame_rate = frame_rate #Hz
        
    def draw(self, loc):
        self.screen.fill((0, 0, 0))
        
        for i in self.p:
            pygame.draw.circle(self.screen, (0, 0, 255), (i.position[0]/self.scale, i.position[1]/self.scale), i.radius/self.scale)
        
        for j in self.sc:
            pygame.draw.circle(self.screen, (255, 255, 255), (j.position[0]/self.scale, j.position[1]/self.scale), j.size/self.scale)
        
        pygame.draw.circle(self.screen, (255, 0, 0), (self.goal[0]/self.scale, self.goal[1]/self.scale), 2)
        pygame.draw.circle(self.screen, (0, 255, 0), (loc[0]/self.scale, loc[1]/self.scale), 2)
        
        pygame.display.flip()
        
    def simulate(self, loc_init):
        total_time = 0
        
        while total_time < self.sim_duration:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()
            
            #talk about this
            deltat = self.clock.tick(self.frame_rate)
            total_time += deltat/1000
            
            for sc in self.sc:
                rdd = np.array([0.0, 0.0])
                for planet in self.p:
                    r = planet.position-sc.position
                    r_mag = np.hypot(r[0], r[1])*1000
                    rdd += (planet.mew/np.power(r_mag, 3))*(r*1000)
                                
                sc.rd += (rdd*deltat*self.t_scale)/1000.0
                sc.position += sc.rd*deltat*self.t_scale
                
                sc_fitness = 1/np.hypot(sc.position[0]-self.goal[0], sc.position[1]-self.goal[1])
                if sc_fitness > sc.fitness:
                    sc.fitness = sc_fitness
                
            self.draw(loc_init)
            
        return self.sc

    def add_spacecraft(self, spacecraft):
        self.sc.append(spacecraft)
        
    def clean_spacecraft(self):
        del self.sc
        self.sc = []
    
    def add_planet(self, planet):
        self.p.append(planet)

class GeneticAlgorithm():
    def __init__(self, Verse, generations, pop_size, loc_i, rm):
        self.universe = Verse
        self.totalgenerations = generations
        self.population_size = pop_size
        self.chromosomes = []
        self.rnd = np.random.default_rng()
        self.bg = PCG64(int(1000000000*self.rnd.random()))
        self.location_initial = loc_i
        self.mutation_rate = rm
        
    def random_population(self):
        for i in range(0, self.population_size):
            #randoma = self.bg.random_raw(1)
            #randomb = bin(randoma[0])
            xstring = ''
            ystring = ''
            for i in range(0, 32):
                xstring = xstring + str(int(round(self.rnd.random(), 0)))
                ystring = ystring + str(int(round(self.rnd.random(), 0)))

            self.chromosomes.append([xstring, ystring])
        
    def select_parent(self, parent_list):
        total_fitness = 0
        for i in parent_list:
            total_fitness += i.fitness
        
        dart = total_fitness*self.rnd.random()
        
        current_fitness = 0
        for i, parent in enumerate(parent_list):
            current_fitness += parent.fitness
            if current_fitness > dart:
                return_index = i
                break
        
        return parent_list[return_index], return_index
        
    def generate_population(self, parents):
        new_genome = []
        
        for i in range(0, self.population_size):
            sample_parents = parents.copy()
            parent1, index = self.select_parent(sample_parents)
            sample_parents.pop(index)
            parent2, index = self.select_parent(sample_parents)
            
            cross_index0 = int(self.rnd.random()*len(parent1.genome[0]))
            cross_index1 = int(self.rnd.random()*len(parent1.genome[1]))
                
            newx = parent1.genome[0][:cross_index0] + parent2.genome[0][cross_index0:]
            newy = parent1.genome[1][:cross_index1] + parent2.genome[1][cross_index1:]
            
            for j in range(0, len(newx)):
                if self.rnd.random() < self.mutation_rate:
                    if newx[j]:
                        newx = newx[:j] + '0' + newx[j+1:]
                    else:
                        newx = newx[:j] + '1' + newx[j+1:]
                        
            for j in range(0, len(newy)):
                if self.rnd.random() < self.mutation_rate:
                    if newy[j]:
                        newy = newy[:j] + '0' + newy[j+1:]
                    else:
                        newy = newy[:j] + '1' + newy[j+1:]
            
            new_genome.append([newx, newy])
        
        self.chromosomes = new_genome.copy() 
        
    def calculate(self):
        self.random_population()
        
        for gens in range(0, self.totalgenerations):
            for genes in self.chromosomes:
                self.universe.add_spacecraft(Spacecraft(self.location_initial.copy(), genes, 150))
            
            population_c = self.universe.simulate(self.location_initial)
            
            self.generate_population(population_c)
            self.universe.clean_spacecraft()

def start():
    resolution = [800, 600]
    frame_rate = 60 #Hz
    xy_scale = 50.0
    timescale = 1.0
    sim_duration = 15
    initial_location = [600*xy_scale, 300*xy_scale]
    desired_location = [100*xy_scale, 300*xy_scale]
    mutation_rate = 0.03
    
    TheVerse = Universe(xy_scale, timescale, desired_location, sim_duration)
    TheVerse.screen_init(resolution, frame_rate)
    TheVerse.add_planet(Planet(400*xy_scale, 300*xy_scale, 3.986e14, 6371))
    
    GA = GeneticAlgorithm(TheVerse, 5, 100, initial_location, mutation_rate)
    
    GA.calculate() 
    
start()