from numpy.random import PCG64
import xml.etree.ElementTree as ET
import numpy as np
import sys, pygame, math
import random

class SimConfig():
    def __init__(self, xmlpath):
        self.filelocation = xmlpath
        self.parse_xml()
        
    def parse_xml(self):
        self.tree = ET.parse(self.filelocation)
        self.xmlroot = self.tree.getroot()
        
        self.dsp_config = {}
        
        self.dsp_config['res'] = [int(self.xmlroot.find('display_config/resolution/x').text), int(self.xmlroot.find('display_config/resolution/y').text)]
        self.dsp_config['framerate'] = int(self.xmlroot.find('display_config/framerate').text)
        self.dsp_config['timescale'] = float(self.xmlroot.find('display_config/timescale').text)
        self.dsp_config['xyscale'] = float(self.xmlroot.find('display_config/xyscale').text)
        
        self.ga_config = {}
        
        self.ga_config['generations'] = int(self.xmlroot.find('ga_config/generations').text)
        self.ga_config['population'] = int(self.xmlroot.find('ga_config/population').text)
        self.ga_config['mutation_rate'] = float(self.xmlroot.find('ga_config/mutation_rate').text)
        self.ga_config['initial_location'] = [float(self.xmlroot.find('ga_config/initial_location/x_location').text), float(self.xmlroot.find('ga_config/initial_location/y_location').text)]
        
        self.universe_config = {}
        planets = []
        
        self.universe_config['duration'] = int(self.xmlroot.find('universe_config/duration').text)
        self.universe_config['desired_location'] = [float(self.xmlroot.find('universe_config/desired_location/x_location').text), float(self.xmlroot.find('universe_config/desired_location/y_location').text)]
        
        for p in self.xmlroot.findall('universe_config/planet'):
            size = float(p.find('size').text)
            gravity = float(p.find('gravity').text)
            x_loc = float(p.find('location/x_location').text)
            y_loc = float(p.find('location/y_location').text)
            planets.append({'size': size, 'gravity': gravity, 'x': x_loc, 'y': y_loc})
        
        self.universe_config['planets'] = planets

class Spacecraft():
    def __init__(self, position_i, chromosome_list, radius):
        self.position = np.array(position_i)
        self.rd = np.array([0.0, 0.0])
        self.genome = chromosome_list
        self.encode_genome()
        self.size = radius
        self.fitness = 0
        self.isLiving = 1
        
    def encode_genome(self):
        for index, chromosome in enumerate(self.genome):
            self.rd[index] = float(str(-1*int(chromosome[0])) 
                + str(int(chromosome[1:4], 2)) + '.' 
                + str(int(chromosome[4:], 2)))
        
class Planet():
    def __init__(self, x, y, mew, radius, scale):
        self.position_km = np.array([x, y])
        self.position_px = np.array([x/scale, y/scale])
        self.mew = mew
        self.radius_km = radius
        self.radius_px = radius/scale
        
class Universe():
    def __init__(self, configuration):
        self.config = configuration
        self.sc = []
        self.p = []
        
    def screen_init(self):
        pygame.init()
        self.screen = pygame.display.set_mode(self.config.dsp_config['res'])
        self.clock = pygame.time.Clock()
        self.colormap = [(255, 0, 0), (255, 255, 255)]
        
    def draw(self, loc):
        self.screen.fill((0, 0, 0))
        
        for i in self.p:
            pygame.draw.circle(self.screen, (0, 0, 255), (i.position_px[0], i.position_px[1]), i.radius_px)
        
        for j in self.sc:
            pygame.draw.circle(self.screen, self.colormap[j.isLiving], (j.position[0]/self.config.dsp_config['xyscale'], j.position[1]/self.config.dsp_config['xyscale']), j.size/self.config.dsp_config['xyscale'])
        
        pygame.draw.circle(self.screen, (255, 0, 0), (self.config.universe_config['desired_location'][0]/self.config.dsp_config['xyscale'], self.config.universe_config['desired_location'][1]/self.config.dsp_config['xyscale']), 2)
        pygame.draw.circle(self.screen, (0, 255, 0), (loc[0]/self.config.dsp_config['xyscale'], loc[1]/self.config.dsp_config['xyscale']), 2)
        
        pygame.display.flip()
        
    def simulate(self):
        total_time = 0
        
        while total_time < self.config.universe_config['duration']:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()
            
            deltat = self.clock.tick(self.config.dsp_config['framerate'])
            total_time += deltat/1000
            
            for sc in self.sc:
                rdd = np.array([0.0, 0.0])
                for planet in self.p:
                    r = planet.position_km-sc.position
                    r_mag = np.hypot(r[0], r[1])*1000
                    rdd += (planet.mew/np.power(r_mag, 3))*(r*1000)
                                
                sc.rd += (rdd*deltat*self.config.dsp_config['timescale'])/1000.0
                sc.position += sc.rd*deltat*self.config.dsp_config['timescale']
                
                if sc.isLiving == 1:
                    #determine off-screen
                    #print(sc.position/self.scale)
                    if sc.position[0]/self.config.dsp_config['xyscale'] < -0.5*self.config.dsp_config['res'][0] or sc.position[0]/self.config.dsp_config['xyscale'] > 1.5*self.config.dsp_config['res'][0]:
                        sc.isLiving = 0
                    
                    if sc.position[1]/self.config.dsp_config['xyscale'] < -0.5*self.config.dsp_config['res'][1] or sc.position[1]/self.config.dsp_config['xyscale'] > 1.5*self.config.dsp_config['res'][1]:
                        sc.isLiving = 0
                    #determine in planet
                    for planet in self.p:
                        r = planet.position_km - sc.position
                        r_mag = np.hypot(r[0], r[1])
                        #print(r_mag, planet.radius, sc.isLiving)
                        if r_mag < planet.radius_km:
                            sc.isLiving = 0
                
                sc_fitness = 1/np.hypot(sc.position[0]-self.config.universe_config['desired_location'][0], sc.position[1]-self.config.universe_config['desired_location'][1])
                if sc_fitness > sc.fitness:
                    sc.fitness = sc_fitness
                
            self.draw(self.config.ga_config['initial_location'])
            
        return self.sc

    def add_spacecraft(self, spacecraft):
        self.sc.append(spacecraft)
        
    def clean_spacecraft(self):
        del self.sc
        self.sc = []
    
    def create_planets(self):
        for p_c in self.config.universe_config['planets']:
            self.p.append(Planet(p_c['x'], p_c['y'], p_c['gravity'], p_c['size'], self.config.dsp_config['xyscale']))

class GeneticAlgorithm():
    def __init__(self, Verse, configuration):
        self.config = configuration
        self.universe = Verse
        self.chromosomes = []
        self.rnd = np.random.default_rng()
        self.bg = PCG64(int(1000000000*self.rnd.random()))
        
    def random_population(self):
        self.chromosomes = []
        for i in range(0, self.config.ga_config['population']):
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
        index_list = []
        for sc in parents:
            if sc.isLiving == 0:
                index_list.append(parents.index(sc))
        index_list.sort()
        index_list.reverse()
        for i in index_list:
            parents.pop(i)
        if len(parents) < 2:
            self.random_population()
        else:
            for i in range(0, self.config.ga_config['population']):
                sample_parents = parents.copy()
                parent1, index = self.select_parent(sample_parents)
                sample_parents.pop(index)
                parent2, index = self.select_parent(sample_parents)
                
                cross_index0 = int(self.rnd.random()*len(parent1.genome[0]))
                cross_index1 = int(self.rnd.random()*len(parent1.genome[1]))
                    
                newx = parent1.genome[0][:cross_index0] + parent2.genome[0][cross_index0:]
                newy = parent1.genome[1][:cross_index1] + parent2.genome[1][cross_index1:]
                
                for j in range(0, len(newx)):
                    if self.rnd.random() < self.config.ga_config['mutation_rate']:
                        if newx[j]:
                            newx = newx[:j] + '0' + newx[j+1:]
                        else:
                            newx = newx[:j] + '1' + newx[j+1:]
                            
                for j in range(0, len(newy)):
                    if self.rnd.random() < self.config.ga_config['mutation_rate']:
                        if newy[j]:
                            newy = newy[:j] + '0' + newy[j+1:]
                        else:
                            newy = newy[:j] + '1' + newy[j+1:]
                
                new_genome.append([newx, newy])
            
            self.chromosomes = new_genome.copy() 
        
    def calculate(self):
        self.random_population()
        
        for gens in range(0, self.config.ga_config['generations']):
            for genes in self.chromosomes:
                self.universe.add_spacecraft(Spacecraft(self.config.ga_config['initial_location'].copy(), genes, 150))
            
            population_c = self.universe.simulate()
            
            self.generate_population(population_c)
            self.universe.clean_spacecraft()

def start():
    config = SimConfig(sys.argv[1])
    
    TheVerse = Universe(config)
    TheVerse.screen_init()
    TheVerse.create_planets()
    
    GA = GeneticAlgorithm(TheVerse, config)
    
    GA.calculate() 
    
start()