import sys
from flask import Flask, render_template,request,make_response
import pandas as pd
import numpy as np
import json  #json request
from werkzeug.utils import secure_filename
import os
import csv #reading csv
import geocoder
from random import randint
import os
from random import randint
from datetime import date
from sklearn import linear_model
global res
from fitness import fitness, BITCOUNT,select_and_crossover
import argparse
import random


sections=["A","B"]


semesters=["sem1","sem2"]


TRACE_ENABLED = True
"""Whether the debug trace is on."""

URANDOM_STRING_LENGTH = 32
"""Size of the entropy string provided from urandom."""


def trace(*text):
  """Outputs a single chunk of text to stderr."""
  if not TRACE_ENABLED:
    return
  data = " ".join([str(chunk) for chunk in text])
  sys.stderr.write(data + "\n")
  sys.stderr.flush()


def decode(member):
  """Displays int as a binary string."""
  return bin(member)[2:].zfill(BITCOUNT)


def mutate(member, num_bits=None):
  if num_bits is None:
    num_bits = random.randint(1, BITCOUNT)
  elif num_bits < 0:
    raise Exception("Number of bits must be non-negative.")
  for pos in random.sample(range(0, BITCOUNT), num_bits):
    member ^= 1 << pos
  return member

def predictor():
    params = Params(**args_dict)
    ga = GeneticAlgo(params)
    ga.initialize()

    for i in ga.evolve():
        print ("Generation", i + 1)
        ga.display()
        print ("")

    
def argmax(values):
  """Returns the index of the largest value in a list."""
  return max(enumerate(values), key=lambda x: x[1])[0]


class Defaults:
  """Default values for params."""
  CROSSING_PROBABILITY = 0.9
  MUTATION_PROBABILITY = 0.01
  ITERATION_COUNT = 30
  INITIAL_POPULATION = 20
  RANDOM_SEED = None
  SELECTION_STRATEGY = "fitness-proportional"
  ELITISM = False


class Params(object):
  """Parameters of a genetic algorithm."""
  def __init__(self, crossing, mutation, iterations, population, random_seed,
               selection_strategy, elitism):
    self.crossing = crossing
    self.mutation = mutation
    self.iterations = iterations
    self.population = population
    self.random_seed = random_seed
    self.selection_strategy = selection_strategy
    self.elitism = elitism


class SelectionStrategy(object):
  """Abstract class for a selection strategy."""
  def select_and_crossover(self, population, crossing):
    raise Exception("Method not implemented.")


class FitnessProportionalSelection(SelectionStrategy):

  def _select(self, population):
    selection = []
    fits = [fitness(member) for member in population]
    sum_fits = sum(fits)
    normalized_fits = [(member, fitness(member) / sum_fits)
                       for member in population]
    normalized_fits = list(sorted(normalized_fits, key=lambda x: x[1],
                                  reverse=True))
    accumulated = 0
    accumulated_fits = []
    for x in normalized_fits:
      accumulated += x[1]
      accumulated_fits.append((x[0], accumulated))

    used = set()
    # TODO(kburnik): This can be optimized.
    while len(selection) < len(population):
      value = random.random()
      for i, x in enumerate(accumulated_fits):
        value -= x[1]
        if value <= 0:
          if i in used:
            continue
          used.add(i)
          selection.append(x[0])
          break
    return list(sorted(selection, key=lambda m: fitness(m), reverse=True))

  def _crossover(self, a, b):
    """Crosses over two members by cutting at radnom point pos (right to left).
    """
    pos = random.randint(1, BITCOUNT - 1)
    mask = (2 ** pos) - 1
    invmask = (2 ** BITCOUNT) - mask - 1
    na = (a & invmask) | (b & mask)
    nb = (b & invmask) | (a & mask)
    return (na, nb)

  def select_and_crossover(self, population, crossing):
    # 1) Select.
    selection = self._select(population)
    # 2) Crossover.
    for i in range(0, len(selection) - 1, 2):
      if random.random() < crossing:
        selection[i], selection[i + 1] = \
            self._crossover(selection[i], selection[i + 1])
    return selection


class UniformSelection(SelectionStrategy):
  """A Uniform selection strategy, all members of the population are equally
     likely to be chosen for crossover. The crossover is done by keeping same
     bits of parents and randomly choosing a the bit that differs."""

  def _crossover(self, a, b):
    # Create mask of different bits from both parents.
    diffmask = a ^ b
    # Copy the same bits to child.
    child = a & b
    parents = [a, b]
    val = 1
    while diffmask > 0:
      # Check if the bit is set.
      if diffmask % 2 == 1:
        # Choose parent randomly. 50/50 chance.
        parent = parents[random.randint(0, 1)]
        # Copy the set bit from the chosen parent to the child.
        child |= (parent & val)
      diffmask /= 2
      val *= 2
    return child

  def select_and_crosover(self, population, crossing):
    selection = set()
    used_pairs = set()
    while len(selection) < len(population):
      pair = tuple(random.sample(population, 2))
      if pair in used_pairs:
        continue
      used_pairs.add(pair)
      if random.random() < crossing:
        child = self._crossover(*pair)
        selection.add(child)
      else:
        selection.add(pair[0])
        selection.add(pair[1])
    selection = list(selection)
    while len(selection) > len(population):
      selection.pop()
    return selection


class GeneticAlgo(object):
  """Represents a genetic algorithm"""
  def __init__(self, params):
    self.params = params
    self.population = []
    self.elite_member = None
    if params.selection_strategy == "fitness-proportional":
      self.selection_strategy = FitnessProportionalSelection()
    elif params.selection_strategy == "uniform":
      self.selection_strategy = UniformSelection()
    else:
      raise Exception("Invalid strategy %s" % params.selection_strategy)

  def initialize(self):
    """Initializes the population."""
    seed = self.params.random_seed
    if seed is None:
      seed = hash(os.urandom(URANDOM_STRING_LENGTH))
    random.seed(seed)
    trace("Using random seed", seed)
    for i in range(self.params.population):
      member = random.randint(0, (2 ** BITCOUNT) - 1)
      self.population.append(member)

  def evolve(self):
    """Generator where each iteration is an evolution step."""
    for generation in range(self.params.iterations):
      # 1) Select and 2) Crossover.
      selection = self.selection_strategy.select_and_crossover(
          self.population, self.params.crossing)

      # Find the elite member if elitism is enabled.
      elite_index = -1
      self.elite_member = None
      if self.params.elitism:
        elite_index = argmax([fitness(member) for member in selection])
        self.elite_member = selection[elite_index]

      # 3) Mutation.
      for i in range(len(selection)):
        # If elitism is enabled, don't mutate no matter the odds.
        if i != elite_index and random.random() < self.params.mutation:
          selection[i] = mutate(selection[i])

      self.population = list(sorted(selection,
                                    key=lambda x: fitness(x),
                                    reverse=True))
      yield generation

  def display(self):
    """Prints the current population"""
    print ("%10s %4s %7s %6s" % ("bin", "int", "fitness", "elite"))
    for member in self.population:
      if self.elite_member == member:
        elite = "ELITE"
      elif self.params.elitism:
        elite = "-"
      else:
        elite = "n/a"
    fits = [fitness(member) for member in self.population]
    minval = min(fits)
    maxval = max(fits)
    avgval = sum(fits) / len(fits)
    print ("min = %.2f, max = %2.f, avg = %.2f" % (minval, maxval, avgval))


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('login.html')

@app.route('/index')
def indexnew():    
    return render_template('home.html')

@app.route('/home')
def homenew():    
    return render_template('home.html', semesters=semesters,sections=sections)


@app.route('/generate', methods = ['POST'])
def generate():
  
  
  sem=request.form["sem"]
  sec=request.form["sec"]
  course=request.form["course"]
  print(sem)
  subfile = request.files['subfile']
  filename = secure_filename(subfile.filename)
  subfile.save(os.path.join("D:\\Upload\\", filename))
  sublist=[]
  file1 =open('D:\\Upload\\'+filename, 'r') 
  Lines = file1.readlines()
  for line in Lines:
    sublist.append(line)

  
  print(sublist)
  subnames=[]
  for i in range(len(sublist)):
    val=sublist[i]
    a1=val.split('-')
    print('-------------Test---------')
    print(a1)
    subnames.append(a1[1])
    
  print(subnames)
  facfile = request.files['facfile']
  filename = secure_filename(facfile.filename)
  facfile.save(os.path.join("D:\\Upload\\", filename))
  faclist=[]
  file1 =open('D:\\Upload\\'+filename, 'r') 
  Lines = file1.readlines()
  for line in Lines:
    faclist.append(line)
  
  sffile = request.files['sffile']
  filename = secure_filename(sffile.filename)
  sffile.save(os.path.join("D:\\Upload\\", filename))
  
  maplist=[]
  file1 =open('D:\\Upload\\'+filename, 'r') 
  Lines = file1.readlines()
  for line in Lines:
    maplist.append(line)
    
  asecfac=[]
  bsecfac=[]
  csecfac=[]
  for i in range(len(maplist)):
    print(i)
    val=maplist[i]
    a1=val.split('-')
    a2=a1[1].split(',')
    demo=[]
    demo.append(a1[0])
    demo.append(subnames[i])
    demo.append(a2[0])
    asecfac.append(demo)
    demo=[]
    demo.append(a1[0])
    demo.append(subnames[i])
    try:
      demo.append(a2[1])
    except:
      demo.append(a2[0])
    bsecfac.append(demo)
    demo=[]
    demo.append(a1[0])
    demo.append(subnames[i])
    try:
      demo.append(a2[2])
    except:
      demo.append(a2[0])
    csecfac.append(demo)

  print(asecfac)
  print(bsecfac)
  print(csecfac)

  subjectslist=select_and_crossover(sublist,sem,sec,course)
  import pandas as pd
  
  #subjectslist=pd.DataFrame(subjectslist)
  print(subjectslist)
  
  print(subjectslist[1])

  secfac=''
  romm=''
  rmno=['Room 1','Room 2','Room 3','Room 4','Room 5','Room 6','Room 7','Room 8','Room 9','Room 10','Room 11','Room 12']
  romm=random.choice(rmno)
  if sec=="A":
    secfac=asecfac

  elif sec=="B":
    secfac=bsecfac

  elif sec=="C":
    secfac=csecfac

  else:
    secfac=asecfac

    
  
    
  return render_template('home.html',subjectslist=subjectslist,sem=sem,secfac=secfac,sec=sec,room=romm)

@app.route('/predict', methods =  ['GET','POST'])
def procs():
    print("hit")
    key=request.args['data']
    print(key)    
    msg=key
    print(msg, flush=True)
    return render_template('predict.html',msg=msg)


if __name__ == '__main__':
    UPLOAD_FOLDER = 'D:/Upload'
    app.secret_key = "secret key"
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True)

