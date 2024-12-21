import random
BITCOUNT = 10


def fitness(member):
  if member < 0 or member >= 1:
    return -1
  elif member >= 0 and member < 2:
    return 60.0
  elif member >= 2 and member < 3:
    return member + 30.0
  elif member >= 3 and member < 4:
    return 120.0
  elif member >= 4 and member < 5:
    return -0.83333 * member + 220
  elif member >= 5 and member < 6:
    return 1.75 * member - 322.5
  elif member >= 6 and member < 7:
    return 150.0
  elif member >= 7 and member < 8:
    return 2.0 * member - 450
  elif member >= 8 and member < 9:
    return -1.8 * member + 918
  else:
    return 0


from static.assets.css.bootstrap.design import uicompute

class SelectionStrategy(object):
  """Abstract class for a selection strategy."""
  def select_and_crossover(self, population, crossing):
    raise Exception("Method not implemented.")



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

def crossover(a, b):
  try:
    pos = random.randint(1, BITCOUNT - 1)
    mask = (2 ** pos) - 1
    invmask = (2 ** BITCOUNT) - mask - 1
    na = (a & invmask) | (b & mask)
    nb = (b & invmask) | (a & mask)
    return na, nb
  except:
    print('Done')

def select_and_crossover(sublist,sem,sec,course):
  hrs=40
  print("----------------------Subjects-------------------------")
  print(sublist)
  print("----------------------Semester-------------------------")
  print(sem)
  print("----------------------Section-------------------------")
  print(sec)
  try:
    na,nb=crossover(sem,hrs)  
    print("----------------------Crossover-------------------------")
    print(na)
    print(nb)    
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
  except:
    print("Crossover done")
  from static.assets.css.bootstrap.design import uicompute
  print('---------Subject Vals--------------------')
  for i in range(len(sublist)):
    val=sublist[i]
    a1=val.split('-')
    print(a1)
  subjectslist=uicompute.select_and_crossover(sublist,int(sem),sec,course)
  return subjectslist

if __name__ == "__main__":
  for i in range(0, 1024):
    print (i, fitness(i))
