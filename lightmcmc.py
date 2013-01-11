#!/usr/bin/python

from __future__ import division
from collections import Counter
from scipy.stats.distributions import binom
from datetime import datetime
from pytools import timedelta_to_seconds
import functools
import numpy as np
import operator
import random
try:
    from parallel import mapp
except ImportError:
    mapp = map


# --------------------------------------------------------------------
# Utils

def product(seq):
    return functools.reduce(operator.mul, seq, 1)


def histogram(xs):
    xlen = float(len(xs))
    return sorted([(x, i / xlen) for (x, i) in Counter(xs).items()])


def flip(weight=0.5):
    return np.random.uniform(0, 1) <= weight


def discrete(ps):
    c = np.cumsum(ps)
    y = np.random.uniform(0, 1)
    return np.searchsorted(c, y)


def random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def memoize(obj):
    cache = obj.cache = {}

    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        if args not in cache:
            cache[args] = obj(*args, **kwargs)
        return cache[args]
    return memoizer


def extract(d, keys):
    return dict((k, d[k]) for k in keys if k in d)


def sample_dict_subset(dic, subset_size):
    subset_keys = random.sample(dic, subset_size)
    return extract(dic, subset_keys)


# --------------------------------------------------------------------
# Rejection

def single_rejection(model, condition, query):
    _, value = model({}, -1)
    while not condition(value):
        _, value = model({}, -1)
    return query(value)


def rejection(model, condition, query, num_samples):
    samples = []
    for _ in range(int(num_samples / 100)):
        samples += mapp(lambda _: single_rejection(model, condition, query),
                        range(100))
    if num_samples % 100:
        samples += mapp(lambda _: single_rejection(model, condition, query),
                        range(num_samples % 100))
    return samples


# --------------------------------------------------------------------
# MCMC

class ZeroProbabilityException(Exception):
    pass


class Choice(object):

    def __init__(self, sampler, scorer, description):
        self.sampler = sampler
        self.scorer = scorer
        self.description = description

    def sample(self, world, name, tick):
        if name in world:
            draw = world[name]
            draw.choice = self
            draw.tick_touched = tick
            if draw.score == 0:
                raise ZeroProbabilityException
            return draw.value
        else:
            value = self.sampler()
            draw = Draw(value, self, tick, fixed=False)
            world[name] = draw
            return value

    def set(self, world, name, tick, fixed_val):
        if name in world:
            draw = world[name]
            draw.choice = self
            draw.tick_touched = tick
            draw.value = fixed_val
            draw.fixed = True
            if draw.score == 0:
                raise ZeroProbabilityException
            return fixed_val
        else:
            draw = Draw(fixed_val, self, tick, fixed=True)
            world[name] = draw
            return fixed_val

    def __repr__(self):
        return self.description


class Draw(object):

    def __init__(self, value, choice, tick_touched,
                 tick_created=None, fixed=False):
        self.value = value
        self.choice = choice
        self.tick_touched = tick_touched
        self.tick_created = tick_created or tick_touched
        self.fixed = fixed

    def resample(self):
        self.value = self.choice.sampler()

    @property
    def score(self):
        return self.choice.scorer(self.value)

    def copy(self):
        return Draw(self.value, self.choice, self.tick_touched,
                    self.tick_created, self.fixed)

    def __repr__(self):
        return str(self.choice) + ": v=" + str(self.value) + \
            ", p=" + str(self.score)


class World(dict):

    def score(self):
        return product([draw.score for draw in self.values()])

    def copy(self):
        new_world = World()
        for (i, draw) in self.items():
            new_world[i] = draw.copy()
        return new_world

    def clean(self, tick):
        inc_fw, inc_bw = 1, 1
        for (name, draw) in self.items():
            if draw.tick_created == tick:
                inc_fw = draw.score * inc_fw
            elif draw.tick_touched != tick:
                self.pop(name)
                inc_bw = draw.score * inc_bw
        return inc_fw, inc_bw

    def propose(self, model, tick):
        proposal = self.copy()
        proposable_draws_pre = [d for d in proposal.values() if not d.fixed]
        draw = random.choice(proposable_draws_pre)
        draw_score_pre = draw.score
        draw.resample()
        draw_score_post = draw.score
        new_world, new_value = model(proposal, tick)
        inc_fw, inc_bw = new_world.clean(tick)
        proposable_draws_post = [d for d in new_world.values() if not d.fixed]
        bw = (1 / len(proposable_draws_post)) * draw_score_pre * inc_bw
        fw = (1 / len(proposable_draws_pre)) * draw_score_post * inc_fw
        return new_world, new_value, bw, fw


def initialize(model, condition):
    world, value = model(World(), -1)
    while not condition(value):
        world, value = model(World(), -1)
    return world, value


def mcmc(model, condition, query, num_steps, num_samples=None, runtime=None):
    assert num_samples or runtime
    world, value = initialize(model, condition)
    results = {"samples": [],
               "timing": []}
    i = 0
    t_0 = datetime.now()
    t = 0
    while (num_samples and i < num_samples) or (runtime and t < runtime):
        for j in range(num_steps):
            try:
                proposal, new_value, bw, fw = world.propose(model, i * j)
            except ZeroProbabilityException:
                pass
            else:
                if condition(new_value):
                    p = (proposal.score() / world.score()) * (bw / fw)
                    if p >= 1 or flip(p):
                        world, value = proposal, new_value
            t = timedelta_to_seconds(datetime.now() - t_0)
            if runtime and t > runtime:
                break
        query_value = query(value)
        i += 1
        results["samples"].append(query_value)
        results["timing"].append(t)
    return results


# --------------------------------------------------------------------
# Random primitives

def church_flip(world, name, tick, p):
    choice = Choice(sampler=lambda: flip(p),
                    scorer=lambda val: p if val else 1 - p,
                    description="flip")
    return choice.sample(world, name, tick)


def church_flip_fixed(world, name, tick, p, fixed_val):
    choice = Choice(sampler=lambda: flip(p),
                    scorer=lambda val: p if val else 1 - p,
                    description="flip")
    return choice.set(world, name, tick, fixed_val)


def sample_geometric(p):
    if flip(p):
        return 1
    else:
        return 1 + sample_geometric(p)


def score_geometric(p, n):
    if n < 1:
        return 0.0
    else:
        return (1 - p) ** (n - 1) * p


def church_geometric(world, name, tick, p):
    choice = Choice(sampler=lambda: sample_geometric(p),
                    scorer=lambda v: score_geometric(p, v),
                    description="geometric")
    return choice.sample(world, name, tick)


def sample_uniform(low, high):
    return np.random.uniform(low, high)


def score_uniform(low, high, v):
    if v < low or v > high:
        return 0.0
    else:
        return 1.0 / (high - low)


def church_uniform(world, name, tick, low, high):
    choice = Choice(sampler=lambda: sample_uniform(low, high),
                    scorer=lambda v: score_uniform(low, high, v),
                    description="uniform")
    return choice.sample(world, name, tick)


def sample_integer(low, high):
    return np.random.randint(low, high)


def score_integer(low, high, v):
    if v < low or v >= high:
        return 0.0
    else:
        return 1.0 / (high - low)


def church_sampleinteger(world, name, tick, low, high):
    choice = Choice(sampler=lambda: sample_integer(low, high),
                    scorer=lambda v: score_integer(low, high, v),
                    description="sample_int")
    return choice.sample(world, name, tick)


def church_listdraw(world, name, tick, lst):
    i = church_sampleinteger(world, name, tick, 0, len(lst))
    return lst[i]


def sample_binomial(n, p):
    return np.random.binomial(n, p)


@memoize
def score_binomial(n, k, p):
    if k > n:
        return 0.0
    else:
        return binom.pmf(k, n, p)


def church_binomial(world, name, tick, n, p):
    choice = Choice(sampler=lambda: sample_binomial(n, p),
                    scorer=lambda k: score_binomial(n, k, p),
                    description="binomial")
    return choice.sample(world, name, tick)


def church_binomial_fixed(world, name, tick, n, p, fixed_val):
    choice = Choice(sampler=lambda: sample_binomial(n, p),
                    scorer=lambda k: score_binomial(n, k, p),
                    description="binomial/fixed")
    return choice.set(world, name, tick, fixed_val)


# --------------------------------------------------------------------
# Examples and tests

def test_constraints():

    def model(init_world, tick):
        world = init_world.copy()
        A = church_flip(world, "B", tick, .5)
        B_w = .00001 if A else .00002
        B = church_flip_fixed(world, "A", tick, B_w, True)
        return (world, {"A": A, "B": B})

    condition = lambda val: val["B"]

    query = lambda val: val["A"]

    np.random.seed(7)
    random.seed(7)

    mcmc_results = mcmc(model, condition, query, num_steps=100, num_samples=173)
    mcmc_samples = mcmc_results["samples"]
    print(histogram(mcmc_samples))


def test_transdimensional():

    def model(init_world, tick):
        world = init_world.copy()
        rbit = lambda name: church_flip(world, name, tick, .5)
        A = rbit("a")
        if A:
            B = rbit(1)
        else:
            z = sum([rbit(1 + n) for n in range(50)])
            B = 1 if ((z % 2) == 1) else 0
        w = 1 / 3 if B else 2 / 3
        C = church_flip(world, "c", tick, w)
        return (world, {"A": A, "B": B, "C": C})

    condition = lambda val: val["C"]

    query = lambda val: val["A"]

    random_seed(9)

    rejection_samples = rejection(model, condition, query, num_samples=500)
    mcmc_results = mcmc(model, condition, query, num_steps=1000, num_samples=100)
    mcmc_samples = mcmc_results["samples"]
    print(histogram(rejection_samples))
    print(histogram(mcmc_samples))


def test_hierarchical():

    ELEMENTS_PER_BAG = 10

    observed_samples = {
        0: 1,
        1: 1,
        2: 1,
        3: 1,
        4: 1,
        5: 10,
        6: 10,
        7: 10,
        8: 10,
        9: 10
        }

    def model(init_world, tick, data=observed_samples):
        world = init_world.copy()
        num_bag_types = church_geometric(world, "num_bag_types", tick, .4)
        bag_ps = [church_uniform(world, "bag_type_p_%i" % i, tick, 0, 1)
                  for i in range(num_bag_types)]
        bag_samples = {}
        for bag in data.keys():
            bag_p = church_listdraw(world, "bag_p_%i" % bag, tick, bag_ps)
            bag_k = church_binomial_fixed(world, "bag_k_%i" % bag, tick,
                                          ELEMENTS_PER_BAG, bag_p, fixed_val=data[bag])
            bag_samples[bag] = bag_k
        return (world, {"num_bag_types": num_bag_types,
                        "bag_samples": bag_samples})

    def condition(val, data=observed_samples):
        return val["bag_samples"] == data

    def query(val, data=observed_samples):
        return val["num_bag_types"]

    def getdata(num_clusters):
        data = {}
        for i in range(10):
            data[i] = ((i % num_clusters) + 1) * 20
        return data

    random_seed(10)

    def runmcmc():
        mcmc_results = mcmc(model, condition, query, num_steps=100, runtime=5 * 60)
        mcmc_samples = mcmc_results["samples"]
        return ("mcmc", len(mcmc_samples), histogram(mcmc_samples))

    all_results = mapp(lambda f: f(), [runmcmc] * 5)
    for result in all_results:
        print result


test_hierarchical()
