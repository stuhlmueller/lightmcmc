#!/usr/bin/python

from __future__ import division
from collections import Counter
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
    return sorted([(x, i/xlen) for (x, i) in Counter(xs).items()])

def flip(weight=0.5):
    return np.random.uniform(0, 1) <= weight

def discrete(ps):
    c = np.cumsum(ps)
    y = np.random.uniform(0, 1)
    return np.searchsorted(c,y)


# --------------------------------------------------------------------
# Rejection

def single_rejection(model, condition, query):
    _, value = model({}, -1)
    while not condition(value):
        _, value = model({}, -1)
    return query(value)

def rejection(model, condition, query, num_samples):
    samples = mapp(lambda _: single_rejection(model, condition, query),
                   range(num_samples))
    return samples


# --------------------------------------------------------------------
# MCMC

class Choice(object):

    def __init__(self, sampler, scorer):
        self.sampler = sampler
        self.scorer = scorer

    def sample(self, world, name, tick):
        if world.has_key(name):
            draw = world[name]
            draw.choice = self
            draw.tick_touched = tick
            return draw.value
        else:
            value = self.sampler()
            draw = Draw(value, self, tick)
            world[name] = draw
            return value


class Draw(object):

    def __init__(self, value, choice, tick_touched, tick_created=None):
        self.value = value
        self.choice = choice
        self.tick_touched = tick_touched
        self.tick_created = tick_created or tick_touched

    def resample(self):
        self.value = self.choice.sampler()
    
    @property
    def score(self):
        return self.choice.scorer(self.value)


class World(dict):

    def score(self):
        return product([draw.score for draw in self.values()])
            
    def copy(self):
        new_world = World()
        for (i, draw) in self.items():
            new_world[i] = Draw(draw.value, draw.choice, draw.tick_touched, draw.tick_created)
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
    
    def propose(self, tick):
        proposal = self.copy()
        proposal_draw_name = random.choice(proposal.keys())
        draw = proposal[proposal_draw_name]
        draw_score_pre = draw.score
        num_choices_pre = len(self)
        draw.resample()
        draw_score_post = draw.score
        new_world, new_value = model(proposal, tick)
        num_choices_post = len(new_world)
        inc_fw, inc_bw = new_world.clean(tick)
        bw = (1/num_choices_post) * draw_score_pre * inc_bw
        fw = (1/num_choices_pre) * draw_score_post * inc_fw
        return new_world, new_value, bw, fw

    
def initialize(model, condition):
    world, value = model(World(), -1)
    while not condition(value):
        world, value = model(World(), -1)
    return world, value


def mcmc(model, condition, query, num_steps, num_samples):
    world, value = initialize(model, condition)
    samples = []
    for i in range(num_samples):
        for j in range(num_steps):
            proposal, new_value, bw, fw = world.propose(i*j)
            if condition(new_value):
                p = (proposal.score()/world.score()) * (bw/fw)
                if p >= 1 or flip(p):
                    world, value = proposal, new_value
        samples.append(query(value))
    return samples


# --------------------------------------------------------------------
# Example

def church_flip(world, name, tick, p):
    choice = Choice(sampler=lambda: flip(p), scorer=lambda val: 1/3 if val else 2/3)
    return choice.sample(world, name, tick)

def model(init_world, tick):
    world = init_world.copy()
    rbit = lambda name: church_flip(world, name, tick, .5)
    A = rbit("a")
    if A:
        B = rbit(1)
    else:
        z = sum([rbit(1+n) for n in range(50)])
        B = 1 if ((z % 2) == 1) else 0
    w = 1/3 if B else 2/3
    C = church_flip(world, "c", tick, w)
    return (world, { "A" : A, "B" : B, "C" : C })

condition = lambda val: val["C"]

query = lambda val: val["A"]

np.random.seed(7)
random.seed(7)

print(histogram(rejection(model, condition, query, num_samples=7000)))
print(histogram(mcmc(model, condition, query, num_steps=500, num_samples=200)))
