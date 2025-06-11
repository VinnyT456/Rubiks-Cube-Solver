import copy
import alphacube
import numpy as np
import itertools

class CubeSolver:
    def __init__(self,scramble):
        self.scramble = scramble
        self.color_key = {
            "White":0,
            "Yellow":1,
            "Orange":2,
            "Red":3,
            "Blue":4,
            "Green":5
        }
        self.ergonomic_bias = {
            "U": 0.9,   "U'": 0.9,  "U2": 0.8,
            "R": 0.8,   "R'": 0.8,  "R2": 0.75,
            "L": 0.55,  "L'": 0.4,  "L2": 0.3,
            "F": 0.7,   "F'": 0.6,  "F2": 0.6,
            "D": 0.3,   "D'": 0.3,  "D2": 0.2,
            "B": 0.05,  "B'": 0.05, "B2": 0.01,
            "u": 0.45,  "u'": 0.45, "u2": 0.4,
            "r": 0.3,   "r'": 0.3,  "r2": 0.25,
            "l": 0.2,   "l'": 0.2,  "l2": 0.15,
            "f": 0.35,  "f'": 0.3,  "f2": 0.25,
            "d": 0.15,  "d'": 0.15, "d2": 0.1,
            "b": 0.03,  "b'": 0.03, "b2": 0.01
        }
        alphacube.load()

    def sort_scramble(self):
        def flatten_scramble(scramble):
            new_scramble = [y for x in scramble for y in x]
            return new_scramble
        final_scramble = [[] for _ in range(6)]
        for face in self.scramble:
            center = face[4]
            final_scramble[self.color_key[center]] = face.copy()
        final_scramble = flatten_scramble(final_scramble)
        return final_scramble
    
    def color_to_number(self,scramble):
        return np.array([self.color_key[_] for _ in scramble])
    
    def solve(self):
        color_scramble = self.sort_scramble()
        final_scramble = self.color_to_number(color_scramble)
        solution = alphacube.solve(
            format = "stickers",
            scramble = final_scramble,
            beam_width = 1024,
            allow_wide = False,
            ergonomic_bias = self.ergonomic_bias
        )
        if (solution == None):
            return []
        return solution
    
class OLLSolver:
    def __init__(self,case):
        self.case = case
        self.oll_cases = {
            'Cross': {
                "R U2 R' U' R U' R'":
                [
                    [1,0,0],
                    [1,0,0],
                    [0,0,0],
                    [1,0,0]
                ],
                "R U R' U R U2 R'":
                [
                    [0,0,1],
                    [0,0,1],
                    [0,0,1],
                    [0,0,0]
                ],
                "(R U2 R') (U' R U R') (U' R U' R')":
                [
                    [1,0,1],
                    [0,0,0],
                    [1,0,1],
                    [0,0,0]
                ],
                "R U2 R2 U' R2 U' R2 U2 R":
                [
                    [0,0,1],
                    [0,0,0],
                    [1,0,1],
                    [0,0,1]
                ],
                "(r U R' U') (r' F R F')":
                [
                    [1,0,0],
                    [0,0,0],
                    [0,0,1],
                    [0,0,0]
                ],
                "y F' (r U R' U') r' F R":
                [
                    [1,0,0],
                    [0,0,1],
                    [0,0,0],
                    [0,0,0]
                ],
                "R2 D (R' U2 R) D' (R' U2 R')":
                [
                    [1,0,1],
                    [0,0,0],
                    [0,0,0],
                    [0,0,0]
                ]
            },
            'T-Shaped': {
                "(R U R' U') (R' F R F')":
                [
                    [1,1,0],
                    [0,0,0],
                    [1,1,0],
                    [0,0,0]
                ],
                "F (R U R' U') F'":
                [
                    [0,1,0],
                    [0,0,0],
                    [0,1,0],
                    [1,0,1]
                ],
            },
            'Squares': {
                "(r' U2 R U R' U r)":
                [
                    [0,0,0],
                    [0,0,1],
                    [0,1,1],
                    [0,1,1]
                ],
                "(r U2 R' U' R U' r')":
                [
                    [1,1,0],
                    [1,0,0],
                    [0,0,0],
                    [1,1,0]
                ],
            },
            'C-Shapes': {
                "(R U R2 U') (R' F R U) R U' F'":
                [
                    [0,1,0],
                    [0,0,1],
                    [0,1,0],
                    [1,0,0]
                ],
                "R' U' (R' F R F') U R":
                [
                    [0,0,0],
                    [1,1,1],
                    [0,0,0],
                    [0,1,0]
                ],
            },
            'W-Shapes': {
                "y2 (R U R' F) (R U R' U') (R' F R U') (R' F R F')":
                [
                    [1,0,0],
                    [0,1,1],
                    [0,1,0],
                    [0,0,0]
                ],
                "(R U R' U) (R U' R' U') (R' F R F')":
                [
                    [0,1,0],
                    [1,1,0],
                    [0,0,1],
                    [0,0,0]
                ],
            },
            'Corners Correct, Edges Flipped': {
                "(r U R' U') M (U R U' R')":
                [
                    [0,1,0],
                    [0,1,0],
                    [0,0,0],
                    [0,0,0]
                ],
                "(R U R' U') M' (U R U' r')":
                [
                    [0,1,0],
                    [0,0,0],
                    [0,1,0],
                    [0,0,0]
                ],
            },
            'P-Shapes': {
                "(R' U' F) (U R U' R') F' R":
                [
                    [1,1,0],
                    [0,0,0],
                    [0,0,1],
                    [0,1,0]
                ],
                "R U B' (U' R' U) (R B R')":
                [
                    [1,0,0],
                    [0,0,0],
                    [0,1,1],
                    [0,1,0]
                ],
                "f' (L' U' L' U) f":
                [
                    [0,0,0],
                    [1,1,1],
                    [0,1,0],
                    [0,0,0]
                ],
                "f (R U R' U') f'":
                [
                    [0,0,0],
                    [0,0,0],
                    [0,1,0],
                    [1,1,1]
                ]
            },
            'I-Shapes': {
                "f (R U R' U') (R U R' U') f'":
                [
                    [0,1,1],
                    [0,0,0],
                    [1,1,0],
                    [1,0,1]
                ],
                "r' U' r (U' R' U R) (U' R' U R) r' U r":
                [
                    [0,1,0],
                    [1,0,1],
                    [0,1,0],
                    [1,0,1]
                ],
                "(R U R' U R U') y (R U' R') F'":
                [ 
                    [1,0,0],
                    [1,1,1],
                    [0,0,1],
                    [0,1,0],
                ],
                "y (R' F R U) (R U' R2 F') R2 U' R' (U R U R')":
                [
                    [0,0,0],
                    [1,1,1],
                    [0,0,0],
                    [1,1,1]
                ],
            },
            'Fish Shapes': {
                "(R U R' U') R' F (R2 U R' U') F'":
                [
                    [1,1,0],
                    [0,1,0],
                    [1,0,0],
                    [1,0,0]
                ],
                "(R U R' U) (R' F R F') (R U2 R')":
                [
                    [0,0,1],
                    [0,1,0],
                    [0,1,1],
                    [0,0,1]
                ],
                "(R U2) (R2 F R F') (R U2 R')":
                [
                    [1,0,0],
                    [0,0,1],
                    [0,1,0],
                    [0,1,0]
                ],
                "F (R U' R' U') (R U R' F')":
                [
                    [1,1,0],
                    [0,1,1], 
                    [0,0,0],
                    [0,0,0]
                ],
            },
            'Knight Move Shapes': {
                "(r U' r') (U' r U r') y' (R' U R)":
                [
                    [0,1,1],
                    [0,0,1],
                    [0,1,1],
                    [0,0,0]
                ],
                "(R' F R) (U R' F' R) (F U' F')":
                [
                    [1,1,0],
                    [0,0,0],
                    [1,1,0],
                    [1,0,0]
                ],
                "(r U r') (R U R' U') (r U' r')":
                [
                    [1,1,0],
                    [1,0,0],
                    [0,1,0],
                    [1,0,0]
                ],
                "(r' U' r) (R' U' R U) (r' U r)":
                [
                    [0,1,0],
                    [0,0,1],
                    [0,1,1],
                    [0,0,1]
                ]
            },
            'Awkward Shapes': {
                "y (R U R' U') (R U' R') (F' U' F) (R U R')":
                [
                    [0,0,0],
                    [1,1,0],
                    [0,1,0],
                    [0,0,1]
                ],
                "y' F U (R U2 R' U') (R U2 R' U') F'":
                [
                    [0,1,1],
                    [0,0,0],
                    [1,0,0],
                    [0,1,0]
                ],
                "(R U R' U R U2 R') F (R U R' U') F'":
                [
                    [0,1,0],
                    [0,1,0],
                    [1,0,1],
                    [0,0,0]
                ],
                "y (R' F R F') (R' F R F') (R U R' U') (R U R')":
                [
                    [1,0,1],
                    [0,1,0],
                    [0,1,0],
                    [0,0,0]
                ]
            },
            'L-Shapes': {
                "F (R U R' U') (R U R' U') F'":
                [
                    [0,1,1],
                    [0,1,0],
                    [1,0,0],
                    [1,0,1]
                ],
                "R' U' (R' F R F') (R' F R F') U R":
                [
                    [1,1,0],
                    [1,0,1],
                    [0,0,1],
                    [0,1,0]
                ],
                "r U' r2' U r2 U r2' U' r":
                [
                    [0,1,1],
                    [0,0,0],
                    [1,0,0],
                    [1,1,1]
                ],
                "r' U r2 U' r2' U' r2 U r'":
                [
                    [0,0,1],
                    [0,0,0],
                    [1,1,0],
                    [1,1,1]
                ],
                "(r' U' R U') (R' U R U') R' U2 r":
                [
                    [0,0,0],
                    [1,0,1],
                    [0,1,0],
                    [1,1,1]
                ],
                "(r U R' U) (R U' R' U) R U2' r'":
                [
                    [0,1,0],
                    [1,0,1],
                    [0,0,0],
                    [1,1,1]
                ]
            },
            'Lightning Bolts': {
                "(r U R' U R U2' r')":
                [
                    [0,1,1],
                    [0,1,1],
                    [0,0,1],
                    [0,0,0],
                ],
                "(r' U' R U' R' U2 r)":
                [
                    [1,0,0],
                    [1,1,0],
                    [1,1,0],
                    [0,0,0]
                ],
                "r' (R2 U R' U R U2 R') U M'":
                [
                    [0,0,1],
                    [0,0,1],
                    [0,1,1],
                    [0,1,0]
                ],
                "M' (R' U' R U' R' U2 R) U' M":
                [
                    [1,1,0],
                    [1,0,0],
                    [1,0,0],
                    [0,1,0]
                ],
                "(L F') (L' U' L U) F U' L'":
                [
                    [0,1,0],
                    [1,0,0],
                    [0,1,1],
                    [0,0,0]
                ],
                "(R' F) (R U R' U') F' U R":
                [
                    [0,1,0],
                    [0,0,0],
                    [1,1,0],
                    [0,0,1]
                ]
            },
            'Dot Shapes': {
                "(R U2') (R2' F R F') U2' (R' F R F')":
                [
                    [0,1,0],
                    [1,1,1],
                    [0,1,0],
                    [1,1,1]
                ],
                "F (R U R' U') F' f (R U R' U') f'":
                [
                    [0,1,1],
                    [0,1,0],
                    [1,1,0],
                    [1,1,1]
                ],
                "f (R U R' U') f' U' F (R U R' U') F'":
                [
                    [0,1,0],
                    [0,1,1],
                    [0,1,1],
                    [0,1,1]
                ],
                "f (R U R' U') f' U F (R U R' U') F'":
                [
                    [1,1,0],
                    [1,1,0],
                    [0,1,0],
                    [1,1,0]
                ],
                "y R U2' (R2' F R F') U2' M' (U R U' r')":
                [
                    [1,1,1],
                    [0,1,0],
                    [0,1,0],
                    [0,1,0]
                ],
                "M U (R U R' U') M' (R' F R F')":
                [
                    [0,1,0],
                    [1,1,0],
                    [0,1,0],
                    [0,1,1]
                ],
                "(R U R' U) (R' F R F') U2' (R' F R F')":
                [
                    [0,1,0],
                    [0,1,1],
                    [1,1,0],
                    [0,1,1]
                ],
                "M U (R U R' U') M2' (U R U' r')":
                [
                    [0,1,0],
                    [0,1,0],
                    [0,1,0],
                    [0,1,0]
                ]
            },
        }

class PLLSolver:
    def __init__(self,case):
        self.case = case
        self.pll_cases = {
            'A-Perm': {},
            'E-Perm': {},
            'F-Perm': {},
            'G-Perm': {},
            'H-Perm': {},
            'J-Perm': {},
            'N-Perm': {},
            'R-Perm': {},
            'T-Perm': {},
            'U-Perm': {},
            'V-Perm': {},
            'Y-Perm': {},
            'Z-Perm': {},
        }