algorithm_word_conversion = {
    "R":"Rotate the right side of the cube once in the clockwise direction",
    "R'":"Rotate the right side of the cube once in the counter-clockwise direction",
    "R2":"Rotate the right side of the cube twice in any direction",
    "L":"Rotate the left side of the cube once in the clockwise direction",
    "L'":"Rotate the left side of the cube once in the counter-clockwise direction",
    "L2":"Rotate the left side of the cube twice in any direction",
    "F":"Rotate the front side of the cube once in the clockwise direction",
    "F'":"Rotate the front of the cube once in the counter-clockwise direction",
    "F2":"Rotate the front of the cube twice in any direction",
    "B":"Rotate the back of the cube once in the clockwise direction",
    "B'":"Rotate the back of the cube once in the counter-clockwise direction",
    "B2":"Rotate the back of the cube twice in any direction",
    "U":"Rotate the top of the cube once in the clockwise direction",
    "U'":"Rotate the top side of the cube once in the counter-clockwise direction",
    "U2":"Rotate the top side of the cube twice in any direction",
    "D":"Rotate the bottom of the cube once in the clockwise direction",
    "D'":"Rotate the bottom of the cube once in the counter-clockwise direction",
    "D2":"Rotate the bottom of the cube twice in any direction",
    "M":"Rotate the middle of the cube once in the clockwise direction",
    "M'":"Rotate the middle of the cube once in the counter-clockwise direction",
    "M2":"Rotate the middle of the cube twice in any direction",
    "y":"Rotate the whole cube once to the right",
    "y'":"Rotate the whole cube once to the left",
    "y2":"Rotate the whole cube twice in any direction",
}

def cross_solution_to_words(cross_solution: list[str]):
    print("Cross Solution:")
    for i in range(len(cross_solution)):
        move = cross_solution[i]
        print(f"{i+1}. {algorithm_word_conversion[move]}")
    print()

def corner_solution_to_words(corner_solution: list[list[str]]):
    for i in range(len(corner_solution)):
        print(f"Corner Solution {i}:")
        for j in range(len(corner_solution[i])):
            move = corner_solution[i][j]
            print(f"{i+1}. {algorithm_word_conversion[move]}")
        print()
    print()

def second_layer_edge_solution_to_words(edge_solution: list[list[str]]):
    for i in range(len(edge_solution)):
        print(f"Edge Solution {i}:")
        for j in range(len(edge_solution[i])):
            move = edge_solution[i][j]
            print(f"{i+1}. {algorithm_word_conversion[move]}")
        print()
    print()

def last_layer_cross_solution_to_words(last_layer_cross_solution: list[str]):
    print("Last Layer Cross Solution:")
    for i in range(len(last_layer_cross_solution)):
        move = last_layer_cross_solution[i]
        print(f"{i+1}. {algorithm_word_conversion[move]}")
    print()

def oll_solution_to_words(oll_solution: list[str]):
    print("OLL Solution:")
    for i in range(len(oll_solution)):
        move = oll_solution[i]
        print(f"{i+1}. {algorithm_word_conversion[move]}")
    print()

def pll_solution_to_words(pll_solution: list[str]):  
    print("PLL Solution:")
    for i in range(len(pll_solution)):
        move = pll_solution[i]
        print(f"{i+1}. {algorithm_word_conversion[move]}")
    print()