import numpy as np
import math
import copy
from sklearn.neighbors import KDTree

class Cube:
    def __init__(self,cube_data):
        self.cube = cube_data
        
    def display_cube(self) -> None:
        print(np.array([face for face in self.cube.values()]))

    def R_move(self) -> dict[str, np.ndarray]:
        #Faces that's affected by the move (U,F,B,R,D)
        self.cube["R"] = np.rot90(self.cube["R"],-1)

        F_right = self.cube["F"][:,2].copy()
        U_right = self.cube["U"][:,2].copy()[::-1]
        D_right = self.cube["D"][:,2].copy()
        B_left = self.cube["B"][:,0].copy()[::-1]

        #Sequence F -> U -> B -> D -> F
        self.cube["F"][:,2], self.cube["U"][:,2], self.cube["B"][:,0], self.cube["D"][:,2] = D_right,F_right,U_right,B_left
    
    def R_prime_move(self) -> dict[str, np.ndarray]:
        #Faces that's affected by the move (U,F,B,R,D)
        self.cube["R"] = np.rot90(self.cube["R"],1)

        F_right = self.cube["F"][:,2].copy()
        U_right = self.cube["U"][:,2].copy()
        D_right = self.cube["D"][:,2].copy()[::-1]
        B_left = self.cube["B"][:,0].copy()[::-1]

        #Sequence F -> D -> B -> U -> F 
        self.cube["F"][:,2], self.cube["U"][:,2], self.cube["B"][:,0], self.cube["D"][:,2] = U_right,B_left,D_right,F_right
    
    def L_move(self) -> dict[str, np.ndarray]:
        #Faces that's affected by the move (U,F,B,L,D)
        self.cube["L"] = np.rot90(self.cube["L"],-1)   

        F_left = self.cube["F"][:,0].copy()
        U_left = self.cube["U"][:,0].copy()
        D_left = self.cube["D"][:,0].copy()[::-1]
        B_right = self.cube["B"][:,2].copy()[::-1]

        #Sequence F -> D -> B -> U -> F
        self.cube["F"][:,0], self.cube["U"][:,0], self.cube["B"][:,2], self.cube["D"][:,0] = U_left,B_right,D_left,F_left
    
    def L_prime_move(self) -> dict[str, np.ndarray]:
        #Faces that's affected by the move (U,F,B,L,D)
        self.cube["L"] = np.rot90(self.cube["L"],1)

        F_left = self.cube["F"][:,0].copy()
        U_left = self.cube["U"][:,0].copy()[::-1]
        D_left = self.cube["D"][:,0].copy()
        B_right = self.cube["B"][:,2].copy()[::-1]

        #Sequence F -> U -> B -> D -> F
        self.cube["F"][:,0], self.cube["U"][:,0], self.cube["B"][:,2], self.cube["D"][:,0] = D_left,F_left,U_left,B_right

    def U_move(self) -> dict[str, np.ndarray]:
        #Faces that's affected by the move (U,F,B,L,R)
        self.cube["U"] = np.rot90(self.cube["U"], -1)

        F_top = self.cube["F"][0,:].copy()
        L_top = self.cube["L"][0,:].copy()
        B_top = self.cube["B"][0,:].copy()
        R_top = self.cube["R"][0,:].copy()

        #Sequence F -> L -> B -> R -> F
        self.cube["F"][0,:], self.cube["L"][0,:], self.cube["B"][0,:], self.cube["R"][0,:] = R_top, F_top, L_top, B_top
    
    def U_prime_move(self) -> dict[str, np.ndarray]:
        #Faces that's affected by the move (U,F,B,L,R)
        self.cube["U"] = np.rot90(self.cube["U"],1)

        F_top = self.cube["F"][0,:].copy()
        L_top = self.cube["L"][0,:].copy()
        B_top = self.cube["B"][0,:].copy()
        R_top = self.cube["R"][0,:].copy()

        #Sequence F -> R -> B -> L -> F
        self.cube["F"][0,:], self.cube["L"][0,:], self.cube["B"][0,:], self.cube["R"][0,:] = L_top, B_top, R_top, F_top     
    
    def D_move(self) -> dict[str, np.ndarray]:
        #Faces that's affected by the move (R,F,B,L,D)
        self.cube["D"] = np.rot90(self.cube["D"],-1)

        F_bottom= self.cube["F"][2,:].copy()
        L_bottom = self.cube["L"][2,:].copy()
        R_bottom = self.cube["R"][2,:].copy()
        B_bottom = self.cube["B"][2,:].copy()

        #Sequence F -> R -> B -> L -> F
        self.cube["F"][2,:], self.cube["R"][2,:], self.cube["B"][2,:], self.cube["L"][2,:] = L_bottom,F_bottom,R_bottom,B_bottom

    def D_prime_move(self) -> dict[str, np.ndarray]: 
        #Faces that's affected by the move (R,F,B,L,D)
        self.cube["D"] = np.rot90(self.cube["D"],1)

        F_bottom = self.cube["F"][2,:].copy()
        L_bottom = self.cube["L"][2,:].copy()
        R_bottom = self.cube["R"][2,:].copy()
        B_bottom = self.cube["B"][2,:].copy()

        #Sequence F -> L -> B -> R -> F
        self.cube["F"][2,:], self.cube["R"][2,:], self.cube["B"][2,:], self.cube["L"][2,:] = R_bottom,B_bottom,L_bottom,F_bottom

    def F_move(self) -> dict[str, np.ndarray]:
        #Faces that's affected by the move (R,F,U,L,D)
        self.cube["F"] = np.rot90(self.cube["F"],-1)

        U_buttom = self.cube["U"][2,:].copy()
        R_left = self.cube["R"][:,0].copy()[::-1]
        D_top = self.cube["D"][0,:].copy()
        L_right = self.cube["L"][:,2].copy()[::-1]

        #Sequence U -> R -> D -> L -> U
        self.cube["U"][2,:], self.cube["R"][:,0], self.cube["D"][0,:], self.cube["L"][:,2] = L_right,U_buttom,R_left,D_top       
    
    def F_prime_move(self) -> dict[str, np.ndarray]:       
        #Faces that's affected by the move (R,F,U,L,D)
        self.cube["F"] = np.rot90(self.cube["F"],1)

        U_buttom = self.cube["U"][2,:].copy()[::-1]
        R_left = self.cube["R"][:,0].copy()
        D_top = self.cube["D"][0,:].copy()[::-1]
        L_right = self.cube["L"][:,2].copy()

        #Sequence U -> L -> D -> R -> U
        self.cube["U"][2,:], self.cube["R"][:,0], self.cube["D"][0,:], self.cube["L"][:,2] = R_left,D_top,L_right,U_buttom
 
    def B_move(self) -> dict[str, np.ndarray]:      
        #Faces that's affected by the move (R,B,U,L,D)
        self.cube["B"] = np.rot90(self.cube["B"],-1)

        U_top = self.cube["U"][0,:].copy()[::-1]
        R_right = self.cube["R"][:,2].copy()
        D_bottom = self.cube["D"][2,:].copy()[::-1]
        L_left = self.cube["L"][:,0].copy()

        #Sequence U -> L -> D -> R -> U
        self.cube["U"][0,:], self.cube["R"][:,2], self.cube["D"][2,:], self.cube["L"][:,0] = R_right,D_bottom,L_left,U_top
            
    def B_prime_move(self) -> dict[str, np.ndarray]:       
        #Faces that's affected by the move (R,B,U,L,D)
        self.cube["B"] = np.rot90(self.cube["B"],1)
        
        U_top = self.cube["U"][0,:].copy()
        R_right = self.cube["R"][:,2].copy()[::-1]
        D_bottom = self.cube["D"][2,:].copy()
        L_left = self.cube["L"][:,0].copy()[::-1]

        #Sequence U -> R -> D -> L -> U
        self.cube["U"][0,:], self.cube["R"][:,2], self.cube["D"][2,:], self.cube["L"][:,0] = L_left,U_top,R_right,D_bottom

    def M_move(self) -> dict[str, np.ndarray]:
        U_middle = self.cube["U"][:,1].copy()
        F_middle = self.cube["F"][:,1].copy()
        D_middle = self.cube["D"][:,1].copy()
        B_middle = self.cube["B"][:,1].copy()

        #Sequence U -> F -> D -> B -> U
        self.cube["U"][:,1], self.cube["F"][:,1], self.cube["D"][:,1], self.cube["B"][:,1] = B_middle,U_middle,F_middle,D_middle
 
    def M_prime_move(self) -> dict[str, np.ndarray]:
        U_middle = self.cube["U"][:,1].copy()
        F_middle = self.cube["F"][:,1].copy()
        D_middle = self.cube["D"][:,1].copy()
        B_middle = self.cube["B"][:,1].copy()

        #Sequence U -> B -> D -> F -> U
        self.cube["U"][:,1], self.cube["B"][:,1], self.cube["D"][:,1], self.cube["F"][:,1] = F_middle,U_middle,B_middle,D_middle

    def y_move(self) -> dict[str, np.ndarray]:      
        #Faces on the cube that will be affected by the rotation
        U_face = self.cube['U'].copy()
        F_face = self.cube['F'].copy()
        D_face = self.cube['D'].copy()
        B_face = self.cube['B'].copy()
        L_face = self.cube['L'].copy()   
        R_face = self.cube['R'].copy()

        #Rotate the necessary faces
        self.cube['U'] = np.rot90(U_face, -1)
        self.cube['D'] = np.rot90(D_face, 1)

        #Rotate the faces in the middle
        self.cube['F'],self.cube["R"],self.cube['B'],self.cube['L'] = R_face, B_face, L_face, F_face

    def y_prime_move(self) -> dict[str, np.ndarray]:
        #Faces on the cube that will be affected by the rotation
        U_face = self.cube['U'].copy()
        F_face = self.cube['F'].copy()
        D_face = self.cube['D'].copy()
        B_face = self.cube['B'].copy()
        L_face = self.cube['L'].copy()   
        R_face = self.cube['R'].copy()

        #Rotate the necessary faces
        self.cube['U'] = np.rot90(U_face, 1)
        self.cube['D'] = np.rot90(D_face, -1)

        #Rotate the faces in the middle
        self.cube['F'],self.cube["R"],self.cube['B'],self.cube['L'] = L_face, F_face, R_face, B_face

    def x_move(self) -> dict[str, np.ndarray]:
        #Faces on the cube that will be affected by the rotation
        U_face = self.cube['U'].copy()
        F_face = self.cube['F'].copy()
        D_face = self.cube['D'].copy()
        B_face = self.cube['B'].copy()
        L_face = self.cube['L'].copy()   
        R_face = self.cube['R'].copy()

        #Rotate the necessary faces
        U_face = np.rot90(U_face, -2)
        B_face = np.rot90(B_face, -2)

        #Rotate the faces in the middle
        self.cube['U'],self.cube["F"],self.cube['D'],self.cube['B'] = F_face, D_face, B_face, U_face
        
        #Rotate the faces on the side
        self.cube['L'] = np.rot90(L_face, 1)
        self.cube['R'] = np.rot90(R_face, -1)
     
    def x_prime_move(self) -> dict[str, np.ndarray]:
        #Faces on the cube that will be affected by the rotation
        U_face = self.cube['U'].copy()
        F_face = self.cube['F'].copy()
        D_face = self.cube['D'].copy()
        B_face = self.cube['B'].copy()
        L_face = self.cube['L'].copy()   
        R_face = self.cube['R'].copy()

        #Rotate the necessary faces
        D_face = np.rot90(D_face, -2)
        B_face = np.rot90(B_face, -2)

        #Rotate the faces in the middle
        self.cube['U'],self.cube["F"],self.cube['D'],self.cube['B'] = B_face, U_face, F_face, D_face
        
        #Rotate the faces on the side
        self.cube['L'] = np.rot90(L_face, -1)
        self.cube['R'] = np.rot90(R_face, 1)     
    
    def set_state(self, state):
        self.cube = copy.deepcopy(state)
    
    def get_state(self):
        return copy.deepcopy(self.cube)

    def get_sticker(self, face, index) -> str:
        return self.cube[face][index]
    
    def get_center_color(self, face):
        return self.cube[face][1,1]
    
    def get_face(self, face):
        return copy.deepcopy(self.cube[face])
    
    def apply_solution(self,solution):
        for _ in solution:
            self.apply_move(_)

    def apply_move(self, move):
        moves = {
            "R": self.R_move,
            "R'": self.R_prime_move,
            "L": self.L_move,
            "L'": self.L_prime_move,
            "U": self.U_move,
            "U'": self.U_prime_move,
            "D": self.D_move,
            "D'": self.D_prime_move,
            "F": self.F_move,
            "F'": self.F_prime_move,
            "B": self.B_move,
            "B'": self.B_prime_move,
            "M": self.M_move,
            "M'": self.M_prime_move,
            "y": self.y_move,
            "y'": self.y_prime_move,
            "x": self.x_move,
            "x'": self.x_prime_move,
        }

        double = False
        if '2' in move:
            double = True
            move = move.replace('2', '')

        if move in moves:
            moves[move]()  # apply once
            if double:
                moves[move]()  # apply again if it's a double

    def undo_move(self, move):
        moves = {
            "R": self.R_prime_move,
            "R'": self.R_move,
            "L": self.L_prime_move,
            "L'": self.L_move,
            "U": self.U_prime_move,
            "U'": self.U_move,
            "D": self.D_prime_move,
            "D'": self.D_move,
            "F": self.F_prime_move,
            "F'": self.F_move,
            "B": self.B_prime_move,
            "B'": self.B_move,
            "M": self.M_prime_move,
            "M'": self.M_move,
            "y": self.y_prime_move,
            "y'": self.y_move,
            "x": self.x_prime_move,
            "x'": self.x_move,
        }

        double = False
        if '2' in move:
            double = True
            move = move.replace('2', '')

        if move in moves:
            moves[move]()  # apply once
            if double:
                moves[move]()  # apply again if it's a double         
class cross_solver(Cube):
    def __init__(self, cube_data):
        super().__init__(cube_data.get_state())
        self.moves = ["U", "U'", "U2", "F", "F'", "F2", "R", "R'", "R2", "L", "L'", "L2", "D", "D'", "D2", "B", "B'", "B2"]
        self.threshold = 0

        self.cross_positions = [
            (('D', (0,1)), ('F', (2,1))),
            (('D', (1,0)), ('L', (2,1))),
            (('D', (1,2)), ('R', (2,1))),
            (('D', (2,1)), ('B', (2,1))),
        ]

        self.goal_edges = [
            {'colors': ('w', 'g'), 'positions': (('D', (0,1)), ('F', (2,1)))},
            {'colors': ('w', 'r'), 'positions': (('D', (1,0)), ('L', (2,1)))},
            {'colors': ('w', 'o'), 'positions': (('D', (1,2)), ('R', (2,1)))},
            {'colors': ('w', 'b'), 'positions': (('D', (2,1)), ('B', (2,1)))},
        ]

    def heuristic(self):
        total_cost = 0.0
        for goal in self.goal_edges:
            desired_white, desired_adjacent = goal['colors']
            goal_pos1, goal_pos2 = goal['positions']
            found = False

            for cross_pos in self.cross_positions:
                (face1, idx1), (face2, idx2) = cross_pos
                color1 = self.get_sticker(face1, idx1)
                color2 = self.get_sticker(face2, idx2)

                if {color1, color2} == {desired_white, desired_adjacent}:
                    if cross_pos == goal['positions']:
                        if self.get_sticker(goal_pos1[0], goal_pos1[1]) == desired_white:
                            cost = 0.0
                        else:
                            cost = 0.5
                    else:
                        cost = 0.5
                    found = True
                    break

            if not found:
                cost = 1.0

            total_cost += cost

        return math.ceil(total_cost)
            
    def ida_star(self):
        self.threshold = self.heuristic()

        def search(path, g):
            h = self.heuristic()
            if h == 0:
                return True, path
            f = g + h
            if f > self.threshold or g >= 8:
                return f, None

            min_cost = float('inf')
            for move in self.moves:
                if path and move[0] == path[-1][0]:
                    continue  # Avoid repeating the same face

                prev_state = self.get_state()
                self.apply_move(move)
                result, new_path = search(path + [move], g + 1)
                self.set_state(prev_state)  # Undo move

                if result is True:
                    return True, new_path
                if isinstance(result, (int, float)) and result < min_cost:
                    min_cost = result

            return min_cost, None

        while True:
            result, path = search([], 0)
            if result is True:
                return path
            if self.threshold >= 8:
                return None
            self.threshold = min(result, 8)

    def solve_cross(self):
        solution = self.ida_star()
        self.apply_solution(solution)
        new_cube = self.get_state()
        return new_cube, solution   

class corner_solver(Cube):
    def __init__(self,cube_data):
        super().__init__(cube_data.get_state())
        self.solutions = []

        self.corner_sticker_table = {
            ('R',(0,0)):[('U',(2,2)),('F',(0,2))],
            ('R',(0,2)):[('U',(0,2)),('B',(0,0))],
            ('R',(2,0)):[('D',(0,2)),('F',(2,2))],
            ('R',(2,2)):[('D',(2,2)),('B',(2,0))],

            ('L',(0,0)):[('U',(0,0)),('B',(0,2))],
            ('L',(0,2)):[('U',(2,0)),('F',(0,0))],
            ('L',(2,0)):[('D',(2,0)),('B',(2,2))],
            ('L',(2,2)):[('D',(0,0)),('F',(2,0))],

            ('U',(0,0)):[('L',(0,0)),('B',(0,2))],
            ('U',(0,2)):[('R',(0,2)),('B',(0,0))],
            ('U',(2,0)):[('L',(0,2)),('F',(0,0))],
            ('U',(2,2)):[('R',(0,0)),('F',(0,2))],

            ('D',(0,0)):[('L',(2,2)),('F',(2,0))],
            ('D',(0,2)):[('R',(2,0)),('F',(2,2))],
            ('D',(2,0)):[('L',(2,0)),('B',(2,2))],
            ('D',(2,2)):[('R',(2,2)),('B',(2,0))],

            ('F',(0,0)):[('U',(2,0)),('L',(0,2))],
            ('F',(0,2)):[('U',(2,2)),('R',(0,0))],
            ('F',(2,0)):[('U',(0,0)),('R',(2,2))],
            ('F',(2,2)):[('U',(0,2)),('L',(2,0))],
            
            ('B',(0,0)):[('U',(0,2)),('R',(0,2))],
            ('B',(0,2)):[('U',(0,0)),('L',(0,0))],
            ('B',(2,0)):[('U',(2,2)),('R',(2,0))],
            ('B',(2,2)):[('U',(2,0)),('L',(2,2))],
        }
        self.opposite_center_color_key = {
            'o':'r',
            'r':'o',
            'b':'g',
            'g':'b'
        }
        self.corner_move_table = {
            ('U', (0,0)): ('U', (0,2)),
            ('U', (0,2)): ('U', (2,2)),
            ('U', (2,2)): ('U', (2,0)),
            ('U', (2,0)): ('U', (0,0)),

            ('F',(0,0)): ('L', (0,0)),
            ('F',(0,2)): ('L', (0,0)),
            
            ('L',(0,0)): ('B',(0,0)),
            ('L',(0,2)): ('B',(0,2)),

            ('B',(0,0)): ('R',(0,0)),
            ('B',(0,2)): ('R',(0,2)),

            ('R',(0,0)): ('F',(0,0)),
            ('R',(0,2)): ('F',(0,2)),
        }
        
        self.positions = [(0,0),(0,2),(2,0),(2,2)]
        self.horizontal_faces = ['F','R','L','B']

        self.corner = None
        self.setup_move_length = 0

        self.new_cube = Cube(cube_data.get_state())

    def get_adjencent_sticker_positions(self,face,position):
        return self.corner_sticker_table[(face,position)]
    
    def locate_white_corner(self) -> tuple[str,(int,int)]:
        """
        Locate the best white corner piece.
        Top face, Middle face, Bottom face
        """

        #Locate the white corner piece on the horizontal faces
        for face in self.horizontal_faces:
            if (self.new_cube.get_sticker(face,(0,0)) == 'w'):
                return (face,(0,0))
            if (self.new_cube.get_sticker(face,(0,2)) == 'w'):
                return (face,(0,2))
        
        #Locate the white corner piece on the top face
        for position in self.positions:
            if (self.new_cube.get_sticker("U",position) == 'w'):
                return ("U", position)
            
        #Locate the white corner pieces that's on the bottom layer
        for face in self.horizontal_faces:
            if (self.new_cube.get_sticker(face,(2,0)) == 'w'):
                return (face,(2,0))
            if (self.new_cube.get_sticker(face,(2,2)) == 'w'):
                return (face,(2,2))
            
        #Locate the white corner piece on the bottom face
        for position in self.positions:
            if (self.new_cube.get_sticker("D",position) == 'w'):
                #Check if corner piece is oriented correctly
                sticker_position_1, sticker_position_2 = self.get_adjencent_sticker_positions("D",position)

                face_1 = sticker_position_1[0]
                face_2 = sticker_position_2[0]

                center_color_1, center_color_2 = self.new_cube.get_center_color(face_1),self.new_cube.get_center_color(face_2)
                sticker_color_1 = self.new_cube.get_sticker(face_1,sticker_position_1[1])
                sticker_color_2 = self.new_cube.get_sticker(face_2,sticker_position_2[1])

                if (sticker_color_1 == center_color_1 and sticker_color_2 == center_color_2):
                    continue
                
                return ("D", position)
            
        return None
    
    def update_corner_after_U_move(self) -> tuple[str,(int,int)]:
        return self.corner_move_table.get(self.corner)
    
    def find_position_of_insertion(self) -> list[str]:
        setup_move = []

        while True:
            corner_sticker_1, corner_sticker_2 = self.get_adjencent_sticker_positions(self.corner[0],self.corner[1])

            face_1 = corner_sticker_1[0]
            face_2 = corner_sticker_2[0]

            sticker_position_1 = corner_sticker_1[1]
            sticker_position_2 = corner_sticker_2[1]

            center_color_1 = self.new_cube.get_center_color(face_1)
            center_color_2 = self.new_cube.get_center_color(face_2)

            sticker_color_1 = self.new_cube.get_sticker(face_1, sticker_position_1)
            sticker_color_2 = self.new_cube.get_sticker(face_2, sticker_position_2)

            if (sticker_color_1 == center_color_2 and sticker_color_2 == center_color_1):
                break
            
            self.new_cube.apply_move("U")
            self.corner = self.update_corner_after_U_move()
            setup_move.append('U')

        return setup_move
    
    def generate_corner_solution_U_face(self) -> list[str]:
        """
        Generate the solution to solve the corner piece at the U face.
        """

        solution = self.find_position_of_insertion()
        
        #Undo the setup move
        for _ in range(len(solution)):
            self.new_cube.apply_move("U'")

        if (solution.count("U") == 2):
            solution = ["U2"]
        if (solution.count("U") == 3):
            solution = ["U'"]

        left_insertion_move = ["L'","U'","L","U"]
        right_insertion_move = ["R","U","R'","U'"]

        if (self.corner[1] == (0,0)):
            solution.append("y'")
            solution.extend(left_insertion_move*3)

        elif (self.corner[1] == (0,2)):
            solution.append("y")
            solution.extend(right_insertion_move*3)

        elif (self.corner[1] == (2,0)):
            solution.extend(left_insertion_move*3)

        elif (self.corner[1] == (2,2)):
            solution.extend(right_insertion_move*3)
        
        return solution
    
    def generate_corner_solution_F_face(self) -> list[str]:
        """
        Generate the solution to solve the corner piece at the F face.
        """

        solution = []

        if (self.corner[1] == (2,0)):
            solution = ["L'","U","L"]
        if (self.corner[1] == (2,2)):
            solution = ["R","U'","R'"]

        #Update the corner position if it was originally in the bottom layer
        self.corner = (self.corner[0],(0,self.corner[1][1])) if (len(solution) != 0) else self.corner

        self.new_cube.apply_solution(solution)

        corner_sticker_1, corner_sticker_2 = self.get_adjencent_sticker_positions(self.corner[0],self.corner[1])

        face_1 = corner_sticker_1[0]
        face_2 = corner_sticker_2[0]

        sticker_position_1 = corner_sticker_1[1]
        sticker_position_2 = corner_sticker_2[1]

        #Sticker on the top
        sticker_color_1 = self.new_cube.get_sticker(face_1,sticker_position_1)

        #Sticker/Center in the front/back
        sticker_color_2 = self.new_cube.get_sticker(face_2,sticker_position_2)
        center_color_2 = self.new_cube.get_center_color(face_2)

         #Center where the where sticker is 
        center_color_white = self.new_cube.get_center_color(self.corner[0])

        if (self.corner[1] == (0,0)):
            
            #Insertion slot right above the corner piece
            if ((sticker_color_1 == center_color_white) and (sticker_color_2 == center_color_2)):
                solution.extend(["U'","L'","U","L"])

            #Insertion slot counterclockwise position to the corner piece
            if ((self.opposite_center_color_key[sticker_color_1] == center_color_2) and (sticker_color_2 == center_color_white)):
                solution.extend(["U'","R","U","R'"])

            #Insertion slot diagonal to the corner piece
            if ((self.opposite_center_color_key[sticker_color_1] == center_color_white) and (self.opposite_center_color_key[sticker_color_2] == center_color_2)):
                solution.extend(["R'","U2","R"])

            #Insertion slot clockwise position to the corner piece
            if ((sticker_color_1 == center_color_2) and (self.opposite_center_color_key[sticker_color_2] == center_color_white)):
                solution.extend(["U","L","U","L'"])

        if (self.corner[1] == (0,2)):

            #Insertion slot right above the corner piece
            if ((sticker_color_1 == center_color_white) and (sticker_color_2 == center_color_2)):
                solution.extend(["U","R","U'","R'"])

            #Insertion slot counterclockwise position to the corner piece
            if ((sticker_color_1 == center_color_2) and (self.opposite_center_color_key[sticker_color_2] == center_color_white)):
                solution.extend(["U'","R'","U'","R"])

            #Insertion slot diagonal to the corner piece
            if ((self.opposite_center_color_key[sticker_color_1] == center_color_white) and (self.opposite_center_color_key[sticker_color_2] == center_color_2)):
                solution.extend(["L","U2","L'"])

            #Insertion slot clockwise position to the corner piece
            if ((self.opposite_center_color_key[sticker_color_1] == center_color_2) and (sticker_color_2 == center_color_white)):
                solution.extend(["U","L'","U'","L"])


        return solution
    
    def generate_corner_solution_R_face(self) -> list[str]:
        """
        Generate the solution to solve the corner piece at the R face.
        """

        solution = []
        
        #If the corner is in the bottom layer move it to the top
        if (self.corner[1] == (2,0)):
            solution = ["R","U'","R'"]
        if (self.corner[1] == (2,2)):
            solution = ["R'","U'","R","U"]

        #Update the corner position if it was originally in the bottom layer
        self.corner = (self.corner[0],(0,self.corner[1][1])) if (len(solution) != 0) else self.corner

        self.new_cube.apply_solution(solution)

        corner_sticker_1, corner_sticker_2 = self.get_adjencent_sticker_positions(self.corner[0],self.corner[1])

        face_1 = corner_sticker_1[0]
        face_2 = corner_sticker_2[0]

        sticker_position_1 = corner_sticker_1[1]
        sticker_position_2 = corner_sticker_2[1]

        #Sticker on the top
        sticker_color_1 = self.new_cube.get_sticker(face_1,sticker_position_1)

        #Sticker/Center in the front/back
        sticker_color_2 = self.new_cube.get_sticker(face_2,sticker_position_2)
        center_color_2 = self.new_cube.get_center_color(face_2)

        #Center where the where sticker is
        center_color_white = self.new_cube.get_center_color(self.corner[0])

        if (self.corner[1] == (0,0)):
        
            #Insertion slot right above the corner piece
            if ((sticker_color_1 == center_color_white) and (sticker_color_2 == center_color_2)):
                solution.extend(["R","U","R'"])

            #Insertion slot counterclockwise position to the corner piece
            if ((self.opposite_center_color_key[sticker_color_1] == center_color_2) and (sticker_color_2 == center_color_white)):
                solution.extend(["U2","R'","U","R"])

            #Insertion slot diagonal to the corner piece
            if ((self.opposite_center_color_key[sticker_color_1] == center_color_white) and (self.opposite_center_color_key[sticker_color_2] == center_color_2)):
                solution.extend(["U2","L","U","L'"])

            #Insertion slot clockwise position to the corner piece
            if ((sticker_color_1 == center_color_2) and (self.opposite_center_color_key[sticker_color_2] == center_color_white)):
                solution.extend(["L'","U","L"])

        #Corner in position (0,2)
        if (self.corner[1] == (0,2)):

            #Insertion slot right above the corner piece
            if ((sticker_color_1 == center_color_white) and (sticker_color_2 == center_color_2)):
                solution.extend(["R'","U'","R"])

            #Insertion slot counterclockwise position to the corner piece
            if ((sticker_color_1 == center_color_2) and (self.opposite_center_color_key[sticker_color_2] == center_color_white)):
                solution.extend(["L","U'","L'"])

            #Insertion slot diagonal to the corner piece
            if ((self.opposite_center_color_key[sticker_color_1] == center_color_white) and (self.opposite_center_color_key[sticker_color_2] == center_color_2)):
                solution.extend(["U2","L'","U'","L"])

            #Insertion slot clockwise position to the corner piece
            if ((self.opposite_center_color_key[sticker_color_1] == center_color_2) and (sticker_color_2 == center_color_white)):
                solution.extend(["U2","R","U'","R'"])

        return solution
    
    def generate_corner_solution_D_face(self) -> list[str]:
        """
        Generate the solution to solve the corner piece at the D face.
        """

        if (self.corner[1] == (0,0)):
            solution = ["L'","U","L"]
            self.new_cube.apply_solution(solution)
            solution.extend(self.generate_corner_solution_L_face())
        
        if (self.corner[1] == (0,2)):
            solution = ["R","U'","R'"]
            self.new_cube.apply_solution(solution)
            solution.extend(self.generate_corner_solution_R_face())

        if (self.corner[1] == (2,0)):
            solution = ["L'","U'","L"]
            self.new_cube.apply_solution(solution)
            solution.extend(self.generate_corner_solution_L_face())

        if (self.corner[1] == (2,2)):
            solution = ["R","U","R'"]
            self.new_cube.apply_solution(solution)
            solution.extend(self.generate_corner_solution_R_face())

        return solution
    
    def generate_corner_solution_B_face(self) -> list[str]:
        """
        Generate the solution to solve the corner piece at the B face.
        """

        solution = []

        #If the corner is in the bottom layer move it to the top
        if (self.corner[1] == (2,0)):
            solution = ["R'","U","R"]
        if (self.corner[1] == (2,2)):
            solution = ["L","U'","L'"]

        #Update the corner position if it was originally in the bottom layer
        self.corner = (self.corner[0],(0,self.corner[1][1])) if (len(solution) != 0) else self.corner

        self.new_cube.apply_solution(solution)

        corner_sticker_1, corner_sticker_2 = self.get_adjencent_sticker_positions(self.corner[0],self.corner[1])

        face_1 = corner_sticker_1[0]
        face_2 = corner_sticker_2[0]

        sticker_position_1 = corner_sticker_1[1]
        sticker_position_2 = corner_sticker_2[1]

        #Sticker on the top
        sticker_color_1 = self.new_cube.get_sticker(face_1,sticker_position_1)

        #Sticker/Center in the front/back
        sticker_color_2 = self.new_cube.get_sticker(face_2,sticker_position_2)
        center_color_2 = self.new_cube.get_center_color(face_2)

        #Center where the where sticker is
        center_color_white = self.new_cube.get_center_color(self.corner[0])

        if (self.corner[1] == (0,0)):

            #Insertion slot right above the corner piece
            if ((sticker_color_1 == center_color_white) and (sticker_color_2 == center_color_2)):
                solution.extend(["U'","R'","U","R"])

            #Insertion slot counterclockwise position to the corner piece
            if ((self.opposite_center_color_key[sticker_color_1] == center_color_2) and (sticker_color_2 == center_color_white)):
                solution.extend(["U'","L","U","L'"])

            #Insertion slot diagonal to the corner piece
            if ((self.opposite_center_color_key[sticker_color_1] == center_color_white) and (self.opposite_center_color_key[sticker_color_2] == center_color_2)):
                solution.extend(["L'","U2","L"])

            #Insertion slot clockwise position to the corner piece
            if ((sticker_color_1 == center_color_2) and (self.opposite_center_color_key[sticker_color_2] == center_color_white)):
                solution.extend(["U","R","U","R'"])

        if (self.corner[1] == (0,2)):

            #Insertion slot right above the corner piece
            if ((sticker_color_1 == center_color_white) and (sticker_color_2 == center_color_2)):
                solution.extend(["U","L","U'","L'"])

            #Insertion slot counterclockwise position to the corner piece
            if ((sticker_color_1 == center_color_2) and (self.opposite_center_color_key[sticker_color_2] == center_color_white)):
                solution.extend(["U'","L'","U'","L"])

            #Insertion slot diagonal to the corner piece
            if ((self.opposite_center_color_key[sticker_color_1] == center_color_white) and (self.opposite_center_color_key[sticker_color_2] == center_color_2)):
                solution.extend(["R","U2","R'"])

            #Insertion slot clockwise position to the corner piece
            if ((self.opposite_center_color_key[sticker_color_1] == center_color_2) and (sticker_color_2 == center_color_white)):
                solution.extend(["U","R'","U'","R"])

        return solution

    def generate_corner_solution_L_face(self) -> list[str]:
        """
        Generate the solution to solve the corner piece at the L face.
        """

        solution = []

        #If the corner is in the bottom layer move it to the top
        if (self.corner[1] == (2,0)):
            solution = ["L","U","L'","U'"]
        if (self.corner[1] == (2,2)):
            solution = ["L'","U","L"]

        #Update the corner position if it was originally in the bottom layer
        self.corner = (self.corner[0],(0,self.corner[1][1])) if (len(solution) != 0) else self.corner

        self.new_cube.apply_solution(solution)

        corner_sticker_1, corner_sticker_2 = self.get_adjencent_sticker_positions(self.corner[0],self.corner[1])

        face_1 = corner_sticker_1[0]
        face_2 = corner_sticker_2[0]

        sticker_position_1 = corner_sticker_1[1]
        sticker_position_2 = corner_sticker_2[1]

        #Sticker on the top
        sticker_color_1 = self.new_cube.get_sticker(face_1,sticker_position_1)

        #Sticker/Center in the front/back
        sticker_color_2 = self.new_cube.get_sticker(face_2,sticker_position_2)
        center_color_2 = self.new_cube.get_center_color(face_2)

        #Center where the where sticker is
        center_color_white = self.new_cube.get_center_color(self.corner[0])

        if (self.corner[1] == (0,0)):

            #Insertion slot right above the corner piece
            if ((sticker_color_1 == center_color_white) and (sticker_color_2 == center_color_2)):
                solution.extend(["L","U","L'"])

            #Insertion slot counterclockwise position to the corner piece
            if ((self.opposite_center_color_key[sticker_color_1] == center_color_2) and (sticker_color_2 == center_color_white)):
                solution.extend(["U2","L'","U","L"])

            #Insertion slot diagonal to the corner piece
            if ((self.opposite_center_color_key[sticker_color_1] == center_color_white) and (self.opposite_center_color_key[sticker_color_2] == center_color_2)):
                solution.extend(["U2","R","U","R'"])

            #Insertion slot clockwise position to the corner piece
            if ((sticker_color_1 == center_color_2) and (self.opposite_center_color_key[sticker_color_2] == center_color_white)):
                solution.extend(["R'","U","R"])

        if (self.corner[1] == (0,2)):

            #Insertion slot right above the corner piece
            if ((sticker_color_1 == center_color_white) and (sticker_color_2 == center_color_2)):
                solution.extend(["L'","U'","L"])

            #Insertion slot counterclockwise position to the corner piece
            if ((sticker_color_1 == center_color_2) and (self.opposite_center_color_key[sticker_color_2] == center_color_white)):
                solution.extend(["R","U'","R'"])

            #Insertion slot diagonal to the corner piece
            if ((self.opposite_center_color_key[sticker_color_1] == center_color_white) and (self.opposite_center_color_key[sticker_color_2] == center_color_2)):
                solution.extend(["U2","R'","U'","R"])

            #Insertion slot clockwise position to the corner piece
            if ((self.opposite_center_color_key[sticker_color_1] == center_color_2) and (sticker_color_2 == center_color_white)):
                solution.extend(["U2","L","U'","L'"])

        return solution

    def generate_corner_solution(self):
        """
        Generate the solution to solve the corner piece.
        """

        solution = []

        if (self.corner[0] == 'U'):
            solution = self.generate_corner_solution_U_face()

        if (self.corner[0] == 'F'):
            solution = self.generate_corner_solution_F_face()

        if (self.corner[0] == 'R'):
            solution = self.generate_corner_solution_R_face()

        if (self.corner[0] == 'D'):
            solution = self.generate_corner_solution_D_face()

        if (self.corner[0] == 'B'):
            solution = self.generate_corner_solution_B_face()

        if (self.corner[0] == 'L'):
            solution = self.generate_corner_solution_L_face()

        return solution

    def solve_corners(self):
        for _ in range(4):
            self.corner = self.locate_white_corner()
            if (self.corner == None):
                break
            solution = self.generate_corner_solution()
            self.apply_solution(solution)
            self.solutions.append(solution)
            new_cube = self.get_state()
            self.new_cube.set_state(new_cube)

        return new_cube, self.solutions

class second_layer_edge_solver(Cube):
    def __init__(self, cube_data):
        super().__init__(cube_data.get_state())

        self.edge_sticker_table = {
            ('U',(0,1)):('B',(0,1)),
            ('U',(1,0)):('L',(0,1)),
            ('U',(1,2)):('R',(0,1)),
            ('U',(2,1)):('F',(0,1)),

            ('F',(0,1)):('U',(2,1)),
            ('F',(1,0)):('L',(1,2)),
            ('F',(1,2)):('R',(1,0)),

            ('R',(0,1)):('U',(1,2)),
            ('R',(1,0)):('F',(1,2)),
            ('R',(1,2)):('B',(1,0)), 

            ('B',(0,1)):('U',(0,1)),
            ('B',(1,0)):('R',(1,2)),
            ('B',(1,2)):('L',(1,0)),

            ('L',(0,1)):('U',(1,0)),
            ('L',(1,0)):('B',(1,2)),
            ('L',(1,2)):('F',(1,0)),
        }

        self.edge_table = {
            "U":{
                #Middle Layer
                ("R",(0,1)):("F",(0,1)),
                ("F",(0,1)):("L",(0,1)),
                ("L",(0,1)):("B",(0,1)),
                ("B",(0,1)):("R",(0,1)),
                #Top Layer
                ("U",(0,1)):("U",(1,2)),
                ("U",(1,2)):("U",(2,1)),
                ("U",(2,1)):("U",(1,0)),
                ("U",(1,0)):("U",(0,1)),
            },
            "U'":{
                #Middle Layer
                ("R",(0,1)):("B",(0,1)),
                ("B",(0,1)):("L",(0,1)),
                ("L",(0,1)):("F",(0,1)),
                ("F",(0,1)):("R",(0,1)),
                #Top Layer
                ("U",(0,1)):("U",(1,0)),
                ("U",(1,0)):("U",(2,1)),
                ("U",(2,1)):("U",(1,2)),
                ("U",(1,2)):("U",(0,1)),
            },
            "U2":{
                #Middle Layer
                ("R",(0,1)):("L",(0,1)),
                ("L",(0,1)):("R",(0,1)),
                ("F",(0,1)):("B",(0,1)),
                ("B",(0,1)):("F",(0,1)),
                #Top Layer
                ("U",(0,1)):("U",(2,1)),
                ("U",(2,1)):("U",(0,1)),
                ("U",(1,0)):("U",(1,2)),
                ("U",(1,2)):("U",(1,0)),
            }
        }

        self.opposite_center_key = {
            "o":"r",
            "r":"o", 
            "b":"g",
            "g":"b",
        }

        self.solutions = [] 
        self.setup_move_length = 0

        self.new_cube = Cube(self.get_state())

    def locate_second_layer_edge(self) ->  list[tuple[str, tuple[int,int]]]:
        faces = ['F','R','L','B']
        edge_location = [(0,1),(1,0),(1,2)]

        def check_solved_edge(edge_sticker_1: tuple[str,tuple[int,int]], edge_sticker_2: tuple[str, tuple[int,int]]) -> bool:

            center_color_1 = self.new_cube.get_center_color(edge_sticker_1[0])
            center_color_2 = self.new_cube.get_center_color(edge_sticker_2[0])

            sticker_color_1 = self.new_cube.get_sticker(edge_sticker_1[0],edge_sticker_1[1])
            sticker_color_2 = self.new_cube.get_sticker(edge_sticker_2[0],edge_sticker_2[1])

            return center_color_1 == sticker_color_1 and center_color_2 == sticker_color_2

        for position in edge_location:
            for face in faces:
                edge_sticker_1 = (face,position)
                edge_sticker_2 = self.edge_sticker_table[(face,position)]
                if (self.new_cube.get_sticker(edge_sticker_1[0],edge_sticker_1[1]) not in ['w','y'] and self.new_cube.get_sticker(edge_sticker_2[0],edge_sticker_2[1]) not in ['w','y']):
                    if (check_solved_edge(edge_sticker_1,edge_sticker_2)):
                        continue
                    return [edge_sticker_1,edge_sticker_2]
            
        return []
    
    def generate_edge_solution(self) -> list[str]:

        edge = self.locate_second_layer_edge()

        if (edge == []):
            return []

        left_insert = ["U'","L'","U","L","U","y'","R","U'","R'"]
        right_insert = ["U","R","U'","R'","U'","y","L'","U","L"]  
        solution = []

        edge.sort(key=lambda x: x[0])

        def find_insertion_algorithm(edge: tuple[tuple[str,tuple[int,int]],tuple[str,tuple[int,int]]]) -> list[str]:
            algorithm = []
            edge_sticker_color = self.new_cube.get_sticker(edge[1][0],edge[1][1])
            
            F_face_center_color = self.new_cube.get_center_color('F')
            L_face_center_color = self.new_cube.get_center_color('L')

            if (edge[0][0] == "R"):
                algorithm = left_insert if (edge_sticker_color == F_face_center_color) else right_insert
            elif (edge[0][0] == "L"):
                algorithm = right_insert if (edge_sticker_color == F_face_center_color) else left_insert
            elif (edge[0][0] == "F"):
                algorithm = left_insert if (edge_sticker_color == L_face_center_color) else right_insert
            elif (edge[0][0] == "B"):
                algorithm = right_insert if (edge_sticker_color == L_face_center_color) else left_insert

            return algorithm
        
        def find_setup_move(edge: tuple[str,tuple[int,int]],sticker_color: str) -> list[str]:
            algorithm = []
            
            if (edge[0] == 'R'):
                if (self.new_cube.get_center_color('B') == sticker_color):
                    algorithm = ["U'"]
                elif (self.new_cube.get_center_color('F') == sticker_color):
                    algorithm = ["U"]
                else:
                    algorithm = ["U2"]

            elif (edge[0] == 'L'):
                if (self.new_cube.get_center_color('B') == sticker_color):
                    algorithm = ["U"]
                elif (self.new_cube.get_center_color('F') == sticker_color):
                    algorithm = ["U'"]
                else:
                    algorithm = ["U2"]

            elif (edge[0] == 'F'):
                if (self.new_cube.get_center_color('R') == sticker_color):
                    algorithm = ["U'"]
                elif (self.new_cube.get_center_color('L') == sticker_color):
                    algorithm = ["U"]
                else:
                    algorithm = ["U2"]

            elif (edge[0] == 'B'):
                if (self.new_cube.get_center_color('R') == sticker_color):
                    algorithm = ["U"]
                elif (self.new_cube.get_center_color('L') == sticker_color):
                    algorithm = ["U'"]
                else:
                    algorithm = ["U2"]

            return algorithm
            
        def update_edge(edge: tuple[tuple[str,tuple[int,int]],tuple[str,tuple[int,int]]],move:str) -> None:
            edge[0] = self.edge_table[move][edge[0]]
            edge[1] = self.edge_table[move][edge[1]]

        #Check if the edge is on the top layer or not
        if (edge[1][0] == "U"):
            center_sticker_color = self.new_cube.get_center_color(edge[0][0])
            sticker_color = self.new_cube.get_sticker(edge[0][0],edge[0][1])
            #Check if the edge is on the right side or not
            if (center_sticker_color != sticker_color):
                #Add the setup move to the solution
                solution.extend(find_setup_move(edge[0],sticker_color))
                #Update the edge position
                update_edge(edge,solution[0])
                #apply the setup move to the copy of the cube
                self.new_cube.apply_solution(solution)
            current_edge_face = edge[0][0]
            #Add the setup move to the solution)
            if (current_edge_face != "F"):
                solution.append("y" if current_edge_face == "R" else "y'" if current_edge_face == "L" else "y2")
            #Determine which insertion algorithm to use
            algorithm = find_insertion_algorithm(edge)
            solution.extend(algorithm)
        else:
            solution = ["y2"] if (edge[0][0] == "B" or edge[1][0] == "B") else []
            if (len(solution) == 0):
                solution.extend(right_insert if (edge[0][0] == "R" or edge[1][0] == "R") else left_insert)
            else:
                solution.extend(right_insert if (edge[0][0] == "L" or edge[1][0] == "L") else left_insert)

            self.new_cube.apply_solution(solution)
            
            solution.extend(self.generate_edge_solution())

        if (solution[0] == solution[1]):
            solution[:2] = ["U2"]

        if (solution[1] == "y" or solution[1] == "y'" or solution[1] == "y2"):
            if ((solution[0] == "U" and solution[2] == "U'") or (solution[0] == "U'" and solution[2] == "U")):
                solution[:3] = [solution[1]]
            if (solution[0] == "U2" and solution[2] == "U"):
                solution[:3] = ["U'",solution[1]]
            if (solution[0] == "U2" and solution[2] == "U'"):
                solution[:3] = ["U",solution[1]]

        return solution

    def solve_edge(self) -> list[list[str]]:
        for _ in range(4):
            solution = self.generate_edge_solution()
            if (solution == []): 
                break
            self.solutions.append(solution)
            self.apply_solution(solution)
            new_cube = self.get_state()
            self.set_state(new_cube)
            self.new_cube.set_state(new_cube)
    
        return new_cube, self.solutions

class last_layer_yellow_cross_solver(Cube):
    def __init__(self, cube_data):
        super().__init__(cube_data.get_state())

    def generate_yellow_cross_solution(self) -> list[str]:
        if (self.get_center_color("U") == 'y'):
            positions = np.array([(0,1), (1,0), (1,2), (2,1)])
            yellow_face = self.get_face("U")
            row, col = zip(*positions)
            yellow_edges = yellow_face[row,col] == 'y'
            number_of_yellow_edges = np.sum(yellow_edges)
            if number_of_yellow_edges == 4:
                return []
            elif number_of_yellow_edges == 2:
                if (yellow_edges[0] and yellow_edges[3]):
                    return ["U","F","R",'U',"R'","U'","F'"]
                elif (yellow_edges[1] and yellow_edges[2]):
                    return ["F","R",'U',"R'","U'","F'"]
                else:
                    if (yellow_edges[0] and yellow_edges[1]):
                        return ["F","R",'U',"R'","U'","R","U","R'","U'","F'"]
                    elif (yellow_edges[1] and yellow_edges[3]):
                        return ["U","F","R",'U',"R'","U'","R","U","R'","U'","F'"]
                    elif (yellow_edges[0] and yellow_edges[2]):
                        return ["U'","F","R",'U',"R'","U'","R","U","R'","U'","F'"]
                    elif (yellow_edges[2] and yellow_edges[3]):
                        return ["U2","F","R",'U',"R'","U'","R","U","R'","U'","F'"]
            elif number_of_yellow_edges == 0:
                return ["F","R","U","R'","U'","F'","U2","F","R","U","R'","U'","R","U","R'","U'","F'"]

    def solve_yellow_cross(self):
        solution = self.generate_yellow_cross_solution()
        self.apply_solution(solution)
        new_cube = self.get_state()
        self.set_state(new_cube)

        return new_cube, solution

class oll_solver(Cube):
    def __init__(self, cube_data):
        super().__init__(cube_data.get_state())
        self.faces = ["F", "R", "B", "L"]
        self.last_layer_cross_algorithm_table = {
            2:{
                "F R U' R' U' R U2 R' U' F'":np.array([
                    [0,0,0],
                    [0,0,0],
                    [1,0,0],
                    [0,0,1],
                ]),
                "R U R' U' L' U R U' L R'":np.array([
                    [1,0,0],
                    [0,0,0],
                    [0,0,1],
                    [0,0,0],
                ]),
                "R2 D R' U2 R D' R' U2 R'":np.array([
                    [1,0,1],
                    [0,0,0],
                    [0,0,0],
                    [0,0,0],
                ]),
            },
            3:{
                "R U R' U R U2 R'":np.array([
                    [0,0,1],
                    [0,0,1],
                    [0,0,1], 
                    [0,0,0],
                ]),
                "R U2 R' U' R U' R'":np.array([
                    [1,0,0],
                    [1,0,0],
                    [0,0,0],
                    [1,0,0],
                ]),
            },
            4:{
                "R U2 R' U' R U R' U' R U' R'":np.array([
                    [1,0,1],
                    [0,0,0],
                    [1,0,1],
                    [0,0,0],   
                ]),
                "R U2 R2 U' R2 U' R2 U2 R":np.array([
                    [0,0,1],
                    [0,0,0],
                    [1,0,0],
                    [1,0,1],
                ]),
            },
        }
        
    def identify_orentiation_case(self): 
        last_layer = []
        for face in self.faces:
            current_face = self.get_face(face)[0].copy()
            last_layer.append((current_face == 'y').astype(int))  # <-- fixed here
        last_layer = np.array(last_layer)
        num_unoriented_corners = np.sum(last_layer)

        if (num_unoriented_corners == 0):
            return []

        orientation_class = self.last_layer_cross_algorithm_table[num_unoriented_corners]         
        last_layer_case = []

        #Ensure that all the possible orientations of the last layer are checked
        for _ in range(4):
            last_layer_case.append(last_layer)
            last_layer = np.roll(last_layer,shift=-1,axis=0)

        for _ in range(4):
            for algorithm, case in orientation_class.items():
                if (np.all(case == last_layer_case[_])):
                    algorithm = algorithm.split()
                    if (_ == 0):
                        return algorithm
                    else:
                        algorithm.insert(0, "U" if _ == 1 else "U2" if _ == 2 else "U'")
                        return algorithm
                
    def solve_oll(self):
        solution = self.identify_orentiation_case()
        self.apply_solution(solution)
        new_cube = self.get_state()

        return new_cube, solution

class pll_solver(Cube):
    def __init__(self, cube_data):
        super().__init__(cube_data.get_state())
        self.faces = ["F", "R", "B", "L"]
        self.color_table = {
            "g":0,
            "o":1,
            "b":2,
            "r":3,
        }
        self.dataset = {
            "R2 U R U R' U' R' U' R' U R'": np.array([[0,3,0],[1,0,1],[2,2,2],[3,1,3]]),
            "R U' R U R U R U' R' U' R2": np.array([[0,1,0],[1,3,1],[2,2,2],[3,0,3]]),
            "M2' U M2' U2 M2' U M2'": np.array([[0,2,0],[1,3,1],[2,0,2],[1,3,1]]),
            "M2' U M2' U M' U2 M2' U2 M' U2": np.array([[0,1,0],[1,0,1],[2,3,2],[3,2,3]]),
            "M2' U' M2' U' M' U2 M2' U2' M' U2":np.array([[0,3,0],[1,2,1],[2,1,2],[3,0,3]])
        }

        self.setup_move = ["R","U2","R'","U'","R","U2","L'","U","R'","U'","L"]
        self.last_layer = []

        unique_cases_dict = self.generate_valid_transformations()

        self.cases, self.algorithms = [],[]

        for label, matrices in unique_cases_dict.items():
            for mat in matrices:
                self.cases.append(mat.flatten())  # Convert 4x3 matrix to 1D array
                self.algorithms.append(label)

        self.cases = np.array(self.cases)
        self.algorithms = np.array(self.algorithms)

        self.model = KDTree(self.cases, metric='euclidean')

    def generate_valid_transformations(self):
        unique_cases_dict = {}

        for case_label, original_matrix in self.dataset.items():
            cases = []

            original_matrix = original_matrix.copy()
            
            # Generate row shifts (U moves)
            for i in range(4):
                cases.append(original_matrix)
                for j in range(3):
                    original_matrix += 1
                    original_matrix %= 4
                    cases.append(original_matrix.copy())
                original_matrix += 1
                original_matrix %= 4
                original_matrix = np.roll(original_matrix, 1, axis=0)

            # Remove duplicates
            unique_cases = []
            seen = set()
            for case in cases:
                case_tuple = tuple(map(tuple, case))
                if case_tuple not in seen:
                    seen.add(case_tuple)
                    unique_cases.append(case)

            unique_cases_dict[case_label] = unique_cases

        return unique_cases_dict

    def get_last_layer(self):
        self.last_layer = np.array([self.get_face(face)[0] for face in self.faces])
        self.preprocess_last_layer()

    def preprocess_last_layer(self):
        for i in range(4):
            for j in range(3):
                self.last_layer[i][j] = self.color_table[self.last_layer[i][j]]

    def find_headlights(self):
        for _ in range(4):
            if (self.last_layer[_][0] == self.last_layer[_][2]):
                return _
        return -1
    
    def find_setup_move(self):
        headlight_position = self.find_headlights()
        if (headlight_position == -1):
            return self.setup_move + ["U2"] + self.setup_move
        elif (headlight_position == 0):
            return ["U"] + self.setup_move
        elif (headlight_position == 1):
            return ["U2"] + self.setup_move
        elif (headlight_position == 2):
            return ["U'"] + self.setup_move
        elif (headlight_position == 3):
            return self.setup_move
        return []
    
    def identify_permutation_case(self):
        algorithm = []
        self.get_last_layer()
        algorithm.extend(self.find_setup_move())
        return algorithm
    
    def reposition_last_layer(self):
        auf_table = {
            0:"U2",
            1:"U'",
            2:"",
            3:"U",
        }
        reposition_move = []
        mask = [int(len(np.unique(row)) == 1) for row in self.last_layer]
        if (1 in mask):
            idx = mask.index(1)
            reposition_move.append(auf_table[idx])
        else:
            new_cube = Cube(self.get_state())
            while True:
                sticker_color = new_cube.get_sticker("F",(0,0))
                center_color = new_cube.get_center_color("F")
                if (sticker_color != center_color):
                    break
                reposition_move.append("U")
                new_cube.apply_move("U")

            if (reposition_move.count("U") == 3):
                reposition_move = ["U'"]
            if (reposition_move.count("U") == 2):
                reposition_move = ["U2"]

        return reposition_move

    def find_final_auf(self):
        new_cube = Cube(self.get_state())
        auf = []
        while True:
            if (len(np.unique(new_cube.get_face("F").flatten())) == 1):
                break

            new_cube.apply_move("U")
            auf.append("U")
        
        if (auf.count("U") == 3):
            auf = ["U'"]
        if (auf.count("U") == 2):
            auf = ["U2"]

        return auf

    def classify_pll_kdtree(self):
        self.last_layer = self.last_layer.astype(int).flatten().reshape(1, -1)
        dist, ind = self.model.query(self.last_layer, k=1) 
        return self.algorithms[ind[0][0]].split(' ')

    def solve_pll(self):
        self.get_last_layer()
        setup_solution = self.identify_permutation_case()
        print(setup_solution)
        self.apply_solution(setup_solution)

        self.get_last_layer()
        reposition_move = self.reposition_last_layer()
        self.apply_solution(reposition_move)

        self.get_last_layer()
        algorithm = self.classify_pll_kdtree()
        self.apply_solution(algorithm)

        auf = self.find_final_auf()
        self.apply_solution(auf)

        new_cube = self.get_state()
        return new_cube, setup_solution

if __name__ == '__main__':
    cube_state = {
        'U': np.array([['y', 'y', 'y'], ['y', 'y', 'y'], ['y', 'y', 'y']]),
        'D': np.array([['w', 'w', 'w'], ['w', 'w', 'w'], ['w', 'w', 'w']]),
        'F': np.array([['g', 'g', 'g'], ['g', 'g', 'g'], ['g', 'g', 'g']]),
        'L': np.array([['r', 'r', 'r'], ['r', 'r', 'r'], ['r', 'r', 'r']]),
        'B': np.array([['b', 'b', 'b'], ['b', 'b', 'b'], ['b', 'b', 'b']]),
        'R': np.array([['o', 'o', 'o'], ['o', 'o', 'o'], ['o', 'o', 'o']])
    }

    cube = Cube(cube_state)

    scramble = "B U2 L2 U L F D F R F' L2 B' R2 B2 D2 B' U2 L2 F' R2 L'".split(' ')

    for move in scramble:
        cube.apply_move(move)

    cross_solver = cross_solver(cube)
    new_cube, cross_solution = cross_solver.solve_cross()
    cube.set_state(new_cube)

    print(cross_solution)

    corner_solver = corner_solver(cube)
    new_cube, corner_solution = corner_solver.solve_corners()
    cube.set_state(new_cube)

    print(*corner_solution,sep='\n')

    edge_solver = second_layer_edge_solver(cube)
    new_cube, edge_solution = edge_solver.solve_edge()
    cube.set_state(new_cube)

    print(*edge_solution,sep='\n')

    last_layer_yellow_cross_solver = last_layer_yellow_cross_solver(cube)
    new_cube, last_layer_yellow_cross_solution = last_layer_yellow_cross_solver.solve_yellow_cross()
    cube.set_state(new_cube)

    print(last_layer_yellow_cross_solution)

    oll_solver = oll_solver(cube)
    new_cube, oll_solution = oll_solver.solve_oll()
    cube.set_state(new_cube)

    print(oll_solution)

    pll_solver = pll_solver(cube)
    new_cube, pll_solution = pll_solver.solve_pll()
    cube.set_state(new_cube)

    cube.display_cube()




