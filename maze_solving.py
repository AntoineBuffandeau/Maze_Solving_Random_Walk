import tkinter as tk 
import numpy as np
import random
import time
from scipy.ndimage import label

# Labyrinthe -> création de la grille avec des chemins selon un choix de complexité et de densité choisi 
# (afin de choisir un temps de calcul abordable pour la prés - à ne pas changer svp !)
def create_maze(size, complexity=0.75, density=0.75):
    shape = (size, size) #taille carré
    maze = np.zeros(shape, dtype=bool) #initialisation grille vide
    maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = True #ajout de murs sur le bords du maze

    #Nombre de bloc (mur/obstacle pour mur et impasse)
    complexity = int(complexity * (5 * (shape[0] + shape[1]))) #adaptation int à la taille du maze
    density = int(density * ((shape[0] // 2) * (shape[1] // 2))) #pareil mais pr densité := nb obstacles

    for _ in range(density):
        x, y = random.randint(0, shape[0] // 2) * 2, random.randint(0, shape[1] // 2) * 2 #choix d'un pts de dapért aléa
        maze[x, y] = True #placement mur à cet endroit
        #on construit ensuite les murs autour de ce pts
        for _ in range(complexity):
            #ajout voisisns si cela est possible
            neighbors = []
            if x > 1: neighbors.append((x - 2, y))
            if x < shape[0] - 2: neighbors.append((x + 2, y))
            if y > 1: neighbors.append((x, y - 2))
            if y < shape[1] - 2: neighbors.append((x, y + 2))
            if neighbors:
                nx, ny = neighbors[random.randint(0, len(neighbors) - 1)]
                if not maze[nx, ny]: #si pas un mur
                    maze[nx, ny] = True #=> devient mur
                    maze[nx + (x - nx) // 2, ny + (y - ny) // 2] = True #création du chemin
                    x, y = nx, ny

    # Convertir le labyrinthe en format utilisable
    maze_array = np.ones(shape, dtype=int)
    maze_array[maze] = 0
    maze_array[1, 1] = 1  #Départ - entréee grille
    maze_array[-2, -2] = 2  #Arrivée ("objectif" dans article) - sortie grille

    return maze_array

def get_possible_moves(maze, position):
    moves = []
    x, y = position #coord actuelle de la position
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]: #parcours de ttes les directions possibles
        if maze[x + dx, y + dy] > 0: #si case voisine accessible
            moves.append((x + dx, y + dy)) #ajout case à la liste des mvts possibles
    return moves

#Fonction pour effectuer une marche aléatoire
def classical_random_walk(maze, start, canvas, cell_size, draw_delay=0.005):
    position = start
    path = [position] 

    while maze[position] != 2:  #on continue jusqu'à atteindre l'arrivée (voir ligne 40 si besoin Titouan)
        moves = get_possible_moves(maze, position)

        if not moves:  #si aucun mvt possible
            return path

        position = random.choice(moves) #on choisit aléa un des mvt possibles
        path.append(position) #ajoute la poition au chemin parcouru

        draw_cell(canvas, position, cell_size, "blue", maze)
        time.sleep(draw_delay) #ajout delais

    return path

#Fonction pour effectuer une marche aléatoire sans retour (sans revenir en arrière)
def classical_no_backtrack(maze, start, canvas, cell_size, draw_delay=0.2):
    position = start
    path = [position]
    visited = set()
    branches = []  #Pour gérer les bifurcations

    while maze[position] != 2:  #on continue jusqu'à atteindre l'arrivée
        moves = get_possible_moves(maze, position)
        moves = [move for move in moves if move not in visited] #exclu mvt déjà visité

        if not moves:  #si aucun mvt à visiter
            if not branches:  #si aucun chemin à explorer
                draw_cell(canvas, position, cell_size, "red", maze)  #pts colorié en rouge => indique impasse
                return path
            
            position, new_path = branches.pop() #on revient à la dernière bifurcation mémorisée
            path = path[:new_path]
        else: #si mvt possible
            if len(moves) > 1:  #si bifurcation, mémoriser le chemin
                branches.append((position, len(path)))

            position = moves.pop()
            path.append(position) #ajoute la nouvelle position au chemin parcouru
            visited.add(position) #pareil à l'ensemble des positions visitées

            draw_cell(canvas, position, cell_size, "blue", maze)
            time.sleep(draw_delay)

    return path

#Fonction simulant une marche aléatoire quantique
def quantum(maze, start, canvas, cell_size, iterations=100, draw_delay=0.2):
    size = maze.shape[0]
    active_points = {start: 1.0}  #points actifs et leurs probabilités
    # Rq => au départ, seule la position de départ est active avec une probabilité de 1

    for _ in range(iterations):
        new_points = {} #dic temporaire

        for position, probability in active_points.items(): #parcours de tt les pts et proba
            moves = get_possible_moves(maze, position)

            if not moves:  #si aucun mvt possible => pts bloqué
                draw_cell(canvas, position, cell_size, "red", maze) 
                continue

            split_prob = probability / len(moves) #proba divisée entre tt les mvts possibles

            for move in moves: #répartition de proba sur chaque mvt
                if move in new_points: #s'il existe on ajoute la proba
                    new_points[move] += split_prob 
                else: #sinon initialisation avec proba calculée
                    new_points[move] = split_prob 

        active_points = new_points
        canvas.delete("active")

        for pos in active_points.keys():
            draw_cell(canvas, pos, cell_size, "yellow", maze, tag="active")  #dessine les points actifs en jaune

        time.sleep(draw_delay)

        for pos in active_points:
            if maze[pos] == 2: #si position = objectif
                draw_cell(canvas, pos, cell_size, "green", maze)  #colorier en vert
                return

    print("Le marcheur quantique n'a pas trouvé la sortie dans la limite des itérations.")

#Fonction simulant une marche aléatoire quantique sans retour 
#seules les lignes commentées ont été modif par rapport à l'autre fct
def quantum_no_backtrack(maze, start, canvas, cell_size, iterations=100, draw_delay=0.2):
    size = maze.shape[0]
    active_points = {start: 1.0} 
    visited_points = set() #modif ici -> ensemble des positions déjà visitées

    for _ in range(iterations):
        new_points = {}

        for position, probability in active_points.items():
            moves = get_possible_moves(maze, position)
            moves = [move for move in moves if move not in visited_points] #exclu les mvts pr les pts qui ont déjà été visités.

            if not moves:
                draw_cell(canvas, position, cell_size, "red", maze) 
                continue

            split_prob = probability / len(moves)

            for move in moves:
                if move in new_points:
                    new_points[move] += split_prob
                else:
                    new_points[move] = split_prob

            visited_points.add(position) #ajout de la position actuelle à l'ensemble des positions visitées

        active_points = new_points

        canvas.delete("active")

        for pos in active_points.keys():
            draw_cell(canvas, pos, cell_size, "yellow", maze, tag="active") 

        time.sleep(draw_delay)

        for pos in active_points:
            if maze[pos] == 2:
                draw_cell(canvas, pos, cell_size, "green", maze)
                return

    print("Le marcheur quantique n'a pas trouvé la sortie dans la limite des itérations.")

def draw_cell(canvas, position, cell_size, color, maze, tag="path"):
    x, y = position
    if maze[x, y] > 0:  # Empêcher de dessiner sur un mur
        canvas.create_rectangle(
            y * cell_size,
            x * cell_size,
            (y + 1) * cell_size,
            (x + 1) * cell_size,
            fill=color,
            outline=color,
            tags=tag
        )
        canvas.update()

class MazeApp:
    def __init__(self, root, maze_size=25, cell_size=30):
        self.root = root
        self.maze_size = maze_size
        self.cell_size = cell_size
        self.maze = create_maze(maze_size)
        self.start = (1, 1)
        self.classical_time = 0
        self.quantum_time = 0
        self.running = False

        self.frame = tk.Frame(root)
        self.frame.pack()

        self.canvas = tk.Canvas(self.frame, width=maze_size * cell_size, height=maze_size * cell_size)
        self.canvas.grid(row=0, column=0)

        self.info_text = tk.Text(self.frame, width=30, height=15)
        self.info_text.grid(row=0, column=1, padx=10)

        self.draw_maze()
        self.buttons()

    def draw_maze(self):
        self.canvas.delete("all")
        for x in range(self.maze_size):
            for y in range(self.maze_size):
                color = "white" if self.maze[x, y] == 1 else "black"
                if self.maze[x, y] == 2:
                    color = "green"
                self.canvas.create_rectangle(
                    y * self.cell_size,
                    x * self.cell_size,
                    (y + 1) * self.cell_size,
                    (x + 1) * self.cell_size,
                    fill=color,
                    outline="gray",
                )
        self.canvas.create_rectangle(
            self.start[1] * self.cell_size,
            self.start[0] * self.cell_size,
            (self.start[1] + 1) * self.cell_size,
            (self.start[0] + 1) * self.cell_size,
            fill="orange",
            outline="gray",
        )

    def buttons(self):
        btn_classical = tk.Button(self.root, text="Classique", command=self.run_classical)
        btn_classical.pack(side=tk.LEFT, padx=10)
        btn_quantum = tk.Button(self.root, text="Quantique", command=self.run_quantum)
        btn_quantum.pack(side=tk.LEFT, padx=10)
        btn_classical_no_backtrack = tk.Button(self.root, text="Classique sans retour", command=self.run_classical_no_backtrack)
        btn_classical_no_backtrack.pack(side=tk.LEFT, padx=10)
        btn_quantum_no_backtrack = tk.Button(self.root, text="Quantique sans retour", command=self.run_quantum_no_backtrack)
        btn_quantum_no_backtrack.pack(side=tk.LEFT, padx=10)
        btn_reset = tk.Button(self.root, text="Réinitialiser", command=self.reset)
        btn_reset.pack(side=tk.LEFT, padx=10)

    def run_classical(self):
        if self.running:
            return
        self.running = True
        self.canvas.delete("path")
        self.canvas.delete("active")
        start_time = time.time()
        classical_path = classical_random_walk(self.maze, self.start, self.canvas, self.cell_size)
        self.classical_time = time.time() - start_time
        self.info_text.insert(tk.END, f"Classique : {self.classical_time:.2f}s, délais : 0.005s\n")
        self.running = False

    def run_quantum(self):
        if self.running:
            return
        self.running = True
        self.canvas.delete("path")
        self.canvas.delete("active")
        start_time = time.time()
        quantum(self.maze, self.start, self.canvas, self.cell_size)
        self.quantum_time = time.time() - start_time
        self.info_text.insert(tk.END, f"Quantique : {self.quantum_time:.2f}s, délais : 0.2s\n")
        self.running = False

    def run_classical_no_backtrack(self):
        if self.running:
            return
        self.running = True
        self.canvas.delete("path")
        self.canvas.delete("active")
        start_time = time.time()
        classical_no_backtrack(self.maze, self.start, self.canvas, self.cell_size)
        self.classical_time = time.time() - start_time
        self.info_text.insert(tk.END, f"Classique sans retour : {self.classical_time:.2f}s, délais : 0.2s\n")
        self.running = False

    def run_quantum_no_backtrack(self):
        if self.running:
            return
        self.running = True
        self.canvas.delete("path")
        self.canvas.delete("active")
        start_time = time.time()
        quantum_no_backtrack(self.maze, self.start, self.canvas, self.cell_size)
        self.quantum_time = time.time() - start_time
        self.info_text.insert(tk.END, f"Quantique sans retour : {self.quantum_time:.2f}s, délais : 0.2s\n")
        self.running = False

    def reset(self):
        self.running = False
        self.maze = create_maze(self.maze_size)
        self.draw_maze()
        self.classical_time = 0
        self.quantum_time = 0
        self.info_text.delete(1.0, tk.END)
        self.canvas.delete("path")
        self.canvas.delete("active")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Labyrinthe Classique vs Quantique")
    app = MazeApp(root)
    root.mainloop()
