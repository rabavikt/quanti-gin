import quanti_gin.data_generator
import tequila as tq
import numpy as np
import quanti_gin
import pandas as pd
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from quanti_gin.data_generator import DataGenerator
from quanti_gin.data_generator import Job
from quanti_gin.shared import minimum_weight_perfect_performance
from quanti_gin.shared import nearest_insertion
from quanti_gin.shared import generate_min_global_distance_edges
from quanti_gin.shared import brute_force
from quanti_gin.shared import simulated_annealing
from quanti_gin.shared import two_opt 
from quanti_gin.shared import genetic_algorithm
from quanti_gin.visualization_for_benchmarking import benchmarking_data_visualize_matplotlib



heueristics = {
    "blossom": minimum_weight_perfect_performance,
    "nearest_insertion": nearest_insertion,
    "nearest_neighbour": generate_min_global_distance_edges,
    "simulated annealing": simulated_annealing,
    "2-opt": two_opt,
    "genetic algorithm": genetic_algorithm,
    "brute force": brute_force
}


def visualize_molecule(coordinates: np.ndarray, geometry: str):
    """
    Simple 3D scatter plot of molecule structure.
    coordinates: Nx3 array of xyz positions.
    geometry: String for every atom of the form: f"h {coordinate[0]:f} {coordinate[1]:f} {coordinate[2]:f}\n"
    """
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')

    xs = [c[0] for c in coordinates]
    ys = [c[1] for c in coordinates]
    zs = [c[2] for c in coordinates]

    ax.scatter(xs, ys, zs, s=120, c="skyblue", edgecolors="k")

    for i, (x, y, z) in enumerate(coordinates):
        ax.text(x, y, z, geometry[i], fontsize=12, color="black")

    ax.set_title("Molecular Structure")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()


def run_benchmark(num_atoms, num_jobs):

    """
    Run benchmark tests comparing different heuristic methods for edge initialization in Separable Pair Approximation (SPA) optimization.

    This function generates random molecular geometries and evaluates the performance of various graph matching heuristics by comparing their resulting 
    energies against the Full Configuration Interaction (FCI) ground state energy.

    Parameters
    ----------
    num_atoms : int
        Number of hydrogen atoms in the molecule. Determines the molecular size 
        and complexity of the graph matching problem.
    num_jobs : int
        Number of random molecular geometries to generate and test. Each job 
        represents one molecular configuration with randomly positioned atoms 
        in 3D space.

    Returns
    -------
    None
        Results are saved directly to a CSV file named 'benchmark_results_{num_atoms}.csv' in the current directory.

    Notes
    -----
    The function tests all heuristics defined in the global `heueristics` dictionary.
    For each molecule and heuristic combination, it computes:
    
    - **runtime**: Time taken by the heuristic to generate edges (seconds)
    - **energy**: Final energy from SPA optimization using the heuristic's edges
    - **ground_state_energy**: Reference FCI ground state energy
    - **energy_gap**: Difference between SPA energy and ground state (energy - ground_state_energy)
    - **edges**: The edge configuration produced by the heuristic
    
    Different heuristics have different function signatures:
    
    - Heuristics requiring only coordinates: nearest neighbor, nearest insertion
    - Heuristics requiring num_atoms and coordinates: blossom, simulated annealing, 
      2-opt, genetic algorithm, brute force

    Output CSV Structure
    --------------------
    The output CSV contains one row per (molecule, heuristic) combination with columns:
    
    - method: Name of the heuristic
    - energy: SPA optimization result energy
    - edges: Edge configuration used
    - runtime: Execution time of the heuristic (seconds)
    - ground_state_energy: FCI ground state energy
    - energy_gap: Difference from ground state
    - error: Error message if the heuristic failed (otherwise absent)
    """

    jobs = DataGenerator.generate_jobs(number_of_jobs=num_jobs, number_of_atoms=num_atoms)
    results = []

    # Optional: Visualize first molecule 

    #first_job = jobs[0]
    #visualize_molecule(first_job.coordinates, first_job.geometry)

    for job in tqdm(jobs):
        coordinates = job.coordinates
        geometry = job.geometry
        mol = tq.Molecule(geometry=geometry, basis_set="sto-3g")
        
        for name, function in heueristics.items():
            try:
                if(name in ("blossom", "simulated annealing", "2-opt", "genetic algorithm", "brute force")):
                    start_time = time.time()
                    edges = function(num_atoms, coordinates)

                    runtime = time.time() - start_time
                else:
                    start_time = time.time()
                    edges = function(coordinates)

                    runtime = time.time() - start_time

                result = DataGenerator.run_spa_optimization(molecule=mol, coordinates=coordinates, edges=edges)

                #_, ground_state_energy = DataGenerator.get_ground_states(molecule = mol)
                #ground_state_energy = DataGenerator.run_fci_optimization(molecule=mol)
                ground_state_energy = mol.compute_energy("fci")
                energy = result["energy"]

                energy_gab = energy - ground_state_energy

                results.append({
                    "method": name,
                    "energy": energy,
                    "edges": edges,
                    "runtime": runtime,
                    "ground state energy": ground_state_energy,
                    "energy gab": energy_gab
                })

            except Exception as e:

                results.append({
                    "method": name,
                    "energy": None,
                    "edges": None,
                    "runtime": None,
                    "ground state energy": None,
                    "energy gab": None,
                    "error": str(e)
                })

    data = pd.DataFrame(results)
    data.to_csv(f"benchmark_results_{num_atoms}.csv", index = False)

def run_benchmark_for_linear_molecules(num_atoms: int, num_jobs: int, axis="x", base_spacing=0.25):

    """
    Run benchmark tests on linear molecular geometries with varying atom spacing.

    This function generates linear (chain-like) molecular geometries along a specified axis and evaluates the performance of various graph matching heuristics by comparing 
    their resulting energies against the Full Configuration Interaction (FCI) ground state energy.

    Parameters
    ----------
    num_atoms : int
        Number of hydrogen atoms in the linear molecule. Determines the chain length and complexity of the graph matching problem.
    num_jobs : int
        Number of linear molecular geometries to generate with varying spacing. 
        Each job uses a slightly different spacing between atoms to test robustness 
        across different bond lengths.
    axis : {"x", "y", "z"}, default="x"
        Cartesian axis along which atoms are arranged linearly.
        - "x": Atoms aligned along x-axis (y=0, z=0)
        - "y": Atoms aligned along y-axis (x=0, z=0)  
        - "z": Atoms aligned along z-axis (x=0, y=0)
    base_spacing : float, default=0.25
        Base distance (in Angstroms) between consecutive atoms. The actual spacing 
        for each job is calculated as: base_spacing + 0.05 * job_number, creating 
        a range of bond lengths from base_spacing to base_spacing + 0.05 * num_jobs.

    Returns
    -------
    None
        Results are saved directly to a CSV file named 'benchmark_results_{num_atoms}_line_{axis}.csv' in the current directory.
    
    **Spacing Progression:**
    For each job i (1 to num_jobs), the spacing is: base_spacing + 0.05 * i
    
    Example with base_spacing=0.25 and num_jobs=3:
    - Job 1: spacing = 0.30 Å
    - Job 2: spacing = 0.35 Å  
    - Job 3: spacing = 0.40 Å

    Output CSV Structure
    --------------------
    The output CSV contains one row per (molecule, heuristic) combination with columns:
    
    - method: Name of the heuristic
    - energy: SPA optimization result energy
    - edges: Edge configuration used
    - runtime: Execution time of the heuristic (seconds)
    - ground_state_energy: FCI ground state energy
    - energy_gap: Difference from ground state
    - error: Error message if the heuristic failed (otherwise absent)
    """

    assert axis in ("x", "y", "z") #Axis must be 'x', 'y', or 'z'

    results = []

    for i in tqdm(range(1, num_jobs + 1)):
        # Vary spacing slightly for each job
        spacing = base_spacing + 0.05 * i
        print(spacing)
        coordinates = []
        for j in range(num_atoms):
            if(axis == "x"):
                coord = [j * spacing, 0.0, 0.0]
            elif(axis == "y"):
                coord = [0.0, j * spacing, 0.0]
            else:
                coord = [0.0, 0.0, j * spacing]
            coordinates.append(coord)

        coordinates = np.array(coordinates)
        geometry = DataGenerator.generate_geometry_string(coordinates)

        #visualize_molecule(coordinates, geometry)

        mol = tq.Molecule(geometry=geometry, basis_set="sto-3g")

        for name, function in heueristics.items():
            try:
                if(name in ("blossom", "simulated annealing", "2-opt", "genetic algorithm", "brute force")):
                    start_time = time.time()
                    edges = function(num_atoms, coordinates)
                    runtime = time.time() - start_time

                else:
                    start_time = time.time()
                    edges = function(coordinates)
                    runtime = time.time() - start_time

                result = DataGenerator.run_spa_optimization(
                    molecule=mol,
                    coordinates=coordinates,
                    edges=edges
                )

                ground_state_energy = mol.compute_energy("fci")
                energy = result["energy"]

                energy_gab = energy - ground_state_energy

                results.append({
                    "method": name,
                    "energy": energy,
                    "edges": edges,
                    "runtime": runtime,
                    "ground state energy": ground_state_energy,
                    "energy gab": energy_gab
                })

            except Exception as e:
                results.append({
                    "method": name,
                    "energy": None,
                    "edges": None,
                    "runtime": None,
                    "ground state energy": None,
                    "energy gab": None,
                    "error": str(e)
                })

    data = pd.DataFrame(results)
    data.to_csv(f"benchmark_results_{num_atoms}_line_{axis}.csv", index=False)

def run_benchmark_for_ring_molecules(num_atoms, num_jobs, radius = 1, radius_increase = 0.25):

    """
    Run benchmark tests on ring/cyclic molecular geometries with varying ring sizes.

    This function generates ring-shaped (cyclic) molecular geometries where atoms are 
    arranged in a circle on the xy-plane and evaluates the performance of heuristics by comparing their resulting energies against the Full 
    Configuration Interaction (FCI) ground state energy. This is useful for testing heuristics on symmetric, cyclic structures.

    Parameters
    ----------
    num_atoms : int
        Number of hydrogen atoms in the ring molecule.
    num_jobs : int
        Number of ring molecular geometries to generate with varying radius. 
        Each job uses a progressively larger ring radius to test heuristic performance across different ring sizes and atomic separations.
    radius : float, default=1.0
        Initial radius (in Angstroms) of the ring for the first job. 
        Atoms are positioned at equal angular intervals on a circle of this radius in the xy-plane.
    radius_increase : float, default=0.25
        Amount (in Angstroms) by which the radius increases for each subsequent job.
        The radius for job i is: radius + (i - 1) * radius_increase

    Returns
    -------
    None
        Results are saved directly to a CSV file named 
        'benchmark_results_{num_atoms}_ring.csv' in the current directory.
    
    **Atomic Positioning:**
    For n atoms on a circle of radius r, atom i is placed at:
    - angle_i = 2π * i / n  (evenly distributed around the circle)
    - x_i = r * cos(angle_i)
    - y_i = r * sin(angle_i)
    - z_i = 0.0
    
    **Radius Progression:**
    For each job i (1 to num_jobs), the radius is: radius + (i - 1) * radius_increase
    
    Example with radius=1.0, radius_increase=0.25, and num_jobs=4:
    - Job 1: radius = 1.00 Å
    - Job 2: radius = 1.25 Å
    - Job 3: radius = 1.50 Å
    - Job 4: radius = 1.75 Å
    
    **Heuristic Evaluation:**
    The function tests all heuristics defined in the global `heueristics` dictionary.
    For each molecule and heuristic combination, it computes:
    
    - **runtime**: Time taken by the heuristic to generate edges (seconds)
    - **energy**: Final energy from SPA optimization using the heuristic's edges
    - **ground_state_energy**: Reference FCI ground state energy
    - **energy_gap**: Difference between SPA energy and ground state (energy - ground_state_energy)
    - **edges**: The edge configuration produced by the heuristic
    
    **Function Signatures:**
    Different heuristics have different input requirements:
    
    - Coordinates only: nearest neighbor, nearest insertion
    - num_atoms + coordinates: blossom, simulated annealing, 2-opt, 
      genetic algorithm, brute force

    Output CSV Structure
    --------------------
    The output CSV contains one row per (molecule, heuristic) combination with columns:
    
    - method: Name of the heuristic
    - energy: SPA optimization result energy
    - edges: Edge configuration used
    - runtime: Execution time of the heuristic (seconds)
    - ground_state_energy: FCI ground state energy
    - energy_gap: Difference from ground state
    - error: Error message if the heuristic failed (otherwise absent)

    """

    jobs = DataGenerator.generate_jobs(number_of_jobs=num_jobs, number_of_atoms=num_atoms)
    results = []

    for job in tqdm(jobs):

        angles = np.linspace(0, 2 * np.pi, num_atoms, endpoint=False)
        #angles = np.sort(np.random.uniform(0, 2 * np.pi, num_atoms))
        coordinates = np.array([[radius * np.cos(angle), radius * np.sin(angle), 0.0] for angle in angles])
      
        #print(radius)
        radius =  radius + radius_increase

        formatted_lines = []
        for coord in coordinates:
            x, y, z = coord
            line = f"h {x:.6f} {y:.6f} {z:.6f}"
            formatted_lines.append(line)
        geometry = "\n".join(formatted_lines)

        #visualize_molecule(coordinates, geometry)

        mol = tq.Molecule(geometry=geometry, basis_set="sto-3g")
        
        for name, function in heueristics.items():
            try:
                if(name in ("blossom", "simulated annealing", "2-opt", "genetic algorithm", "brute force")):
                    start_time = time.time()
                    edges = function(num_atoms, coordinates)

                    runtime = time.time() - start_time
                else:
                    start_time = time.time()
                    edges = function(coordinates)

                    runtime = time.time() - start_time

                result = DataGenerator.run_spa_optimization(molecule=mol, coordinates=coordinates, edges=edges)

                ground_state_energy = mol.compute_energy("fci")
                energy = result["energy"]

                energy_gap = energy - ground_state_energy

                results.append({
                    "method": name,
                    "energy": energy,
                    "edges": edges,
                    "runtime": runtime,
                    "ground state energy": ground_state_energy,
                    "energy gab": energy_gap
                })

            except Exception as e:

                results.append({
                    "method": name,
                    "energy": None,
                    "edges": None,
                    "runtime": None,
                    "ground state energy": None,
                    "energy gab": None,
                    "error": str(e)
                })

    data = pd.DataFrame(results)
    
    data.to_csv(f"benchmark_results_{num_atoms}_ring.csv", index=False)

if __name__ == "__main__":  

    run_benchmark(num_atoms=6, num_jobs=200)

    run_benchmark_for_linear_molecules(num_atoms=4, num_jobs=200, axis="x")
   
    run_benchmark_for_ring_molecules(num_atoms=10, num_jobs=200, radius=1)

    benchmarking_data_visualize_matplotlib("benchmark_results_4_ring.csv", 
                                                methods_to_plot=["blossom", "nearest_insertion", "nearest_neighbour", "simulated annealing", "2-opt", "genetic algorithm", "brute force"],
                                                show_first_n_molecules=55)
