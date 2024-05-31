## Giorgio Angelotti - 2024

import numpy as np
import trimesh
from tqdm import tqdm
import os
from PIL import Image
from copy import deepcopy
from scipy.spatial import KDTree
from plyfile import PlyData
import pandas as pd
Image.MAX_IMAGE_PIXELS = None
import blosc2
import subprocess


## Taken from ThaumatoAnakalyptort
def orient_uvs(vertices):
        # Rotate vertices and calculate the needed area
        vertices[:, 0] = 1.0 - vertices[:, 0]
        u_range = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
        v_range = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
        u_longer_v = u_range > v_range
        u_return = vertices[:, 0]
        v_return = vertices[:, 1]
        area_return = u_range * v_range
        for angle in range(-70, 70, 5):
            u_prime = vertices[:, 0] * np.cos(np.deg2rad(angle)) - vertices[:, 1] * np.sin(np.deg2rad(angle))
            v_prime = vertices[:, 0] * np.sin(np.deg2rad(angle)) + vertices[:, 1] * np.cos(np.deg2rad(angle))
            u_prime_range = np.max(u_prime) - np.min(u_prime)
            v_prime_range = np.max(v_prime) - np.min(v_prime)
            if u_prime_range < v_prime_range and u_longer_v:
                continue
            elif u_prime_range > v_prime_range and not u_longer_v:
                continue
            area = u_prime_range * v_prime_range
            if area < area_return:
                u_return = u_prime
                v_return = v_prime
                area_return = area

        return np.stack((u_return, v_return), axis=-1) 

'''
def extract_dimensions(header):
    # Convert header to a string if it's in bytes
    header_str = header.decode('utf-8') if isinstance(header, bytes) else header
    # Initialize default values
    width, height, dim = 0, 0, 0

    # Extract width
    width_match = re.search(r'width:\s*(\d+)', header_str)
    if width_match:
        width = int(width_match.group(1))

    # Extract height
    height_match = re.search(r'height:\s*(\d+)', header_str)
    if height_match:
        height = int(height_match.group(1))

    # Extract dim
    dim_match = re.search(r'dim:\s*(\d+)', header_str)
    if dim_match:
        dim = int(dim_match.group(1))

    return width, height, dim

def read_vcps(filename):
    with open(filename, 'rb') as file:
        header = b''
        while not header.endswith(b'<>\n'):
            header += file.read(1)

        # Extract dimensions from the header
        width, height, dim = extract_dimensions(header)
        
        # Read the binary data
        binary_data = file.read()

    # Convert binary data to a NumPy array
    data = np.frombuffer(binary_data, dtype=np.float64)

    # Reshape the array based on the dimensions
    # Each pixel has 'dim' values
    reshaped_data = data.reshape((height, width, dim))

    return reshaped_data

class Mesh:
    def __init__(self, num_points, num_cells):
        self.points = np.zeros((num_points, 3), dtype=np.float64)
        self.cells = np.zeros((num_cells, 3), dtype=np.int32)

    def set_points(self, points):
        self.points[:] = points.reshape(-1, 3)

    def to_open3d(self):
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(self.points)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(self.cells)
        return mesh_o3d

class OrderedPointSetMesher:
    def __init__(self, input_array, generate_triangles=False):
        self.input = input_array
        self.generate_triangles = generate_triangles
        height, width, _ = self.input.shape
        self.output = Mesh(height * width, 2 * (height - 1) * (width - 1) if generate_triangles else 0)

    def compute(self):
        if self.input.size == 0:
            raise ValueError("Attempted to mesh empty point set.")

        # Transfer vertex info directly
        self.output.set_points(self.input)

        if not self.generate_triangles:
            return self.output

        # Generate triangles using numpy operations
        height, width = self.input.shape[:2]
        idx_grid = np.arange(height * width).reshape(height, width)


        # 0  1  2
        # 3  4  5
        # 6  7  8

        # Triangle 1 (0, 1, 4)
        # Triangle 2 (0, 4, 3)

        # Indices for the top-left vertices of quads
        idx_tl = idx_grid[:-1, :-1].ravel()  # Top left

        # Right neighbor indices (for Triangle 1)
        idx_tr = idx_grid[:-1, 1:].ravel()  # Top right 

        # Bottom neighbor indices (for Triangle 2)
        idx_bl = idx_grid[1:, :-1].ravel()  # Bottom left

        # Diagonal neighbor (bottom right, shared by both triangles)
        idx_br = idx_grid[1:, 1:].ravel()  # Bottom right

        # Creating two triangles for each quad
        tri1 = np.vstack([idx_tl, idx_tr, idx_br]).T
        tri2 = np.vstack([idx_tl, idx_br, idx_bl]).T

        # Assign triangles
        self.output.cells[:tri1.shape[0]] = tri1
        self.output.cells[tri1.shape[0]:] = tri2

        return self.output
'''

def main(args):
        assert args.chunk >= 64, "Chunk size should be greater than 64."
        print(f"Loading mesh {args.segment_id}", end="\n")   
        obj_mesh = trimesh.load_mesh(os.path.join(args.work_dir, args.segment_id, f"{args.segment_id}.obj"))
        print(f"Loaded segment mesh {args.segment_id}", end="\n")                      
        V = np.asarray(obj_mesh.vertices)
        resolution = np.round(np.max(np.max(V,axis=0)-np.min(V,axis=0))).astype(int)
        print(f"Segment resolution for voxelizer: {resolution}", end="\n")
        try:
            with Image.open(os.path.join(args.work_dir, args.segment_id, f"{args.segment_id}.png")) as img:
                h,w = img.size
                print(f"Extracted dimensions {h}, {w}", end="\n")
        except:
            with Image.open(os.path.join(args.work_dir, args.segment_id, f"{args.segment_id}_mask.png")) as img:
                h,w = img.size
                print(f"Extracted dimensions {h}, {w}", end="\n")


        UV = np.asarray(obj_mesh.visual.uv)
        print(f"Loaded UV.", end="\n")

        UV[:,0] *= h
        UV[:,1] *= w
        UV = orient_uvs(UV)
        UV -= np.min(UV, axis=0)
        print(f"UV scaled and rotated.", end="\n")

        UV_color = deepcopy(UV[:,args.axis]/np.max(UV[:,args.axis]))
        print(f"Labels picked on UV axis: {args.axis}", end="\n")
        del UV

        # normalization to ensure that the coordinate 0 will be not mapped to 0 (which is assigned to void/air)
        UV_color *= 254
        UV_color += 1

        UV_color = UV_color.astype(np.uint8)
        print(f"Colors mapped to uint8.", end="\n")

        origin = np.floor(np.min(V,axis=0)).astype(int)
        np.savetxt(os.path.join(args.work_dir, args.segment_id, f"{args.segment_id}_origin.txt"), origin)
        print(f"Computed origin of frame: {origin}.", end="\n")


        shape = np.round(np.max(V,axis=0)-np.min(V,axis=0)).astype(int)
        tree = KDTree(V)
        print(f"Computed KDTree on vertices.", end="\n")

        CHUNK = (args.chunk, args.chunk, args.chunk)

        print("Voxelizing...", end="\n")
        subprocess.run([args.obj2voxel, os.path.join(args.work_dir, args.segment_id, f"{args.segment_id}.obj"), os.path.join(args.work_dir, args.segment_id, f"{args.segment_id}.ply"),
                        "-r", str(resolution)])
        print("Voxelized! Ply generated!", end="\n")

        print("Assigning labels...", end="\n")
        plydata = PlyData.read(os.path.join(args.work_dir, args.segment_id, f"{args.segment_id}.ply"), mmap=True)
        shape = tuple(shape)
        vertex_data = plydata['vertex'].data

        # Determine the number of vertices
        num_vertices = len(vertex_data)

        clevel = 9
        nthreads = args.workers
        cparams = {
                "codec": blosc2.Codec.ZSTD,
                "clevel": clevel,
                "filters": [blosc2.Filter.BITSHUFFLE, blosc2.Filter.BYTEDELTA],
                "filters_meta": [0, 0],
                "nthreads": nthreads,
            }


        b2 = blosc2.empty(
                shape, dtype=np.uint8, chunks=CHUNK, blocks=(64,64,64), urlpath=os.path.join(args.work_dir, args.segment_id, f"{args.segment_id}.b2nd"), cparams=cparams
            )
        
        CHUNK = np.array(CHUNK)

        for start in tqdm(range(0, num_vertices, args.batch_size), desc="Assigning label and compressing:"):
            end = min(start + args.batch_size, num_vertices)
                
            # Read a batch of data
            batch = vertex_data[start:end]

            # Convert list of tuples to DataFrame for easier grouping
            df = pd.DataFrame(batch.tolist(), columns=vertex_data.dtype.names)
            
            # Group by ijk // chunk_size
            df['chunk_x'] = df['x'] // CHUNK[0]
            df['chunk_y'] = df['y'] // CHUNK[1]
            df['chunk_z'] = df['z'] // CHUNK[2]

            groups = df.groupby(['chunk_x', 'chunk_y', 'chunk_z'])

            for name, group in groups:
                    chunk_x, chunk_y, chunk_z = name
                    #group = computed_groups.loc[name[:3]]
                    ind_voxels = group[['x', 'y', 'z']].to_numpy()

                    ind_voxels %= np.array(CHUNK)

                    occupied_voxels = group[['x', 'y', 'z']].to_numpy() + origin

                    _, indices = tree.query(occupied_voxels, k=1, workers=-1)
                    # Calculate the shape of the temporary chunk
                    temp_shape_x = min(CHUNK[0], shape[0] - chunk_x * CHUNK[0])
                    temp_shape_y = min(CHUNK[1], shape[1] - chunk_y * CHUNK[1])
                    temp_shape_z = min(CHUNK[2], shape[2] - chunk_z * CHUNK[2])
                    
                    # Initialize the temporary chunk
                    temp_chunk = np.zeros((temp_shape_x, temp_shape_y, temp_shape_z), dtype=np.uint8)
                        
                    # Clip the indices to ensure they are within the valid range of temp_chunk
                    clipped_x = np.clip(ind_voxels[:, 0], 0, temp_shape_x - 1)
                    clipped_y = np.clip(ind_voxels[:, 1], 0, temp_shape_y - 1)
                    clipped_z = np.clip(ind_voxels[:, 2], 0, temp_shape_z - 1)
                    
                    # Mark the occupied voxels in the temporary chunk
                    temp_chunk[clipped_x, clipped_y, clipped_z] = UV_color[indices]
                        
                    # Calculate the slice indices for the output grid
                    start_x = chunk_x * CHUNK[0]
                    end_x = min((chunk_x + 1) * CHUNK[0], shape[0])
                    start_y = chunk_y * CHUNK[1]
                    end_y = min((chunk_y + 1) * CHUNK[1], shape[1])
                    start_z = chunk_z * CHUNK[2]
                    end_z = min((chunk_z + 1) * CHUNK[2], shape[2])
                    

                    # Update the output grid
                    b2[start_x:end_x, start_y:end_y, start_z:end_z] += temp_chunk
                

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Voxelization and labeling of 3D mesh segments")
    parser.add_argument("--work_dir", type=str, required=True, help="Working directory containing segment folders")
    parser.add_argument("--segment_id", type=str, required=True, help="Segment ID to process")
    parser.add_argument("--chunk", type=int, default=256, help="Chunk size for compressing")
    parser.add_argument("--axis", type=int, default=0, help="UV axis to use for labeling")
    parser.add_argument("--workers", type=int, default=16, help="Workers for Blosc2")
    parser.add_argument("--batch_size", type=int, default=8000000, help="Batch size for processing vertices")
    parser.add_argument("--obj2voxel", type=str, required=True, help="Path to obj2voxel executable")
    args = parser.parse_args()

    main(args)




