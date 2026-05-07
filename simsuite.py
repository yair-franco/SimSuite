"""
SimSuite - Python module for ground motion analysis with GMMs and SeisSol

Yair Franco, 2026
"""

import re
import warnings
import time

import numpy as np
import pandas as pd
import pyvista as pv
import vtk

class Distances:
    """Calculators for earthquake-site distance metrics needed for GMMs

    **These calculations assume the (n,e,z) point (0,0,0) is the HYPOCENTER.**

    Note this is different to some conventions where the origin is the center of the
    top edge of the rupture.

    Algorithms from Baker et al. 'Seismic Hazard and Risk Analysis' (2021) Chapter 3
    
    Attr:
        strike: fault strike [deg]
        dip: fault dip [deg]
        fault_length: [km]
        fault_width: [km]
        hyp_depth: hypocentral depth [km]
        hyp_along_strike: distance of hypocenter along strike [km]
        ztor: depth of top edge of rupture [km]

    Args for calc_ functions:
        points: site where GMM is calculated [(n,3) array of (x,y,z) coords]
        convert_xyz: whether to convert xyz coords to the necessary neu coord format [bool, default: True]

    """
    def __init__(
            self,
            strike=None,
            dip=None,
            fault_length=None,
            fault_width=None,
            hyp_depth=None,
            hyp_along_strike=None,
            ztor=None):
        self.strike = strike
        self.dip = dip
        self.fault_length = fault_length
        self.fault_width = fault_width
        self.hyp_depth = hyp_depth
        self.hyp_along_strike = hyp_along_strike
        self.ztor = ztor

    def _require(self,*names):
        missing = [n for n in names if getattr(self, n) is None]
        if missing:
            raise ValueError(f"Missing required properties: {', '.join(missing)}")
        
    def frac_hyp_coords(self):
        """Fractional coordinates of hypocenter along fault, needed as a reference point to rotate point using matrices
        """
        self._require('dip','hyp_depth','ztor','fault_length','fault_width','hyp_along_strike')
        self.x_L = self.hyp_along_strike / self.fault_length
        self.x_W = (self.hyp_depth - self.ztor) / (self.fault_width * np.sin(np.radians(self.dip)))

        if self.x_W > 1:
            warnings.warn("x_W is not > 1; your hypocentral depth may not be within the fault plane.", UserWarning)

    def points_nez(self,points):
        """Swaps x and y so converted coordinates are (n,e,z)
        Otherwise coordinates are incorrectly calculated as (e,n,z)
        Also compensates for hypocentral depth in z-coordinate,
        which is 0 at the hypocenter

        Input: (N,3) array of xyz points
        Output: (N,3) array of nez points with corrected z-coords
        """
        point_arr = np.copy(points)
        point_arr[:,[0,1]] = point_arr[:,[1,0]]
        point_arr[:,[2]] = point_arr[:,[2]] - self.hyp_depth
        return point_arr

    def point_uvw(self,points,convert_xyz=True):
        """Converts point in (x,y,z) (or (e,n,z)) to fault-relative coordinates (u,v,w)
        Dip-dependent, needed for Rrup
        """
        self._require('strike','dip')
        theta = np.radians(self.strike)
        delta = np.radians(self.dip)

        TdTth = np.array(
            [[np.cos(theta),               np.sin(theta),               0],
            [-np.sin(theta)*np.cos(delta), np.cos(theta)*np.cos(delta), np.sin(delta)],
            [np.sin(theta)*np.sin(delta), -np.cos(theta)*np.sin(delta), np.cos(delta)]]
        )
        
        if convert_xyz:
            points = self.points_nez(points)
        
        self.uvw = np.dot(TdTth, points.T).T

    def point_uvPrimew(self,points,convert_xyz=True):
        """Converts point in (x,y,z) (or (e,n,z)) to fault-relative coordinates (u,v,w)
        Not dependent on dip, used for Rjb and Rx
        """
        self._require('strike')
        theta = np.radians(self.strike)

        Tth = np.array(
            [[np.cos(theta),  np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0,              0,             1]]
        )

        if convert_xyz:
            points = self.points_nez(points)

        self.uvPrimew = np.dot(Tth, points.T).T

    def calc_Rrup(self, points,convert_xyz=True):
        """Distance to rupture plane
        
        Input: (n,3) array of xyz coords (centered at the surface)


        Set convert_xyz to False if coordinates are already in (n,e,z) format
        Otherwise, (x,y,z) is automatically converted to (n,e,z) instead of the incorrect (e,n,z)
        """
        self._require('fault_length', 'fault_width')
        self.frac_hyp_coords()
        self.point_uvw(points,convert_xyz)

        u, v, w = self.uvw.T
        x_L, x_W = self.x_L, self.x_W
        L = self.fault_length
        W = self.fault_width

        u_c = np.maximum(np.minimum(u, (1-x_L) * L), -x_L * L)
        v_c = np.maximum(np.minimum(v, (1-x_W) * W), -x_W * W)

        Rrup = np.sqrt(((u - u_c) ** 2) + ((v - v_c) ** 2) + (w ** 2))

        return Rrup

    def calc_Rjb(self, points,convert_xyz=True):
        """Distance to surface fault projection (0 at projection)

        Input: (n,3) array of xyz coords (centered at the surface)

        Set convert_xyz to False if coordinates are already in (n,e,z) format
        Otherwise, (x,y,z) is automatically converted to (n,e,z) instead of the incorrect (e,n,z)
        """
        self._require('dip', 'fault_length', 'fault_width')
        self.frac_hyp_coords()
        self.point_uvPrimew(points,convert_xyz)  # Transform all points at once

        u, vPrime, _ = self.uvPrimew.T  # Unpack the transformed coordinates
        x_L, x_W = self.x_L, self.x_W
        WPrime = self.fault_width * np.cos(np.radians(self.dip))
        L = self.fault_length

        u_c = np.maximum(np.minimum(u, (1 - x_L) * L), -x_L * L)
        vPrime_c = np.maximum(np.minimum(vPrime, (1 - x_W) * WPrime), -x_W * WPrime)

        Rjb = np.sqrt( ((u - u_c) ** 2) + ((vPrime - vPrime_c) ** 2) )

        return Rjb

    def calc_Rx(self, points,convert_xyz=True):
        """Distance to x-axis

        Input: (n,3) array of xyz coords (centered at the surface)

        Set convert_xyz to False if coordinates are already in (n,e,z) format
        Otherwise, (x,y,z) is automatically converted to (n,e,z) instead of the incorrect (e,n,z)
        """
        self._require('dip', 'fault_width')
        self.frac_hyp_coords()
        self.point_uvPrimew(points,convert_xyz)  # Transform all points at once

        _, vPrime, _ = self.uvPrimew.T
        x_W = self.x_W
        WPrime = self.fault_width * np.cos(np.radians(self.dip))

        Rx = vPrime + (x_W * WPrime)

        return Rx
    
    def calc_Ry0(self,points,convert_xyz=True):
        """Distance to closest end of the top of the fault

        Input: (n,3) array of xyz coords (centered at the surface)
        """
        self._require('dip', 'fault_width')
        self.frac_hyp_coords()
        self.point_uvPrimew(points,convert_xyz)
        pass

def gen_surface_grid(lenx,leny,resx,resy):
    """Generates (resx*resx,3)-shape array of points with xyz-coords for map view

    Note that the Distances functions will automatically convert xyz to neu 
    (unless convert_xyz is set to False), which is the format needed for 
    the distance calculations
    """
    x = np.linspace(-lenx/2,lenx/2,resx)
    y = np.linspace(-leny/2,leny/2,resy)
    z = np.zeros((resx,resy))

    xx, yy = np.meshgrid(x,y)

    points = np.vstack([xx.ravel(),yy.ravel(),z.ravel()]).T
    return points

def gen_rand_points(num_points):
    """Generates random xyz points around the origin,
    all on z=0 surface

    Note that the Distances functions will automatically convert xyz to neu 
    (unless convert_xyz is set to False), which corrects
    for the hypocentral depth and is the format needed for 
    the distance calculations.
    """
    rng = np.random.default_rng()
    points = 5000 * (2 * rng.random((num_points,3)) - 1)
    points[:,2] = 0
    return points

class SeisSolOutput:
    """Handlers for SeisSol outputs.

    Handles SeisSol outputs for data analysis and plotting. Includes functions
    for XDMF and CSV-formatted outputs.

    Attr:
        prefix: output folder prefix [string]

        This prefix is set in the SeisSol .par setup file in the Output line

        ...
        &Output
        OutputFile = 'output/{file_prefix}'
        ...

        It is the name of the folder containing the output files. All files therein
        are assumed to share this prefix.
    """
    def __init__(self, prefix, print_status=False):
        self.prefix = prefix
        if print_status: print(f"Now reading SeisSol outputs for folder {prefix}")

    def load_surface_mesh(self,t):
        """Loads surface mesh from file prefix at time t.

        Args:
            t: simulation time step to load. Must be a valid time in mesh data
        Returns:
            UnsructuredGrid: object plottable by PyVista with Plotter.add_mesh()
        """
        p = self.prefix
        reader = vtk.vtkXdmfReader()
        reader.SetFileName(f'{p}/{p}-surface.xdmf')
        reader.Update()
        reader.UpdateTimeStep(t)
        return pv.wrap(reader.GetOutput())
    
    def load_fault_mesh(self,t):
        """Loads fault mesh from file prefix at time t.

        Args:
            t: simulation time step to load. Must be a valid time in mesh data
        Returns:
            UnsructuredGrid: object plottable by PyVista with Plotter.add_mesh()
        """
        p = self.prefix
        reader = vtk.vtkXdmfReader()
        reader.SetFileName(f'{p}/{p}-fault.xdmf')
        reader.Update()
        reader.UpdateTimeStep(t)
        return pv.wrap(reader.GetOutput())
    
    def get_time_steps(self):
        """Get simulation time steps. Useful for selecting valid time steps.

        Returns:
            Array: time steps of simulation
        """
        p = self.prefix

        tt = pv.XdmfReader(f'{p}/{p}.xdmf').time_values

        return tt
    
    def get_mw(self, roundTo = 2):
        """Given SeisSol energy csv output, get magnitude at last time step of simulation.

        Args:
            roundTo: decimal points magnitude will be returned at (default: 2)
        
        Returns: 
            float: moment magnitude
        """
        p = self.prefix
        energy_file = f'{p}/{p}-energy.csv'
        energy_df = pd.read_csv(energy_file)
        M0_df = energy_df[energy_df['variable'] == 'seismic_moment'][['time','measurement']]
        M0_final = M0_df.iloc[-1].measurement
        # Kanamori 1997 magnitude
        return round(2/3 * (np.log10(M0_final*1e7) - 16.1),roundTo)
    
    def get_moment(self):
        """Given SeisSol energy csv output, get moment data

        Returns:
            Array: simulation time steps from csv
            Array: total moment release at each time step
            Array: moment release rate at each time step
        """
        p = self.prefix
        energy_file = f'{p}/{p}-energy.csv'
        energy_df = pd.read_csv(energy_file)
        M0_df = energy_df[energy_df['variable'] == 'seismic_moment'][['time','measurement']]
        times = M0_df['time'].to_numpy()
        moments = M0_df['measurement'].to_numpy()
        rates = np.gradient(moments,times)
        
        return times, moments, rates
    
    def det_dip(self):
        """Gets dip from file prefix.

        Assumes prefix has Franco, 2026 naming convention 
        where dip is encoded with the prefix 'd'; e.g., 'xxx-d45-xxx' for 45° dip

        Returns:
            float: dip in degrees
        """
        try:
            m = re.search(r'_d(\d+)', self.prefix)
            dip = (float(m.group(1)))
            return dip
        except ValueError:
            warnings.warn(f"For this function to work properly, you must ensure \
                        dip is encoded within the filename with the prefix 'd'.\
                        Your input is {self.prefix}", UserWarning)
            
    def get_rotd50(self,t):
        """
        Compute RotD50 for the whole mesh at the loaded time step at once.

        Args:
            t: time step for surface mesh to calculate [float]
        Returns:
            np.ndarray: Array of RotD50 values for all cells in m/s.
        """
        mesh = self.load_surface_mesh(t)
        v1 = np.array(mesh.cell_data['v1'])  # Shape: (n_cells, n_time_steps)
        v2 = np.array(mesh.cell_data['v2'])  # Shape: (n_cells, n_time_steps)

        thetaR = np.linspace(0, np.pi, 180, endpoint=False)
        cosR, sinR = np.cos(thetaR), np.sin(thetaR)

        # Compute rotated amplitudes for all cells at once
        yR = np.outer(cosR, v1) + np.outer(sinR, v2)  # Shape: (360, n_cells)

        # Compute RotD50 for each cell
        rotd50 = np.median(np.abs(yR),axis=0)  # Median across angles (shape: (n_cells,))
        return rotd50
    
    def get_pgv_rotd50(self,print_status=False):
        """
        Get peak Rotd50s for the mesh for the full simulation duration.

        Args:
            print_status: print useful info about execution time [bool; default False]

        Returns:
            np.ndarray: Array of maximum RotD50 values for all cells in m/s.

            This can be added to a loaded surface mesh using \
            surface_mesh.cell_data.set_array(mesh_pgv_rotd,'pgv_rotd50') \
            to then plot with PyVista plotters.
        """
        fullstart = time.time()
        # store pgvs here
        n_cells = self.load_surface_mesh().n_cells
        mesh_pgv_rotd = np.zeros(n_cells)
        tt = self.get_time_steps()

        for t in tt:
            start = time.time()
            rotd = self.get_rotd50(t)
            mesh_pgv_rotd = np.maximum(mesh_pgv_rotd,rotd)
            end = time.time()
            if print_status: print(f'Getting PGV Rotd50 for simulation {self.prefix}. \
                                   Time step {t}/{tt[-1]}. \
                                    Iteration time: {round(end - start,3)} s',end='\r')

        fullend = time.time()
        if print_status: print(f'\nExecution time: {round(fullend - fullstart,3)} s')

        return mesh_pgv_rotd