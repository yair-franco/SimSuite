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

# openquake scenario functions
from openquake.hazardlib.contexts import RuptureContext
from openquake.hazardlib.contexts import DistancesContext
from openquake.hazardlib.contexts import SitesContext
from openquake.hazardlib import imt, const

# active crustal models
from openquake.hazardlib.gsim.abrahamson_2014 import AbrahamsonEtAl2014
from openquake.hazardlib.gsim.boore_2014 import BooreEtAl2014
from openquake.hazardlib.gsim.campbell_bozorgnia_2014 import CampbellBozorgnia2014
from openquake.hazardlib.gsim.chiou_youngs_2014 import ChiouYoungs2014

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

        Args: 
            - (N,3): array of xyz points
        Returns: 
            - (N,3): array of nez points with corrected z-coords
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
        Not dependent on dip; used for Rjb and Rx
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
        
        Args: 
            - (n,3): array of xyz coords (centered at the surface)


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

        Args: 
            - (n,3): array of xyz coords (centered at the surface)

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

        Args: 
            - (n,3): array of xyz coords (centered at the surface)

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

        Args: 
            - (n,3): array of xyz coords (centered at the surface)
        """
        self._require('dip', 'fault_width')
        self.frac_hyp_coords()
        self.point_uvPrimew(points,convert_xyz)
        pass

    def shift_x_tor(self,points):
        """Shifts coordinate x values so that (0,0,0) is the center of the top of the rupture (tor),
        instead of at the hypocenter.
        Assumes hypocenter is centered along strike and strike is 0.

        Args: 
            - (n,3): array of xyz coords (centered at the surface)
        """

        shift = self.hyp_depth / np.tan(np.radians(self.dip))
        shifted_x = points[:,0] - shift
        return shifted_x


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

            This can be added to a loaded surface mesh using
            surface_mesh.cell_data.set_array(mesh_pgv_rotd,'pgv_rotd50')
            to then plot with PyVista plotters.
        """
        fullstart = time.time()
        # store pgvs here
        n_cells = self.load_surface_mesh(0).n_cells
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
        
class Materials:
    """Handlers for calculating material ground properties based on given parameters.

    Simplifies finding missing parameters for calculating certain properties like ground velocity,
    and gets parameters from given properties in reverse.

    All functions return their output in the same units as they are input.

    Attr:
        mu: shear modulus (Pa)
        rho: density (kg/m^3)
        K: bulk modulus (Pa)
        l: lambda, Lamé first parameter
        Vp: P-wave velocity (m/s)
        Vs: S wave velocity (m/s)
    """
    def __init__(self, mu=None, rho=None, l=None, K=None, Vp=None, Vs=None):
        self.mu = mu
        self.rho = rho
        self.l = l
        self.K = K if K is not None else None

        if K == None and l != None:
            self.K = self.l + ((2/3) * self.mu)
        
        self.Vp = Vp
        self.Vs = Vs

    def _require(self, *names):
        missing = [n for n in names if getattr(self, n) is None]
        if missing:
            raise ValueError(f"Missing required properties: {', '.join(missing)}")

    def get_Vs(self):
        """Gets shear velocity from mu and rho"""
        self._require('mu','rho')
        return np.sqrt(self.mu/self.rho)

    def get_Vp(self):
        """Gets p-wave velocity from K, mu, and rho"""
        self._require('K','mu','rho')
        return np.sqrt((self.K + ((4/3) * self.mu)) / self.rho)

    def get_mu(self):
        """Gets mu from Vs and rho"""
        self._require('Vs','rho')
        return self.Vs**2 * self.rho

    def get_K(self):
        """Gets K from lambda and mu"""
        self._require('l','mu')
        return self.l + ((2/3) * self.mu)

    # def get_K(self):
    #     self._require('Vp','rho','mu')
    #     return (self.Vp**2 * self.rho) - ((4/3) * self.mu)

def get_fault_norm(strike,dip):
    """Calculates normal vector of fault plane from strike and dip.

    Args:
        strike: strike angle (in degrees)
        dip: dip angle (in degrees)

    Returns:
        A normalized vector in the normal direction to the plane.
    """
    strike_rad = np.deg2rad(strike)
    dip_rad = np.deg2rad(dip)

    nx = -np.sin(dip_rad) * np.sin(strike_rad)
    ny = -np.sin(dip_rad) * np.cos(strike_rad)
    nz = np.cos(dip_rad)

    N = np.array([nx, ny, nz])

    Nn = N / np.linalg.norm(N)
    # reduce float rounding error values to 0, and round non-zero values
    Nn = np.array([0 if abs(c) <= 1e-7 else c for c in Nn])
    Nn.round()

    return Nn

def OpenquakeACScenario(rake,distances,seissoloutput=None,mag=None,dip=None,convert_from_ln=True):
    """Simplifies syntax for getting PGVs for an Active Crust model scenario using OpenQuake

    This is a simplified context in which Vs30 is assumed at 760 m/s for all points, 
    and other parameters are also generalized.

    Args:
        - rake: rupture rake (degrees), which defines fault mechanism. Can be a number, or the strings 
        "strikeslip" for 0, "normal" for -90, or "thrust" for 90 (float or string)
        - coords: array of (x,y,z) coords to be used by GMMs. These are automatically converted
        by the Distances class to (n,e,u) for rotation
        - distances: object of distance handlers initialized with Distances class (object)
        - seissoloutput: object of SeisSol output handlers initialized with SeisSolOutput class (object, default None)
        - mag: earthquake magnitude. If not specified, it is extracted from SeisSol output (float, default None)
        - dip: rupture dip. If not specified, it is extracted from the distances input (float, default None)

    Returns:
        - (resx,resy) array: (x,y) grid of GMM PGVs whose resolution is based on SeisSol simulation mesh
    """

    if mag is None:
        if seissoloutput:
            mag = seissoloutput.get_mw()
        else:
            raise ValueError("No magnitude found.\
                             Please input a magnitude or a SeisSolOutput object for a magnitude.")

    if dip is None:
        dip = distances.dip

    if not isinstance(rake,str):
        if rake > 180 or rake < -180:
            raise ValueError("Rake must be between -180 and 180")
    if not isinstance(rake,(float,int)):
        if rake in ("ll","llss","left-lateral"):
            rake = 0
        elif rake in ("rl","rlss","right-lateral"):
            rake = 180
        elif rake in ("n","ns","normal"):
            rake = -90
        elif rake in ("t","r","rs","thrust", "reverse"):
            rake = 90
        else:
            raise ValueError("No valid rake value entered. \
                          Valid strings are 'left-lateral', 'right-lateral' 'normal', 'reverse'.")

        mesh_points = seissoloutput.load_surface_mesh(0).cell_centers().points / 1000

        #OpenQuake scenario setup
        rctx = RuptureContext()
        rctx.mag = mag
        rctx.rake = rake
        rctx.dip = dip
        rctx.ztor = distances.ztor
        rctx.width = distances.fault_width
        rctx.hypo_depth = distances.hyp_depth

        dctx = DistancesContext()

        #Distances calculated with Distances object functions
        rrups = distances.calc_Rrup(mesh_points)
        rjbs = distances.calc_Rjb(mesh_points)
        rxs = distances.calc_Rx(mesh_points)
        # ry0s = distances.calc_Ry0(mesh_points)

        npts = rrups.size # number of distance points

        dctx.rrup = rrups 
        dctx.rjb = rjbs 
        dctx.rx = rxs 
        # dctx.ry0 = ry0s

        sitecol_dict = {'sids': np.arange(1, npts + 1), # site id
                'vs30': 760.0 + np.zeros(npts, dtype=float), # Vs30 value in m/s
                'vs30measured': np.zeros(npts, dtype=bool), # needed parameter, but does not affect PGVs here
                'z1pt0': 50.0 + np.zeros(npts, dtype=float), # depth in m (not km) to 1.0 km/s horizon
                'z2pt5': np.nan + np.zeros(npts, dtype=float), # depth in km (not m) to 2.5 km/s horizon
                }
        sitecollection = pd.DataFrame(sitecol_dict)

        sctx = SitesContext(sitecol=sitecollection)

        # Calculate GMM PGVs
        IMT = imt.PGV()
        uncertaintytype = const.StdDev.TOTAL

        # Initialize GMMs
        ASK14 = AbrahamsonEtAl2014()
        BSSA14 = BooreEtAl2014()
        CB14 = CampbellBozorgnia2014()
        CY14 = ChiouYoungs2014()

        # Get log ground motions
        # First value are median PGVs, second value standard deviation
        ln_ask14, ln_sd_ask14 = ASK14.get_mean_and_stddevs(sctx, rctx, dctx, IMT, [uncertaintytype])
        ln_bssa14, ln_sd_bssa14 = BSSA14.get_mean_and_stddevs(sctx, rctx, dctx, IMT, [uncertaintytype])
        ln_cb14, ln_sd_cb14 = CB14.get_mean_and_stddevs(sctx, rctx, dctx, IMT, [uncertaintytype])
        ln_cy14, ln_sd_cy14 = CY14.get_mean_and_stddevs(sctx, rctx, dctx, IMT, [uncertaintytype])

        # Get average
        ln_median_ac14 = 0.25 * (ln_ask14 + ln_bssa14 + ln_cb14 + ln_cy14)
        ln_sd_ac14 = 0.25 * (ln_sd_ask14[0] + ln_sd_bssa14[0] + ln_sd_cb14[0] + ln_sd_cy14[0]) # sds are in a list of length 1

        if convert_from_ln:
            # ask, sd_ask = np.exp(ln_ask14), np.exp(ln_sd_ask14)
            # bssa, sd_bssa = np.exp(ln_bssa14), np.exp(ln_sd_bssa14)
            # cb, sd_cb = np.exp(ln_cb14), np.exp(ln_sd_cb14)
            # cy, sd_cy = np.exp(ln_cy14), np.exp(ln_sd_cy14)

            ac, sd_ac = np.exp(ln_median_ac14), np.exp(ln_sd_ac14)

            return ac, sd_ac
        else:
            return ln_median_ac14, ln_sd_ac14

            

            
        




