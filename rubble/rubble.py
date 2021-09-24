from astropy import units as u
from astropy import constants as c
import numpy as np
import scipy.optimize as spopt
import scipy.integrate as spint
import scipy.interpolate as spinterp
import scipy.linalg as spla
import scipy.special as spsp
import logging
import time
import warnings


def rho_d_TL02(rho_d0, z, Hg, tau_s0, alpha, Sc=1.0):
    """ Vertical density profile of particles

    REFs: Equation 31 in Takeuchi & Lin 2002
    """

    return rho_d0 * np.exp(-z ** 2 / (2 * Hg ** 2) - Sc * tau_s0 / alpha * (np.exp(z ** 2 / Hg ** 2) - 1))


class Rubble:
    """ Simulate the local evolution of solid size distributions in Protoplanetary Disks (PPDs) """

    # Making flags Descriptors so kernels can be updated when flags change
    class FlagProperty:

        def __init__(self, name, doc_string=None):
            self.name = name
            if doc_string is not None:
                self.__doc__ = doc_string  # give doc string printed in help(Rubble)

        def __get__(self, instance, owner):
            if instance is None:
                return self
            else:
                # get attribute from the instance
                return getattr(instance, '_%s' % self.name)  # return x._prop

        def __set__(self, instance, value):
            # set attribute and the corresponding key in the "remote" dict (dict not necessary, just for bookkeeping)
            instance._flag_dict[self.name] = value  # x._flag_dict["prop"] = value
            setattr(instance, '_%s' % self.name, value)  # x._prop = value
            
            # whenever a flag changes, execute flag_updated()
            instance.flag_updated(self.name)

    # flags
    debug_flag = FlagProperty("debug_flag",
                              "whether or not to enable experimental features (default: False)")
    frag_flag = FlagProperty("frag_flag", 
                             "whether or not to calculate fragmentation (default: True)")
    mass_transfer_flag = FlagProperty("mass_transfer_flag",
                                      "whether or not to include mass transfer as a fragmentation branch (default: "
                                      "True)")
    bouncing_flag = FlagProperty("bouncing_flag", 
                                 "whether or not to include bouncing (besides coagulation and fragmentation; default: "
                                 "True)")
    vel_dist_flag = FlagProperty("vel_dist_flag", 
                                 "whether or not to consider velocity distribution in collisional outcome "
                                 "calculations (default: True)")
    closed_box_flag = FlagProperty("closed_box_flag", 
                                   "whether or not to use a closed box (i.e., no solid loss or supply; default: True)")
    dyn_env_flag = FlagProperty("dyn_env_flag",
                                "whether or not to consider dynamic dust exchange with the environment "
                                "(i.e., varying Mdot, so call init_update_solids each time; default: False)")
    simple_St_flag = FlagProperty("simple_St_flag", 
                                  "whether or not to only use Epstein regime for Stokes number (i.e., ignore Stokes "
                                  "regime 1; default: False)")
    full_St_flag = FlagProperty("full_St_flag", 
                                "whether or not to include Stokes regime 3 & 4 for Stokes number (default: False => "
                                "so Epstein & Stokes regime 1)")
    f_mod_flag = FlagProperty("f_mod_flag",
                              "whether or not to use a modulation factor to limit the coagulation with mass bins with "
                              "< 1 particle numbers (e.g., over an 0.1 AU wide annulus at 1 AU; default: False)")
    uni_gz_flag = FlagProperty("uni_gz_flag",
                               "whether or not to use unidirectional ghost zones (so the left gz also coagulate "
                               "and the right one fragment; default: False => masses sink to inactive ghost zones)")
    feedback_flag = FlagProperty("feedback_flag",
                                 "whether or not to consider the feedback (i.e., back reaction) of solids, so that "
                                 "diffusion coefficients will be damped with (1+eps)^{-K}, reducing H_d and dv_ij")

    def __init__(self, num_grid, amin, amax, q, Sigma_d, 
                 rho_m=1.6, run_name="rubble_test", **kwargs):
        """
        Initialize the radius/mass grid of solid particles, distribute solids and prepare kernels

        Parameters
        ----------
        num_grid : int
            number of grid cells along radius/mass grid
        amin : float
            minimum particle radius, in units of cm
        amax : float
            maximum particle radius, in units of cm
        q : float
            initial power law index of the size distribution
        Sigma_d : float
            total dust surface density, in units of g/cm^2
        rho_m : float
            material density of solids, in units of g/cm^3, default to 1.6
        kwargs
            other disk parameters and numerical parameters (some are necessary)

        About kwargs
        ------------
        1. Rubble needs the following gas quantities to calculate Stokes number:
            - Sigma_g: gas surface density (g/cm^2), default to 100 * Sigma_d
            - T      : gas temperature:, default to 280 K
            - H      : gas scale height, default to 1.78e11 cm (assuming 0.5 AU)
            - alpha  : alpha-parameter, default to 1e-3
        
        2. The following keywords can be used to customize the initial solid distribution 
            (by default, solids will be distribute to all size bins)
            (assuming a_delta, a_i, a_f are float numbers denoting sizes)
            - delta_dist = a_delta, put all solids in one single bin that is closest to a_delta
            - ranged_dist = [a_i, a_f], put solids into bins within size range from a_i to a_f
        
        3. Please use `help(Rubble)` to check available flags and their purposes in definitions of "Data descriptors"

        4. Numerically, you may vary the following quantities to control solid evolution:
            - xi    : power law index of fragment distribution, default to 1.83
            - chi   : mass excavated from target by projectile due to cratering, default to 1 (unit: projectile mass)
            - chi_MT: mass transferred from the projectile to target, default to 0.1 (unit: projectile mass)
            - u_b   : velocity threshold for bouncing, default to 5 cm/s
            - u_f   : velocity threshold for fragmentation, default to 100 cm/s
        
        5. Also, Rubble needs the following quantities if closed_box_flag is False:
            This flag is used to study the dust accumulation in the pressure bump at/near the inner disk boundary.
            In this scenario, it is recommended to set full_St_flag to True to accommodate high temperature case.
            REFs: Li, Chen, and Lin, in prep
            - Raccu   : accumulation radius, default to 0.1 AU
            - Mdot    : stellar accretion rate, default to 3e-9 solar mass / year
            - Z       : dust-to-gas ratio of the accreted materials, default to 0.01
            - a_critD : critical particle size below which will be lifted by the accretion funnel, default to 0.01 cm
            - a_max_in: maximum solid size in the accreted materials (i.e., max supply size), default to 10 cm

        Notes
        -----
        1. The results of solving the Smoluchowski equation are known to depend on resolution, initial setup,
        and the length of time step (and also the Python environment and the machine used).  Be sure to test
        different setups and find the converging behaviors.

        2. One instance of Rubble is designed to run one physical simulation with one set of parameters.
        Restarting the same simulation is only partially supported.

        """

        # establish radius grid
        self.log_amin, self.log_amax, self.Ng, Ng = np.log10(amin), np.log10(amax), num_grid, num_grid
        self.num_dec = self.log_amax - self.log_amin                  # number of decades that a span
        self.dlog_a = self.num_dec / (Ng - 1)                         # step in log space along radius
        self.comm_ratio = (10**self.num_dec)**(1 / (Ng - 1))          # the common ratio of the radius sequence
        # one ghost zone is needed on each side (gz means ghost zone; l/r means left/right)
        self.log_algz, self.log_argz = self.log_amin-self.dlog_a, self.log_amax+self.dlog_a  # left/right ghost zone
        self.a = np.logspace(self.log_algz, self.log_argz, Ng+2)      # (!) cell center of radius grid, units: cm
        self.log_a = np.linspace(self.log_algz, self.log_argz, Ng+2)  # log cell center of radius grid
        self.log_a_edge = np.append(self.log_a-self.dlog_a/2, self.log_argz+self.dlog_a/2)  # log cell edge
        self.a_edge = 10**(self.log_a_edge)                           # cell edge of radius grid

        # establish mass grid
        self.rho_m = rho_m                                            # material density of solids
        self.dlog_m = self.num_dec * 3 / (Ng - 1)                     # step in log space along mass
        self.m = 4*np.pi/3 * rho_m * self.a**3                        # cell center of mass grid, units: g
        self.log_m = np.log10(self.m)                                 # log cell center of mass grid
        self.log_m_edge = 4*np.pi/3 * rho_m * self.a_edge**3          # log cell edge of mass grid
        self.m_edge = 4*np.pi/3 * rho_m * self.a_edge**3              # cell edge of mass grid

        # initial power law index (conventionally leaving out minus sign)
        self.q = q                          # the power law index of the size distribution, i.e., dN/da propto a^{-q}
        self.p = (q + 2) / 3                # the power law index of the mass distribution, i.e., dM/da propto a^{-p}
        self.s = q - 4                      # index of the surface density distribution per log(a)
        # other physical quantities
        self.Sigma_d = Sigma_d              # total dust surface density, in units of g/cm^2
        
        self.cgs_k_B = c.k_B.to(u.g*u.cm**2/u.s**2/u.K).value  # k_B in cgs units
        self.cgs_mu = 2.3 * c.m_p.to(u.g).value                # mean molecular weigth * proton mass (sometimes 2.34)
        self.sigma_H2 = 2.0e-15                                # cm^2, cross section for H2 (may fail when T < 70K)
        self.sqrt_2_pi = np.sqrt(2 * np.pi)
        self.S_annulus = 2*np.pi*1.0*0.1 * (u.au.to(u.cm))**2  # surface an 0.1 AU wide annulus at 1 AU
        self.feedback_K = 1                                    # 1/(1 + <eps>)^{...} when inc. feedback effects
        self.kwargs = kwargs
        self.init_disk_parameters()                            # extract disk parameters from self.kwargs

        # vertically integrated dust surface density distribution per logarithmic bin of grain radius
        self.sigma = np.zeros(Ng+2)         # see comments above, in units of g/cm^2
        self.dsigma_in = np.zeros(Ng+2)     # supply from outer disk
        self.dsigma_out = np.zeros(Ng+2)    # loss due to accretion to the central star
        self.Nk = np.zeros(Ng+2)            # vertically integrated dust number density per log mass bin, units: 1/cm^2
        self.dN = np.zeros(Ng+2)            # dN per time step
        self.Na = np.zeros(Ng+2)            # vertically integrated dust number density per log size bin, units: 1/cm^2
        self.St = np.zeros(Ng+2)            # Stokes number
        self.St_12 = 0                      # particles below this are tightly-coupled
        self.St_regimes = np.zeros(Ng+2)    # St regimes (1: Epstein; 2: Stokes, Re<1; 3: 1<Re<800; 4: Re>800)
        self.H_d = np.zeros(Ng+2)           # dust scale height
        self.eps = np.zeros(Ng+2)           # midplane dust-to-gas density ratio
        self.eps_tot = 0.01                 # total midplane dust-to-gas density ratio
        self.FB_eps_cap = 100               # max eps to cap the feedback effects
        self.Re_d = np.zeros(Ng+2)          # Reynolds-number Re = 2 a u / nu_mol
        self.Hratio_loss = None             # function to calculate solid loss fraction due to accretion flow

        # basic matrixes used in simulation; following the subscript, use i as the highest dimension, j second, k third,
        # meaning the changes due to i happens along the axis=0, changes due to j on axis=1, due to k on axis=2
        # e.g, for arr = [1,2,3],                   then arr_i * arr_j will give arr_ij,
        # arr_i is [[1,1,1],    arr_j is [[1,2,3],       meaning arr_ij[i][j] is from m_i and m_j
        #           [2,2,2],              [1,2,3],
        #           [3,3,3]],             [1,2,3]]
        self.m_i = np.tile(np.atleast_2d(self.m).T, [1, self.Ng+2])
        self.m_j = np.tile(self.m, [self.Ng+2, 1])
        self.m_sum_ij = self.m + self.m[:, np.newaxis]
        self.m_prod_ij = self.m * self.m[:, np.newaxis]
        self.a_sum_ij = self.a + self.a[:, np.newaxis]
        self.a_prod_ij = self.a * self.a[:, np.newaxis]

        # only indexing='ij' will produce the same shape indices
        self.mesh2D_i, self.mesh2D_j = np.meshgrid(np.arange(self.Ng+2), np.arange(self.Ng+2), indexing='ij')
        self.mesh3D_i, self.mesh3D_j, self.mesh3D_k = np.meshgrid(np.arange(self.Ng+2), np.arange(self.Ng+2),
                                                                  np.arange(self.Ng+2), indexing='ij')
        self.idx_ij_same = self.mesh3D_i == self.mesh3D_j
        self.idx_jk_same = self.mesh3D_j == self.mesh3D_k
        self.idx_jk_diff = self.mesh3D_j != self.mesh3D_k

        self.zeros2D = np.zeros([self.Ng+2, self.Ng+2])
        self.ones3D = np.ones([self.Ng+2, self.Ng+2, self.Ng+2])

        # intermediate matrixes used in simulation
        self.dv_BM = np.zeros([self.Ng+2, self.Ng+2])          # relative velocity due to Brownian motion
        self.dv_TM = np.zeros([self.Ng+2, self.Ng+2])          # relative velocity due to gas turbulence
        self.dv = np.zeros([self.Ng+2, self.Ng+2])             # relative velocity, du_ij
        self.geo_cs = np.pi * self.a_sum_ij**2                 # geometrical cross section, sigma_ij, units: cm^2
        self.h_ss_ij = np.zeros([self.Ng+2, self.Ng+2])        # h_i^2 + h_j^2, ss means "sum of squared", units: cm^2
        self.vi_fac = np.zeros([self.Ng+2, self.Ng+2])         # vertically-integration factor, sqrt(2 pi h_ss_ij)
        self.kernel = np.zeros([self.Ng+2, self.Ng+2])         # the general kernel, only du_ij * geo_cs_ij
        self.p_c = np.zeros([self.Ng+2, self.Ng+2])            # the coagulation probability
        self.p_b = np.zeros([self.Ng+2, self.Ng+2])            # the bouncing probability
        self.p_f = np.zeros([self.Ng+2, self.Ng+2])            # the fragmentation probability
        self.K = np.zeros([self.Ng+2, self.Ng+2])              # the coagulation kernel
        self.L = np.zeros([self.Ng+2, self.Ng+2])              # the fragmentation kernel
        self.f_mod = np.zeros([self.Ng+2, self.Ng+2])          # the modulation factor to disable bins with tiny Nk

        # TODO: it is possible to save RAM cost by eliminating M1-M4, e.g., just use M and add to tM step by step
        self.C = np.zeros([self.Ng+2, self.Ng+2, self.Ng+2])   # the epsilon matrix to distribute coagulation mass
        self.gF = np.zeros([self.Ng+2, self.Ng+2, self.Ng+2])  # the gain coeff of power-law dist. of fragments
        self.lF = np.ones([self.Ng+2, self.Ng+2, self.Ng+2])   # the loss coeff of power-law dist. of fragments
        self.M1 = np.zeros([self.Ng+2, self.Ng+2, self.Ng+2])  # part of kernel of the Smoluchowski equation
        self.M2 = np.zeros([self.Ng+2, self.Ng+2, self.Ng+2])  # part of kernel of the Smoluchowski equation
        self.M3 = np.zeros([self.Ng+2, self.Ng+2, self.Ng+2])  # part of kernel of the Smoluchowski equation
        self.M4 = np.zeros([self.Ng+2, self.Ng+2, self.Ng+2])  # part of kernel of the Smoluchowski equation

        # ultimate matrixes used in the implicit step
        self.I = np.identity(self.Ng+2)                        # identity matrix
        self.S = np.zeros(self.Ng+2)                           # source func
        self.J = np.zeros([self.Ng+2, self.Ng+2])              # Jacobian of the source function
        self.M = np.zeros([self.Ng+2, self.Ng+2, self.Ng+2])   # kernel of the Smoluchowski equation
        self.tM = np.zeros([self.Ng+2, self.Ng+2, self.Ng+2])  # the vertically-integrated kernel, t: tilde

        # numerical variables
        self.t = 0                                             # run time, units: yr
        self.dt = 0                                            # time step, units: yr
        self.cycle_count = 0                                   # number of cycles
        self.rerr = 0                                          # relative error of the total surface density
        self.s2y = u.yr.to(u.s)                                # ratio to convert seconds to years
        self.out_dt = 0                                        # time interval to output simulation
        self.next_out_t = 0                                    # next time to output simulation
        self.res4out = np.zeros([1+(self.Ng+2)*2])             # results for output, [t, sigma, Nk] for now
        self.rerr_th = self.kwargs.get('rerr_th', 1e-6)        # threshold for relative error to issue warnings
        self.rerr_th4dt = self.kwargs.get('rerr_th4dt', 1e-6)  # threshold for relative error to lower timestep
        self.neg2o_th = self.kwargs.get('neg2o_th', 1e-15)     # threshold (w.r.t. Sigma_d) for reset negative Nk to 0
        self.negloop_tol = self.kwargs.get('negloop_tol', 20)  # max. No. of loops to reduce dt to avoid negative Nk
        self.dynamic_dt = False                                # whether or not to use a larger desired dt; set by run()
        self.dyn_dt = self.kwargs.get('dyn_dt', 1)             # desired dt (only used if it speeds up runs)
        self.tol_dyndt = self.kwargs.get('tol_dyndt', 1e-7)    # tolerance for relative error to use larger dynamic dt
        self.dyn_dt_success = False                            # if previous dynamic dt succeed, then continue using it

        self.log_func = print                                  # function to write log info
        self.warn_func = print                                 # function to write warning info
        self.err_func = print                                  # function to write error info
        self.run_name = run_name                               # name of this simulation run
        self.dat_file = None                                   # file handler for writing data
        self.dat_file_name = run_name + ".dat"                 # filename for writing data
        self.log_file = None                                   # file handler for writing logs
        self.log_file_name = run_name + ".log"                 # filename for writing logs

        # flags
        self.static_kernel_flag = self.kwargs.get('static_kernel', True)  # assuming kernel to be static
        self.flag_activated = False                            # simply set flag values and skip flag_updated()
        self._flag_dict = {}                                   # an internal flag dict for bookkeeping
        self.debug_flag = self.kwargs.get('debug', False)      # whether or not to enable experimental features
        self.frag_flag = self.kwargs.get('frag', True)         # ..................calculate fragmentation
        self.mass_transfer_flag = self.kwargs.get('MT', True)  # ..................calculate mass transfer
        self.bouncing_flag = self.kwargs.get('bouncing', True) # ..................include bouncing
        self.vel_dist_flag = self.kwargs.get('VD', True)       # ..................include velocity distribution
        self.closed_box_flag = self.kwargs.get('CB', True)     # ..................use a closed box (so no loss/supply)
        self.dyn_env_flag = self.kwargs.get('dyn_env', False)  # ..................consider dynamic loss/supply
        self.simple_St_flag = self.kwargs.get('simSt', False)  # ..................use only Epstein regime for St
        self.full_St_flag = self.kwargs.get('fullSt', False)   # ..................use full four regimes for St
        self.f_mod_flag = self.kwargs.get('f_mod', False)      # ..................use f_mod for coagulation
        self.uni_gz_flag = self.kwargs.get('uni_gz', False)    # ..................use unidirectional ghost zones
        self.feedback_flag = False                             # ..................inc. solid feedback to alpha_D

        # run preparation functions
        self.init_powerlaw_fragmentation_coeff()               # extract frag-related parameters from self.kwargs
        self.piecewise_coagulation_coeff()                     # based on mass grid, no dependence on other things
        self.powerlaw_fragmentation_coeff()                    # based on mass grid, no dependence on other things
        self.distribute_solids()                               # initial setup: distribute solids to each size/mass bins
        if not self.closed_box_flag:
            self.init_update_solids()  # initialize accretion part, based on Mdot, Raccu, H, Z
            # this will update self.S_annulus, which will be used in self.update_kernels

        self.update_kernels()                                  # the kernel is static if gas disk remains static

        self.flag_activated = True  # some flags may change static_kernel_flag
        self.feedback_flag = self.kwargs.get('FB', False)      # some flags require initialization of H_d, St, eps

    def __reset(self):
        """ Reset most internal data with zero-filling
            such a reset function is not helpful if we just use __init__ to initialize everything
            TODO: need to re-think the role of this function
        """

        pass

    def init_disk_parameters(self):
        """ Obtain disk parameters, de-clutter other functions """

        self.Sigma_g = self.kwargs.get('Sigma_g', self.Sigma_d*100)  # total gas surface density, in units of g/cm^2
        self.Sigma_dot = 0.0                                         # accretion rate on the surface density
        self.alpha = self.kwargs.get('alpha', 1e-3)                  # alpha-prescription, may != diffusion coeff
        self.T = self.kwargs.get('T', 280)                           # temperature, in units of K
        self.Omega = 5.631352229752323e-07                           # Keplerian orbital frequency, 1/s (0.5 AU, 1 Msun)
        self.H = self.kwargs.get('H', 178010309011.3974)             # gas scale height, cm (2.088e11 if with gamma=1.4)

        # derive more
        self.rho_g0 = self.Sigma_g / (self.sqrt_2_pi * self.H)       # midplane gas density
        self.lambda_mpf = 1 / (self.rho_g0 / self.cgs_mu * self.sigma_H2)  # mean free path of the gas, in units of cm
        self.c_s = (self.cgs_k_B * self.T / self.cgs_mu) ** 0.5      # gas sound speed, cm/s (1.1861e5 if gamma=1.4)
        self.nu_mol = 0.5 * np.sqrt(8 / np.pi) * self.c_s * self.lambda_mpf  # molecular viscosity

    def distribute_solids(self):
        """ Distribute the solids into all the grid by power law """

        if 'delta_dist' in self.kwargs:
            a_idx = np.argmin(abs(self.a - self.kwargs['delta_dist']))
            if a_idx < 1:
                raise ValueError(f"delta distribution outside simulation domain, a_min on grid is {self.a[1]:.3e}")
            if a_idx > self.Ng:
                raise ValueError(f"delta distribution outside simulation domain, a_max on grid is {self.a[-2]:.3e}")
            self.sigma[a_idx] = self.Sigma_d / self.dlog_a
        elif 'ranged_dist' in self.kwargs:
            a_idx_i = np.argmin(abs(self.a - self.kwargs['ranged_dist'][0]))
            a_idx_f = np.argmin(abs(self.a - self.kwargs['ranged_dist'][1]))
            if a_idx_f < a_idx_i:  # order reversed
                a_idx_i, a_idx_f = a_idx_f, a_idx_i
            if a_idx_i < 1:
                raise ValueError(f"a_min in ranged_dist is too small, a_min on grid is {self.a[1]:.3e}")
            if a_idx_i > self.Ng:  # outside right boundary
                raise ValueError(f"a_min in ranged_dist is too large, a_max on grid is {self.a[-2]:.3e}")
            if a_idx_f > self.Ng:  # outside right boundary
                raise ValueError(f"a_max in ranged_dist is too large, a_max on grid is {self.a[-2]:.3e}")
            if a_idx_f == a_idx_i:
                a_idx_f += 1
            tmp_sigma = np.zeros(self.Ng + 2)
            tmp_sigma[a_idx_i:a_idx_f + 1] = self.a[a_idx_i:a_idx_f + 1] ** (-self.s)
            C_norm = self.Sigma_d / np.sum(tmp_sigma * self.dlog_a)
            self.sigma = tmp_sigma * C_norm
        elif 'input_dist' in self.kwargs:
            try:
                self.sigma = self.kwargs['input_dist']
                self.Sigma_d = self.get_Sigma_d(self.sigma / (3 * self.m))
            except Exception as e:
                self.warn_func("fail to take the input distribution, revert back to default. "+e.__str__())
                tmp_sigma = self.a[1:-1] ** (-self.s)
                C_norm = self.Sigma_d / np.sum(tmp_sigma * self.dlog_a)
                self.sigma[1:-1] = tmp_sigma * C_norm
        else:
            tmp_sigma = self.a[1:-1] ** (-self.s)
            C_norm = self.Sigma_d / np.sum(tmp_sigma * self.dlog_a)
            self.sigma[1:-1] = tmp_sigma * C_norm
        self.Nk = self.sigma / (3 * self.m)
        self.Na = self.sigma / self.m
        self._Sigma_d = self.get_Sigma_d(self.Nk)  # numerical Sigma_d after discretization

    def piecewise_coagulation_coeff(self):
        """ Calculate the coefficients needed to piecewisely distribute coagulation mass

        There are two ways to distribute mass gain from m_i+m_j: one is piecewise reconstruction in Brauer+2008,
        the other is to retain all mass gained in the cell and convert to number density at the closest cell center.
        In this function, we go with the first method by default. Specify 'coag2nearest' to use the second method.

        Notes:
        assume m_m < m_i+m_j < m_n, where m_m and m_n are the nearest cell centers
        Q_ij = K_ij n_i n_j will be distributed to m_m and m_n by epsilon and (1 - epsilon), respectively
        also [epsilon Q_ij m_m + (1-epsilon) Q_ij m_n] should = Q_ij (m_i + m_j)
        thus epsilon = [m_n - (m_i + m_j)] / (m_n - m_m)

        for the second method, use epsilon = (m_i + m_j) / m_nearest

        Keep in mind that we need to put all the masses exceeding the ghost zone to the ghost zone
        e.g., when m_m is the right ghost zone, epsilon = (m_i + m_j) / m_rgz
        """

        coag2nearest = self.kwargs.get('coag2nearest', False)
        
        if not coag2nearest:
            # formula from Brauer+2008
            merger_n = np.searchsorted(self.m, self.m_sum_ij)  # idx for m_n, searchsorted returns where to insert
            merger_m = merger_n - 1                            # idx for m_m, 
            epsilon = np.zeros([self.Ng+2, self.Ng+2])         # epsilon, see reference
            epsilon2 = np.zeros([self.Ng+2, self.Ng+2])        # 1 - epsilon for non-ghost zone
            ngz_mask = merger_m<=self.Ng                       # non-ghost-zone mask

            epsilon[merger_m>self.Ng] = self.m_sum_ij[merger_m>self.Ng] / self.m[merger_m[merger_m>self.Ng]]
            epsilon[ngz_mask] = ((self.m[merger_n[ngz_mask]] - self.m_sum_ij[ngz_mask]) 
                                / (self.m[merger_n[ngz_mask]] - self.m[merger_m[ngz_mask]]))

            epsilon2[ngz_mask] = 1 - epsilon[ngz_mask]

            tmp_i, tmp_j = self.mesh2D_i, self.mesh2D_j
            self.C[tmp_i.flatten(), tmp_j.flatten(), merger_m.flatten()] = epsilon.flatten()
            self.C[tmp_i[ngz_mask], tmp_j[ngz_mask], merger_n[ngz_mask]] = epsilon2[ngz_mask]
        
        else:
            nth_right_edge = np.searchsorted(self.m_edge, self.m_sum_ij)
            merger = nth_right_edge - 1                        # idx for m_nearest 
            epsilon = np.zeros([self.Ng+2, self.Ng+2])         # epsilon, see reference
            ngz_mask = merger<=self.Ng                         # non-ghost-zone mask
            
            epsilon[merger>self.Ng] = self.m_sum_ij[merger>self.Ng] / self.m[-1]
            epsilon[ngz_mask] = self.m_sum_ij[ngz_mask] / self.m[merger[ngz_mask]]
            
            tmp_i, tmp_j = self.mesh2D_i, self.mesh2D_j
            merger[merger>self.Ng] = self.Ng + 1               # make sure k stays in bound 
            self.C[tmp_i.flatten(), tmp_j.flatten(), merger.flatten()] = epsilon.flatten()

    def init_powerlaw_fragmentation_coeff(self):
        """ Initialize numerical parameters needed for calculating coefficients for fragmentation kernels """

        # power law index of fragment distribution, ref: Brauer+2008
        self.xi = self.kwargs.get('xi', 1.83)
        if self.xi > 2:
            self.warn_func(f"The power law index of fragment distribution, xi = {self.xi} > 2, which means most mass "
                           + f"goes to smaller fragments. Make sure this is what you want.")
        
        # mass excavated from the larger particle due to cratering
        self.mratio_cratering = self.kwargs.get('mratio_cratering', 10)  # minimum mass ratio for cratering to happen
        self.chi = self.kwargs.get('chi', 1)
        if self.chi >= 10:
            self.warn_func(f"The amount of mass excavated from target by projectile due to cratering (happends when "
                           + f"mass ratio in a collision >= 10), in units of the projectile mass, is chi = {self.chi} "
                           + f">= 10, which means a complete fragmentation also happen to the target sometimes. "
                           + f"To keep it simple, chi has been reduced to 9.99.")
            self.chi = 9.99

        # mass transfer from the projectile to the target (N.B.: mass transfer is kind of reversed cratering)
        self.chi_MT = self.kwargs.get('chi_MT', -0.1)
        self.mratio_c2MT = self.kwargs.get('mratio_MT', 15)  # mass ratio to begin transition (cratering=>mass transfer)
        self.mratio_MT = self.kwargs.get('mratio_MT', 50)  # minimum mass ratio for full mass transfer effects
        if self.mass_transfer_flag:
            if self.chi_MT > 0: # using negative values to add mass back to targets
                self.chi_MT *= -1
            if self.chi_MT < -1.0:
                self.warn_func(f"Mass transfer fraction {self.chi_MT} cannot exceed 1.0. The maximum mass available to "
                            + f"transfer is 100% in units of the projectile mass.  To make it physical, chi_MT has "
                            + f"been reduced to 0.99")
                self.chi_MT = -0.99
            if self.mratio_MT < 20:
                self.warn_func(f"The minimum mass ratio for full mass transfer effects (mratio_MT = {self.mratio_MT}) "
                               + f"is smaller than 20. Mass transfer is more likely to happen when the mass difference"
                               + f" is larger. Be careful here.")
                if self.mratio_MT < 1:
                    raise ValueError(f"The minimum mass ratio for full mass transfer effects (mratio_MT = {self.mratio_MT}) "
                                     + f"is smaller than 1. Please use a larger and reasonable value.")
            if self.mratio_cratering > self.mratio_MT:
                self.warn_func(f"Mass ratio for cratering to take place (mratio_cratering = {self.mratio_cratering}) "
                               + f"is larger than the minimum mass ratio for full mass transfer effects "
                               + f"(mratio_MT = {self.mratio_MT}).  mratio_cratering has been changed to "
                               + f"mratio_MT - 10.")
                self.mratio_cratering = max(1, self.mratio_MT - 10)
            if self.mratio_c2MT > self.mratio_MT:
                self.warn_func(f"Mass ratio to begin transition from cratering to mass transfer (mratio_c2MT = "
                               + f"{self.mratio_c2MT}) is smaller than the minimum mass ratio for cratering to happen "
                               + f"(mratio_MT = {self.mratio_MT}).  mratio_c2MT has been changed to "
                               + f"mratio_MT - 5.")
                self.mratio_c2MT = max(1, self.mratio_MT - 5)
            if self.mratio_c2MT < self.mratio_cratering:
                self.warn_func(f"Mass ratio to begin transition from cratering to mass transfer (mratio_c2MT = "
                               + f"{self.mratio_c2MT}) is larger than the minimum mass ratio for full mass transfer "
                               + f"effects (mratio_cratering = {self.mratio_cratering}).  mratio_c2MT has been "
                               + f"changed to mratio_cratering.")
                self.mratio_c2MT = self.mratio_cratering            
        else:
            self.chi_MT = self.chi # revert back to cratering

    def powerlaw_fragmentation_coeff(self):
        """ Calculate the coefficients needed to distribute fragments by a power law

        This function also includes the PRESCRIBED cratering effects and mass transfer effects and a smooth
        transition between these two effects.

        REFs: Birnstiel+2010, Windmark+2012
        """
        
        C_norm = np.zeros(self.Ng+2)
        tmp_Nk = np.tril(self.m_j**(-self.xi + 1), -1)                   # fragments of i into j (= i-1, ..., 0)
        C_norm[1:] = self.m[1:] / np.sum(tmp_Nk * self.m_j, axis=1)[1:]  # this only skip the first row, still i-1 to 0
        
        if False:
            """ Below are a few alternate options to distribute fragments """
            # A. this one somehow slows the program dramatically!!!
            tmp_Nk = np.tril(self.m_j**(-self.xi+1))                     # fragments of i into j (= i, ..., 0)
            C_norm = self.m / np.sum(tmp_Nk * self.m_j, axis=1)
            # B. this one also slows the program dramatically!!!
            tmp_Nk = np.tril(self.m_j**(-self.xi+1))                     # fragments of i into j (= i, ..., 1)
            C_norm[1:] = self.m[1:] / np.sum(tmp_Nk[:, 1:] * self.m_j[:, 1:], axis=1)[1:]
            tmp_Nk[:, 0] = 0
        
        frag_Nk = tmp_Nk * C_norm[:, np.newaxis]                         # how unit mass at i will be distributed to j
        
        # copy to local variables for simplicity and improve readability
        chi, chi_MT = self.chi, self.chi_MT
        mratio_cratering = self.mratio_cratering
        mratio_c2MT, mratio_MT = self.mratio_c2MT, self.mratio_MT

        idx_m_c = int(np.ceil(np.log10(mratio_cratering)/self.dlog_m))   # how many grid points for mratio_cratering
        idx_m_MT = int(np.ceil(np.log10(mratio_MT)/self.dlog_m))         # how many grid points for mratio_MT
        idx_ij_diff = self.mesh3D_i - self.mesh3D_j                      
        idx_i_MT = idx_ij_diff >= idx_m_MT                               # find ij for full MT (m_i/m_j > mratio_MT)
        idx_i_cratering = (idx_ij_diff >= idx_m_c) & (idx_ij_diff < idx_m_MT)  # find ij for cratering & transition
        idx_ji_diff = self.mesh3D_j - self.mesh3D_i
        idx_j_MT = idx_ji_diff >= idx_m_MT                               # find ji for full MT (m_j/m_i > mratio_MT)
        idx_j_cratering = (idx_ji_diff >= idx_m_c) & (idx_ji_diff < idx_m_MT)  # find ji for cratering & transition
        idx_i_too_large = self.mesh3D_i - self.mesh3D_j >= idx_m_c       
        idx_j_too_large = self.mesh3D_j - self.mesh3D_i >= idx_m_c       
        idx_ij_close = (~(idx_i_too_large ^ idx_j_too_large))            # find ij for complete fragmentation
        
        # ***** both frag case *****
        idx_both_frag = (self.mesh3D_i > self.mesh3D_k) & (self.mesh3D_j > self.mesh3D_k)

        tmp_idx = idx_both_frag & idx_i_MT
        self.gF[tmp_idx] = frag_Nk[self.mesh3D_j[tmp_idx], self.mesh3D_k[tmp_idx]] * (1 + chi_MT)

        tmp_idx = idx_both_frag & idx_i_cratering
        mi_over_mj = (self.m[self.mesh3D_i] / self.m[self.mesh3D_j])
        # using a cosine curve to smoothly transition from chi to chi_MT
        chi_cratering = ((chi + chi_MT) / 2
                         + (chi - chi_MT) / 2 * np.cos((mi_over_mj - mratio_c2MT)*np.pi / (mratio_MT - mratio_c2MT)))
        chi_cratering[mi_over_mj < mratio_c2MT] = chi
        self.gF[tmp_idx] = frag_Nk[self.mesh3D_j[tmp_idx], self.mesh3D_k[tmp_idx]] * (1 + chi_cratering[tmp_idx])

        tmp_idx = idx_both_frag & idx_j_MT
        self.gF[tmp_idx] = frag_Nk[self.mesh3D_i[tmp_idx], self.mesh3D_k[tmp_idx]] * (1 + chi_MT)

        tmp_idx = idx_both_frag & idx_j_cratering
        mj_over_mi = (self.m[self.mesh3D_j] / self.m[self.mesh3D_i])
        chi_cratering = ((chi + chi_MT) / 2 
                         + (chi - chi_MT) / 2 * np.cos((mj_over_mi - mratio_c2MT)*np.pi / (mratio_MT - mratio_c2MT)))
        chi_cratering[mj_over_mi < mratio_c2MT] = chi
        self.gF[tmp_idx] = frag_Nk[self.mesh3D_i[tmp_idx], self.mesh3D_k[tmp_idx]] * (1 + chi_cratering[tmp_idx])

        tmp_idx = idx_both_frag & idx_ij_close
        self.gF[tmp_idx] = (frag_Nk[self.mesh3D_i[tmp_idx], self.mesh3D_k[tmp_idx]] 
                            + frag_Nk[self.mesh3D_j[tmp_idx], self.mesh3D_k[tmp_idx]])

        # ***** i frag *****
        idx_i_frag = (self.mesh3D_i > self.mesh3D_k) & (self.mesh3D_j <= self.mesh3D_k)
        tmp_idx = idx_i_frag & idx_ij_close
        self.gF[tmp_idx] = frag_Nk[self.mesh3D_i[tmp_idx], self.mesh3D_k[tmp_idx]]

        # ***** j frag *****
        idx_j_frag = (self.mesh3D_i <= self.mesh3D_k) & (self.mesh3D_j > self.mesh3D_k)
        tmp_idx = idx_j_frag & idx_ij_close
        self.gF[tmp_idx] = frag_Nk[self.mesh3D_j[tmp_idx], self.mesh3D_k[tmp_idx]]

        # can't simplify this by self.gF[idx_ij_same] *= 0.5 b/c some self.gF[i][j][k] = 0 when i == j
        self.gF[self.idx_ij_same][self.gF[self.idx_ij_same] == 0.0] = 1.0
        self.gF[self.idx_ij_same] *= 0.5
        
        #self.lf = np.copy(self.ones3D) # should already be ones
        self.lF[idx_j_cratering] *= (chi_cratering[idx_j_cratering] * self.m[self.mesh3D_i[idx_j_cratering]]
                                     / self.m[self.mesh3D_j[idx_j_cratering]])
        self.lF[idx_j_MT] *= chi_MT * self.m[self.mesh3D_i[idx_j_MT]] / self.m[self.mesh3D_j[idx_j_MT]]
        self.lF[self.idx_jk_diff] = 0
        self.lF[self.idx_ij_same] *= 0.5

    def calculate_dv(self):
        """ Calculate the relative velocities between particles
        
        Currently, we only consider relative velocities from Brownian motions and gas turbulence.

        REFs: Ormel & Cuzzi 2007, Birnstiel+2010
        """

        # ***** Brownian motions *****
        self.dv_BM = np.sqrt(8 * self.cgs_k_B * self.T * self.m_sum_ij / (np.pi * self.m_prod_ij))

        # ***** turbulent relative velocities: three regimes *****
        u_gas_TM = self.c_s * np.sqrt(3 / 2 * self.alpha)
        Re = self.alpha * self.Sigma_g * self.sigma_H2 / (2 * self.cgs_mu)
        St_12 = 1 / 1.6 * Re ** (-1 / 2)
        self.St_12, self.Re = St_12, Re # mainly for debug use so far

        dv_TM = np.zeros([self.Ng + 2, self.Ng + 2])
        St_i = np.tile(np.atleast_2d(self.St).T, [1, self.Ng + 2])
        St_j = np.tile(self.St, [self.Ng + 2, 1])

        # first, tightly coupled particles
        St_tiny = (St_i < St_12) & (St_j < St_12)
        dv_TM[St_tiny] = Re ** (1 / 4) * abs(St_i[St_tiny] - St_j[St_tiny])
        # then, if the larger particle is in intermediate regime
        St_inter = ((St_i >= St_12) & (St_i < 1)) | ((St_j >= St_12) & (St_j < 1))
        St_star12 = np.maximum(St_i[St_inter], St_j[St_inter])
        epsilon = St_i[St_inter] / St_j[St_inter]
        epsilon[epsilon > 1] = 1 / epsilon[epsilon > 1]
        dv_TM[St_inter] = np.sqrt(St_star12) * np.sqrt(
            (2 * 1.6 - (1 + epsilon) + 2 / (1 + epsilon) * (1 / (1 + 1.6) + epsilon ** 3 / (1.6 + epsilon))))
        # third, if the larger particle is very large
        St_large = (St_i >= 1) | (St_j >= 1)
        dv_TM[St_large] = np.sqrt(1 / (1 + St_i[St_large]) + 1 / (1 + St_j[St_large]))

        if self.kwargs.get("W12_VD", False):
            # for debug use for now, simplify the turbulent relative velocities to two regimes
            # for both particles with St < 1, use the formula below (from Windmark+2012b)
            dv_TM[~St_large] = np.sqrt(np.maximum(St_i[~St_large], St_j[~St_large]) * 3.0)

        if self.feedback_flag:
            # reduce the turbulent relative velocities by the weighted midplane dust-to-gas density ratio
            # effectively, using alpha_D = alpha / (1 + <eps>)^K, and K defaults to 1
            # N.B., H_d are initialized to zeros, so this flag must be False in the first call
            u_gas_TM *= 1 / np.sqrt(min(self.FB_eps_cap, (1 +
                (self.sigma/self.H_d).sum()/self.sqrt_2_pi*self.dlog_a / self.rho_g0)**self.feedback_K))

        self.dv_TM = dv_TM * u_gas_TM
        # print(f"u_gas_TM={u_gas_TM:.3e}, St_12={St_12:.3e}, Re={Re:.3e}")

        # sum up all contributions to the relative velocities
        self.dv = np.sqrt(self.dv_BM ** 2 + self.dv_TM ** 2)
        # if you want to Gaussian smooth the 2D dv array, use this
        # sigma = [3, 3] # [sigma_y, sigma_x]
        # self.dv = sp.ndimage.filters.gaussian_filter(self.dv, sigma, mode='constant')

    def calculate_St(self):
        """ Calculate the Stokes number of particles (and dv) 
        
        REFs: Section 3.5 and Eqs. 10, 14, 57 in Birnstiel+2010
        """

        # first iteration, assume Re_d < 1 and then calculate St based on lambda_mpf
        if self.simple_St_flag:
            # only use Epstein regime
            self.St_regimes = np.ones_like(self.St)
            self.St = self.rho_m * self.a / self.Sigma_g * np.pi / 2
        else:
            # default: use Epstein regime + Stokes regime 1
            self.St_regimes[self.a < self.lambda_mpf * 9 / 4] = 1
            self.St_regimes[self.a >= self.lambda_mpf * 9 / 4] = 2
            self.St[self.St_regimes == 1] = (self.rho_m * self.a / self.Sigma_g * np.pi / 2)[self.St_regimes == 1]
            self.St[self.St_regimes == 2] = (self.sqrt_2_pi / 9 * (self.rho_m * self.sigma_H2 * self.a ** 2)
                                            / self.cgs_mu / self.H)[self.St_regimes == 2]
        
        self.calculate_dv()

        if not self.full_St_flag:
            return None
        
        """
        now include Stokes regime 3 and 4
        more iterations based on particle Reynolds-number which depends on relative velocities between solids and gas
        here we use dv[0] (relative velocity w.r.t. the smallest particle) as the solid-gas relative velocities
        TODO: find a better way to calculate the solid-gas relative velocities when amin is large
        """
        self.Re_d = 2 * self.a * self.dv[0] / self.nu_mol
        if self.Re_d[0] > 1 or self.St[0] > 2e-3:
            raise ValueError(
                "The smallest particle is too large to be a good approximation for calculating solid-gas relative "
                + "velocities.  You may consider setting Rubble.simple_St_flag = True to only use Epstein regime "
                + "and Stokes regime 1, neglecting Stokes regime 2 and 3.")
        if self.Re_d.max() > 1:
            tmp_St_regimes = np.copy(self.St_regimes)
            self.St_regimes[(self.Re_d >= 1) & (self.Re_d < 800)] = 3
            self.St_regimes[self.Re_d >= 800] = 4
            tmp_loop_count = 0
            while np.any(tmp_St_regimes != self.St_regimes):
                tmp_St_regimes = np.copy(self.St_regimes)
                self.St[self.St_regimes == 1] = (self.rho_m * self.a / self.Sigma_g * np.pi / 2)[self.St_regimes == 1]
                self.St[self.St_regimes == 2] = (2 * self.rho_m * self.a ** 2 / (9 * self.nu_mol * self.rho_g0) 
                                                 * self.c_s / self.H)[self.St_regimes == 2]
                self.St[self.St_regimes == 3] = (2 ** 0.6 * self.rho_m * self.a ** 1.6 
                                                 / (9 * self.nu_mol ** 0.6 * self.rho_g0 * self.dv[0] ** 0.4)
                                                * self.c_s / self.H)[self.St_regimes == 3]
                self.St[self.St_regimes == 4] = ((6 * self.rho_m * self.a / (self.rho_g0 * self.dv[0])) 
                                                 * self.c_s / self.H)[self.St_regimes == 4]
                self.calculate_dv()
                self.Re_d = 2 * self.a * self.dv[0] / self.nu_mol
                self.St_regimes[self.a < self.lambda_mpf * 9 / 4] = 1
                self.St_regimes[(self.a >= self.lambda_mpf * 9 / 4) & (self.Re_d < 1)] = 2
                self.St_regimes[(self.Re_d >= 1) & (self.Re_d < 800)] = 3
                self.St_regimes[self.Re_d >= 800] = 4
                tmp_loop_count += 1
                if tmp_loop_count > 5:
                    if np.count_nonzero(tmp_St_regimes != self.St_regimes) < min(10, int(self.Ng / 15)):
                        self.warn_func(
                            f"A few sizes failed to converge to self-consistent Stokes number after 4 iterations. "
                            + f"This usually happens around the transition point between Stokes regime 1(2) and 2(3). "
                            + f"No need to worry for now. The results won't change too much.")
                        break
                    else:
                        raise RuntimeError(
                            "A range of sizes failed to converge to self-consistent Stokes number after 5 iterations. "
                            + f"Please submit an issue to https://github.com/astroboylrx/Rubble")

    def calculate_kernels(self):
        """ Calculate the kernel for coagulation and fragmentation (ghost zones only takes mass)
        
        REFs: Brauer+2008, Birnstiel+2010, 2011, Windmark+2012
        """

        # velocity thresholds for bouncing and fragmentation
        u_b = self.kwargs.get('u_b', 5.0)       # in units of cm/s, ref for 5: Windmark+2012
        u_f = self.kwargs.get('u_f', 100.0)     # in units of cm/s, ref for 100: Birnstiel+2010

        if self.vel_dist_flag:
            # we can push a bit further to include the velocity distribution (Windmark+ A&A 544, L16 (2012))
            # P(v|v_rms) = np.sqrt(54/np.pi) * v**2 / v_rms**3 * np.exp(-3/2 * (v / v_rms)**2)
            # the indefinite integral: 
            #     int_P(v) = -np.sqrt(54/np.pi)/3 * np.exp(-3/2 * (v/v_rms)**2) * (v/v_rms) 
            #                + spsp.erf(np.sqrt(1.5) * v/v_rms)
            # definite integral:
            #     int_P(v,{0,w}) = -np.sqrt(54/np.pi)/3 * np.exp(-3/2 * (w/v_rms)**2) * (w/v_rms) 
            #                      + spsp.erf(np.sqrt(1.5) * w/v_rms)
            #     int_P(v,{w,infty}) = 1 + np.sqrt(54/np.pi)/3 * np.exp(-3/2 * (w/v_rms)**2) * (w/v_rms)
            #                          - spsp.erf(np.sqrt(1.5) * w/v_rms)
            # 
            # we can now write down p_f and p_c

            p_f = (1 + np.sqrt(54/np.pi)/3 * np.exp(-3/2 * (u_f/self.dv)**2) * (u_f/self.dv) 
                   - spsp.erf(np.sqrt(1.5) * u_f/self.dv))

            # we may try FURTHER and put a bouncing barrier in
            if self.bouncing_flag:
                p_c = (-np.sqrt(54/np.pi)/3 * np.exp(-3/2 * (u_b/self.dv)**2) * (u_b/self.dv) 
                       + spsp.erf(np.sqrt(1.5) * u_b/self.dv))
            else:
                p_c = 1 - p_f
        else:
            delta_u = 0.2 * u_f                     # transition width, ref for 0.2: Birnstiel+2011
            soften_u_f = u_f - delta_u              # p_f = 0 when du_ij < soften_u_f
            p_f = np.copy(self.zeros2D)             # set all to zero
            p_f[self.dv > u_f] = 1.0                # set where du_ij > u_f to 1
            p_f_mask = (self.dv > soften_u_f) & (self.dv < u_f)
            p_f[p_f_mask] = 1 - (u_f - self.dv[p_f_mask]) / delta_u  # set else values

            if self.bouncing_flag:
                p_c = np.copy(self.zeros2D)
                p_c[self.dv < u_b] = 1.0
            else:
                p_c = 1 - p_f

        if self.uni_gz_flag == True:
            # unidirectional ghost zones, where the left one also coagulate and the right one also fragment
            self.p_f.fill(0)
            self.p_f[1:, 1:] = p_f[1:, 1:]
            self.p_c.fill(0)
            self.p_c[:-1, :-1] = p_c[:-1, :-1]
            self.p_b.fill(0)
            self.p_b = (1 - self.p_f - self.p_c)
        elif self.uni_gz_flag == False:
            # set the probabilities to zero for any p_{ij} that involves m_0 and m_last (i.e., inactive ghost zones)
            self.p_f.fill(0)
            self.p_f[1:-1, 1:-1] = p_f[1:-1, 1:-1]
            self.p_c.fill(0)
            self.p_c[1:-1, 1:-1] = p_c[1:-1, 1:-1]
            self.p_b.fill(0)
            self.p_b[1:-1, 1:-1] = (1 - self.p_f - self.p_c)[1:-1, 1:-1]
        elif self.uni_gz_flag == 2:
            # set the probabilities to zero for any p_{ij} that involves m_last (i.e., active left + inactive right)
            self.p_f.fill(0)
            self.p_f[1:-1, 1:-1] = p_f[1:-1, 1:-1]
            self.p_c.fill(0)
            self.p_c[:-1, :-1] = p_c[:-1, :-1]
            self.p_b.fill(0)
            self.p_b[:-1, :-1] = (1 - self.p_f - self.p_c)[:-1, :-1]
        elif self.uni_gz_flag == 3:
            # set the probabilities to zero for any p_{ij} that involves m_0 (i.e., inactive left + active right)
            self.p_f.fill(0)
            self.p_f[1:, 1:] = p_f[1:, 1:]
            self.p_c.fill(0)
            self.p_c[1:-1, 1:-1] = p_c[1:-1, 1:-1]
            self.p_b.fill(0)
            self.p_b[1:, 1:] = (1 - self.p_f - self.p_c)[1:, 1:]
        elif self.uni_gz_flag == 4:
            # fully active ghost zones
            self.p_f.fill(0)
            self.p_f = p_f
            self.p_f[0, 0] = 0
            self.p_c.fill(0)
            self.p_c = p_c
            self.p_c[-1, -1] = 0
            self.p_b = 1 - self.p_f - self.p_c
        else:
            raise ValueError(f"uni_gz_flag must be one of: True (both gz active), False (both gz inactive), "
                             f"2 (active left gz + inactive right gz), or 3 (inactive left gz + active right gz).")
        
        # Note: since K is symmetric, K.T = K, and dot(K_{ik}, n_i) = dot(K_{ki}, n_i) = dot(n_i, K_{ik})
        self.kernel = self.dv * self.geo_cs     # general kernel = du_ij * geo_cs
        self.L = self.kernel * self.p_f         # frag kernel, L_ij
        self.K = self.kernel * self.p_c         # coag kernel, K_ij

        if self.f_mod_flag is True:
            # use the modulation function to limit the interactions between mass bins that have low particle numbers
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', '', RuntimeWarning)
                tmp_Nk = self.Nk * self.S_annulus
                self.f_mod = np.exp(-1 / tmp_Nk[:, np.newaxis] - 1 / tmp_Nk)
                # modify K and L directly since they'll be re-generated each time
                self.K = self.K * self.f_mod
                self.L = self.L * self.f_mod
        
        """
        RL new notes: b/c Q_ij = K_ij N_i N_j gives "the number of collisions per second between two particle species"
        when we consider the number loss of N_i collide N_i, the # of collision events reduces to Q_ij / 4.
        However, two particles are needed per event, so the number loss is Q_ij / 2
        
        we divide M_ijk into four parts as in Eq. 36 in Birnstiel+2010, below is the first term
        | we need to make all parts of M_ijk as 3D matrixes, for later convenience in implicit step
        | all the parts can be further calculated by self.Nk.dot(self.Nk.dot(M)) to reduce the j and then the i axis
        | PS: check out the doc of np.dot at https://numpy.org/doc/stable/reference/generated/numpy.dot.html
        |                                   ****************
        | N.B., vec.dot(ndarray) reduces the SECOND-TO-LAST axis dimension of ndarray
        |                                   ****************
        | e.g., dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m]), or dot(a, b)[k,m] = sum(a[:] * b[k,:,m])
        | b/c numpy treat the last two axis of b as a matrix, any higher dimensions are stacks of matrixes
        | you may try: M = np.arange(27).reshape([3, 3, 3]); v = np.array([0.5, 1.5, 2.5]); print(v.dot(M))
        | 
        | Alternatively, we can use np.tensordot(Nk, M, (0, 1)), which is equiv to Nk.dot(M)
        | but np.tensordot provides the possibility of computing dot product along specified axes
        """
        
        self.M1 = 0.5 * self.C * self.K[:, :, np.newaxis]       # multiply C and K with matching i and j
        self.M1[self.idx_ij_same] *= 0.5                        # less collision events in single particle species

        # self.Nk.dot(self.Nk.dot(self.M2)) is equiv to self.Nk * self.K.dot(self.Nk), the original explicit way
        self.M2 = self.K[:, :, np.newaxis] * self.ones3D
        self.M2[self.idx_jk_diff] = 0.0                         # b/c M2 has a factor of delta(j-k)
        self.M2[self.idx_ij_same] *= 0.5                        # less collision events in single particle species
        
        # the 3rd/4th term is from fragmentation
        if self.frag_flag:
            self.M3 = 0.5 * self.L[:, :, np.newaxis] * self.gF      # multiply gF and L with matching i and j
            #self.M3[self.mesh3D_i == self.mesh3D_j] *= 0.5         # self.gF already considered this 0.5 factor
            
            self.M4 = self.L[:, :, np.newaxis] * self.lF            # multiply lF and L with matching i and j
            #self.M4[self.idx_jk_diff] = 0.0                        # self.lF alrady considered this
            #self.M4[self.idx_ij_same] *= 0.5                       # self.lF alrady considered this
        else:
            self.M3.fill(0)
            self.M4.fill(0)
        
        # sum up all the parts
        self.M = self.M1 - self.M2 + self.M3 - self.M4
        
        # now convert to vertically integrated kernel
        self.tM = self.M / self.vi_fac[:, :, np.newaxis]

    def update_kernels(self, update_coeff=False):
        """ Update collisional kernels """

        if self.f_mod_flag is True and self.feedback_flag is False and self.cycle_count > 0:
            # use the modulation function to limit the interactions between mass bins that have low particle numbers
            # if w/o feedback effects, we don't need to go through the entire update_kernels procedure, only new f_mod
            # and new tM are needed. Thus, we squeeze them here
            if self.cycle_count == 1:
                # Previously, self.K/L has been over-written with *= self.f_mod. We need to re-generate them when going
                # into this branch for the first time
                self.L = self.kernel * self.p_f
                self.K = self.kernel * self.p_c
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', '', RuntimeWarning)
                tmp_Nk = self.Nk * self.S_annulus
                self.f_mod = np.exp(-1 / tmp_Nk[:, np.newaxis] - 1 / tmp_Nk)
            # for f_mod only, we cannot just modify K/L since they won't be re-generated
            tmp_K = self.K * self.f_mod
            tmp_L = self.L * self.f_mod

            self.M1 = 0.5 * self.C * tmp_K[:, :, np.newaxis]  # multiply C and K with matching i and j
            self.M1[self.idx_ij_same] *= 0.5  # less collision events in single particle species
            self.M2 = tmp_K[:, :, np.newaxis] * self.ones3D
            self.M2[self.idx_jk_diff] = 0.0  # b/c M2 has a factor of delta(j-k)
            self.M2[self.idx_ij_same] *= 0.5  # less collision events in single particle species

            if self.frag_flag:
                self.M3 = 0.5 * tmp_L[:, :, np.newaxis] * self.gF  # multiply gF and L with matching i and j
                self.M4 = tmp_L[:, :, np.newaxis] * self.lF  # multiply lF and L with matching i and j
            else:
                self.M3.fill(0)
                self.M4.fill(0)

            self.M = self.M1 - self.M2 + self.M3 - self.M4
            # now convert to vertically integrated kernel
            self.tM = self.M / self.vi_fac[:, :, np.newaxis]
            return None

        # first, update disk parameters if needed in the future
        # self._update_disk_parameters()

        # then, calculate solid properties
        self.calculate_St()

        if self.feedback_flag:
            # make a closer guess on the total midplane dust-to-gas density ratio
            self.H_d = self.H / np.sqrt(1 + self.St / self.alpha
                                        * min(self.FB_eps_cap, (1 + self.eps.sum())**self.feedback_K))
            self.eps = self.sigma * self.dlog_a / self.sqrt_2_pi / self.H_d / self.rho_g0
            self.eps_tot = self.eps.sum()
            self.calculate_St()
            # use root finding to self-consistently calculate the weighted midplane dust-to-gas ratio
            self._root_finding_tmp = self.sigma * self.dlog_a / self.sqrt_2_pi / self.rho_g0 / self.H

            """
            N.B.: in fact St also depends on eps through dv, but we assume St won't change too much (especially for
            particles with small St) and solve for eps that makes H_d and eps self-consistent.
            
            One caveat when eps_tot>>1 though: each time update_kernels() is called, St in the high mass tail varies,
            leading to *slightly* different St, dv, and thus *slightly* different kernels (K varies more, L less)
            """
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    tmp_sln = spopt.root_scalar(lambda et : et
                        - np.sum(self._root_finding_tmp * (1 + self.St / self.alpha
                                                           * min(self.FB_eps_cap, (1 + et)**self.feedback_K))**0.5),
                                            x0 = self.eps_tot, x1 = self.eps_tot*5,
                                            method='brentq', bracket=[min(self.FB_eps_cap*0.9, self.eps_tot/5),
                                                                      self.eps_tot*100])
                    if tmp_sln.converged and np.isfinite(tmp_sln.root):
                        self.H_d = self.H / np.sqrt(1 + self.St / self.alpha
                                                    * min(self.FB_eps_cap, (1 + tmp_sln.root)**self.feedback_K))
                        self.eps = self.sigma * self.dlog_a / self.sqrt_2_pi / self.H_d / self.rho_g0
                        self.eps_tot = self.eps.sum()
                        self.calculate_St()
                    else:
                        raise RuntimeError("solution not converged")
            except Exception as e:
                self.warn_func("Root finding for the total midplane dust-to-gas density ratio failed. "
                               + "\nError message: " + e.__str__()
                               + "\nFall back to use five iterations.")
                for idx_fb in range(4):
                    # if eps.sum() is already larger than the desired capped value, skip
                    #if (1 + self.eps.sum())**self.feedback_K > self.FB_eps_cap:
                    #    break
                    # on a second thought, even if eps.sum() > cap, one more loop is needed to make H_d consistent
                    # manually finding closer solution
                    self.H_d = self.H / np.sqrt(1 + self.St / self.alpha
                                                * min(self.FB_eps_cap, (1 + self.eps.sum())**self.feedback_K))
                    self.eps = self.sigma * self.dlog_a / self.sqrt_2_pi / self.H_d / self.rho_g0
                    self.eps_tot = self.eps.sum()
                    self.calculate_St()
        else:
            # using Eq. 28 in Youdin & Lithwick 2007, ignore the factor: np.sqrt((1 + self.St) / (1 + 2*self.St))
            self.H_d = self.H / np.sqrt(1 + self.St / self.alpha)
            # the ignored factor may further reduce H_d and lead to a larger solution to midplane eps_tot
            self.eps = self.sigma * self.dlog_a / self.sqrt_2_pi / self.H_d / self.rho_g0
            self.eps_tot = self.eps.sum()

        if self.kwargs.get("B10_Hd", False):
            # for debug use only, calculate the solid scale height based on Eq. 51 in Birnstiel+2010
            # this formula mainly focuses on smaller particles and results in super small H_d for St >> 1
            # which may be improved by adding limits from the consideration of KH effects
            self.H_d = self.H * np.minimum(1, np.sqrt(self.alpha / (np.minimum(self.St, 0.5) * (1 + self.St ** 2))))
            self.eps = self.sigma * self.dlog_a / self.sqrt_2_pi / self.H_d / self.rho_g0

        self.h_ss_ij = self.H_d ** 2 + self.H_d[:, np.newaxis] ** 2
        self.vi_fac = np.sqrt(2 * np.pi * self.h_ss_ij)

        if update_coeff:
            # currently, no flag requires updateing coagulation coeff
            # self.piecewise_coagulation_coeff()
            self.gF = np.zeros([self.Ng+2, self.Ng+2, self.Ng+2])  # reset the gain coeff
            self.lF = np.ones([self.Ng+2, self.Ng+2, self.Ng+2])   # reset the loss coeff
            self.init_powerlaw_fragmentation_coeff()
            self.powerlaw_fragmentation_coeff()

        # if needed, update how solid loss/supply should be calculated
        if self.dyn_env_flag:
            self.init_update_solids()

        # finally, re-evaluate kernels
        self.calculate_kernels()

    def flag_updated(self, flag_name):
        """ Update flag sensitive kernels if needed whenever a flag changes """

        if self.flag_activated:
            if flag_name in ["f_mod_flag", "feedback_flag"]:
                if self.f_mod_flag is True or self.feedback_flag is True:
                    self.static_kernel_flag = False
                else:
                    self.static_kernel_flag = True
            if flag_name in ["closed_box_flag", ]:
                if self.closed_box_flag is False:
                    self.init_update_solids()

            if flag_name in ["mass_transfer_flag", ]:
                self.update_kernels(update_coeff=True)
            else:
                self.update_kernels()
        else:
            # do not update kernel during the initial setup
            pass

    def show_flags(self):
        """ print all the flags to show status """

        print(f"{'frag_flag:':32}", self.frag_flag)
        print(f"{'bouncing_flag:':32}", self.bouncing_flag)
        print(f"{'mass_transfer_flag:':32}", self.mass_transfer_flag)
        print(f"{'f_mod_flag:':32}", self.f_mod_flag)
        print(f"{'vel_dist_flag:':32}", self.vel_dist_flag)
        print(f"{'simple_St_flag:':32}", self.simple_St_flag)
        print(f"{'full_St_flag:':32}", self.full_St_flag)
        print(f"{'feedback_flag:':32}", self.feedback_flag)
        print(f"{'uni_gz_flag:':32}", self.uni_gz_flag)
        print(f"{'closed_box_flag:':32}", self.closed_box_flag)
        print(f"{'dyn_env_flag:':32}", self.dyn_env_flag)
        print(f"{'static_kernel_flag:':32}", self.static_kernel_flag)
        print(f"{'debug_flag:':32}", self.debug_flag)
        print(f"{'flag_activated:':32}", self.flag_activated)

    def _user_setup(self, test='BMonly'):
        """ Customized setup for testing purposes (mainly for debugging)
        TODO: after code refactoring, we need to check if this still works
        """

        if test == 'BMonly':
            # for Brownian motion only (and only same-sized solids collide)
            self.simple_St_flag = True
            self.vel_dist_flag = False
            self.bouncing_flag = False
            self.frag_flag = False
            self.mass_transfer_flag = False
            self.dv = self.dv_BM
            self.dv[self.mesh2D_i != self.mesh2D_j] = 0
            self.calculate_kernels()
        elif test == 'BM+turb':
            # same-sized BM plus the relative turbulence velocities between same-sized solids
            self.simple_St_flag = True
            self.vel_dist_flag = False
            self.bouncing_flag = False
            self.frag_flag = False
            self.mass_transfer_flag = False
            self.dv = self.dv_BM
            self.dv[self.mesh2D_i != self.mesh2D_j] = 0
            dv_TM = self.c_s * np.sqrt(2 * self.alpha * self.St)
            dv_TM[self.St > 1] = self.c_s * np.sqrt(2 * self.alpha / self.St[self.St > 1])
            self.dv[self.mesh2D_i == self.mesh2D_j] = (self.dv[self.mesh2D_i == self.mesh2D_j]**2 + dv_TM**2)**0.5
            self.calculate_kernels()
        elif test == 'BM+turb+fulldv':
            # full collisions with BM and turbulence from a simpler description
            self.simple_St_flag = True
            self.vel_dist_flag = False
            self.bouncing_flag = False
            self.frag_flag = False
            self.mass_transfer_flag = False
            self.dv = self.dv_BM
            v_TM = self.c_s * np.sqrt(self.alpha * self.St)
            v_TM[self.St > 1] = self.c_s * np.sqrt(self.alpha / self.St[self.St > 1])
            self.dv += np.sqrt(v_TM**2 + v_TM[:, np.newaxis]**2)            
            self.calculate_kernels()
        elif test == 'constK':
            # constant kernel
            # manually set the number of m_0 to 1.0 (using Nk[1] b/c Nk[0] is for ghost zone)
            self.Nk[1] = self.kwargs.get("n_0", 1)
            self.sigma[1] = self.Nk[1] * 3 * self.m[1]
            self.Na[1] = self.Nk[1] * 3
            self._Sigma_d = self.get_Sigma_d(self.Nk)
            self.Sigma_d = self.get_Sigma_d(self.Nk)

            self.simple_St_flag = True
            self.vel_dist_flag = False
            self.bouncing_flag = False
            self.mass_transfer_flag = False
            self.frag_flag = False
            self.static_kernel_flag = True

            self.dv.fill(1.0)
            self.geo_cs.fill(self.kwargs.get("alpha_c", 1.0))
            self.vi_fac.fill(1.0)
            self.calculate_kernels()
        elif test == 'sumK':
            # sum kernel
            # manually set the number of m_0 to 1.0 (using Nk[1] b/c Nk[0] is for ghost zone)
            self.Nk[1] = self.kwargs.get("n_0", 1)
            self.sigma[1] = self.Nk[1] * 3 * self.m[1]
            self.Na[1] = self.Nk[1] * 3
            self._Sigma_d = self.get_Sigma_d(self.Nk)
            self.Sigma_d = self.get_Sigma_d(self.Nk)

            self.simple_St_flag = True
            self.vel_dist_flag = False
            self.bouncing_flag = False
            self.mass_transfer_flag = False
            self.frag_flag = False
            self.static_kernel_flag = True

            self.dv.fill(1.0)
            self.geo_cs = self.kwargs.get('beta_c', 1.0) * self.m_sum_ij
            self.vi_fac.fill(1.0)
            self.calculate_kernels()
        elif test == "productK":
            # product kernel
            # manually set the number of m_0 to 1.0 (using Nk[1] b/c Nk[0] is for ghost zone)
            self.Nk[1] = self.kwargs.get("n_0", 1)
            self.sigma[1] = self.Nk[1] * 3 * self.m[1]
            self.Na[1] = self.Nk[1] * 3
            self._Sigma_d = self.get_Sigma_d(self.Nk)
            self.Sigma_d = self.get_Sigma_d(self.Nk)

            self.simple_St_flag = True
            self.vel_dist_flag = False
            self.bouncing_flag = False
            self.mass_transfer_flag = False
            self.frag_flag = False
            self.static_kernel_flag = True
            self.uni_gz_flag = 4  # let the last ghost zone join coagulation with others

            self.dv.fill(1.0)
            self.geo_cs = self.kwargs.get('gamma_c', 1.0) * self.m_prod_ij
            self.vi_fac.fill(1.0)
            self.calculate_kernels()
        else:
            raise ValueError(f"Unknown test case: {test}")

    def init_update_solids(self):
        """ Initialized accretion info and calculate the loss/supply rate of solids """

        Mdot = self.kwargs.get('Mdot', 3e-9)         # in units of Msun/yr
        Raccu = self.kwargs.get('Raccu', 0.01)       # in units of AU
        Z = self.kwargs.get('Z', 0.01)               # dust-to-gas ratio

        self.S_annulus = 2 * np.pi * Raccu * 0.1*Raccu * (u.au.to(u.cm))**2
        self.Sigma_dot = Mdot*((u.Msun/u.yr).to(u.g/u.s)) / (2*np.pi * Raccu*(u.au.to(u.cm)) * self.H)
        
        a_min_in = self.kwargs.get('a_min_in', self.a[1])  # smallest solids drifting in, in units of cm
        a_max_in = self.kwargs.get('a_max_in', 10)         # largest solids drifting in, in units of cm
        if a_min_in > a_max_in:
            self.warn_func(f"The size range of solids drifting in seems off: {a_min_in} > {a_max_in}. Reversed.")
            a_min_in, a_max_in = a_max_in, a_min_in
        a_idx_i = np.argmin(abs(self.a - a_min_in))
        a_idx_f = np.argmin(abs(self.a - a_max_in))
        a_idx_i = max(a_idx_i, 1)
        a_idx_i = min(a_idx_i, self.Ng+1)
        a_idx_f = max(a_idx_f, 1)
        a_idx_i = min(a_idx_i, self.Ng+1)

        tmp_sigma = np.zeros(self.Ng+2)
        tmp_sigma[a_idx_i:a_idx_f+1] = self.a[a_idx_i:a_idx_f+1]**(0.5)  # MRN dist
        C_norm = Z * self.Sigma_dot / np.sum(tmp_sigma * self.dlog_a)
        self.dsigma_in = tmp_sigma * C_norm
        
        self.a_critD = self.kwargs.get('a_critD', 0.01)   # critical dust size that will be lifted, in units of cm

        if self.kwargs.get("TL02_loss", False):
            # RL: this seems not self-consistent; TODO: we should keep on formula for calculating H_p
            Hratio_TL02 = np.zeros(512)
            St_TL02 = np.logspace(-10, 5, 512)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'overflow encountered in exp')
                for i in range(512):
                    Hratio_TL02[i] = (spint.quad(lambda z : rho_d_TL02(1., z, 1., St_TL02[i], 1e-2), 1., np.inf)[0]
                                    / spint.quad(lambda z : rho_d_TL02(1., z, 1., St_TL02[i], 1e-2), 0., np.inf)[0])
            if self.a_critD < self.a[0]:
                Hratio_TL02[:] = 0
            elif self.a_critD > self.a[-1]:
                pass
            else:
                St_crit = self.St[np.argmax(self.a > self.a_critD)]
                Hratio_TL02[St_TL02 > St_crit] = 0
            # we should be able to safely extrapolate to further values
            self.Hratio_loss = spinterp.interp1d(St_TL02, Hratio_TL02, fill_value='extrapolate')

    def update_solids(self, dt):
        """ Update the particle distribution every time step if needed """

        if self.closed_box_flag: # we may use this to de-clutter other checks on this flag
            return None

        # first, calculate sigma loss due to accretion tunnels
        if self.kwargs.get("TL02_loss", False):
            self.dsigma_out = self.sigma / self.Sigma_g * self.Hratio_loss(self.St) * self.Sigma_dot
            # the old treatment is too slow and depends on numerical parameters, so deprecated
            # self.dsigma_out = self.sigma * self.dlog_a / self.Sigma_g * self.Hratio_loss(self.St) * self.Sigma_dot
        else:
            self.Hratio_loss = 1 - spsp.erf(1 / np.sqrt(2) / (self.H_d / self.H))
            self.Hratio_loss[self.a > self.a_critD] = 0
            self.dsigma_out = self.sigma / self.Sigma_g * self.Hratio_loss * self.Sigma_dot

        self.sigma -= np.minimum(self.dsigma_out * dt, self.sigma)

        # second, add dust supply from outer disk
        self.sigma += self.dsigma_in * dt

        # update Nk and total surface density
        self.Nk = self.sigma / (3 * self.m)
        self.Na = self.sigma / self.m
        self._Sigma_d = self.get_Sigma_d(self.Nk)  # numerical Sigma_d after discretization

    def get_Sigma_d(self, any_N):
        """ Integrate the vertically integrated surface number density per log mass to total dust surface density """

        return np.sum(any_N * 3 * self.m * self.dlog_a)

    def _get_dN(self, dt):
        """ Get the current dN for debug use """

        self.S = self.Nk.dot(self.Nk.dot(self.tM))  # M_ijk N_i N_j equals to M_jik N_i N_j
        self.J = self.Nk.dot(self.tM + np.swapaxes(self.tM, 0, 1))
        return spla.solve(self.I / dt - self.J.transpose(), self.S)

    def _one_step_implicit(self, dt):
        """ Group matrix solving part for future optimization 
        
        Notes: When mass grid spans more than 16 orders of magnitude, it gives ill-conditioned matrix warning
        (use np.linalg.cond to check the condition number) due to fragmentation (b/c the fragments of larger solids
        during a collision is distributed across the entire mass grid smaller than themselves).

        Solving an ill-conditioned linear system will still give "exact" results (numpy/scipy, mpmath, Mathematica, or
        Matlab all give the same results within machine/working precision), but you'll get wildly different results for
        small perturbations of A or b.

        In this function, we filter out the ill-conditioned matrix warning.
        """

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', '', spla.LinAlgWarning) # message needs to be empty to avoid warnings
            
            self.S = self.Nk.dot(self.Nk.dot(self.tM))  # M_ijk N_i N_j equals to M_jik N_i N_j
            self.J = self.Nk.dot(self.tM + np.swapaxes(self.tM, 0, 1))
            # when Ng >= 64, spla.solve becomes more efficient than inversing a matrix
            # self.dN = self.S.dot(spla.inv(self.I / dt - self.J))
            self.dN = spla.solve(self.I / dt - self.J.transpose(), self.S)

    def one_step_implicit(self, dt):
        """ Evolve the particle distribution for dt with one implicit step"""

        # ultimately, kernels needs to be updated every time step due to the changes on Sigma_g, T, Pi, kappa, etc.
        #if 'const_kernel' not in self.kwargs: # previous implementation
        if not self.static_kernel_flag:
            self.update_kernels()

        # if previous step successfully used dynamic dt, then continue using it
        # because it is likely that dyn_dt can work for a long time
        _dt = dt  # save a backup
        skip_this_cycle = False
        if self.dyn_dt_success:  # this will be False if dynamic_dt is never used
            dt = self.dyn_dt * self.s2y
        self._one_step_implicit(dt)
        tmp_rerr = abs(self.get_Sigma_d(self.dN) / self._Sigma_d)

        if self.dyn_dt_success and tmp_rerr > self.tol_dyndt:  # if fails, discontinue using it
            self.dyn_dt_success = False
            skip_this_cycle = True  # to avoid multiple trial in one cycle
            self.log_func(
                f"continue using dyn dt failed with rerr={tmp_rerr:.3e}; revert back to original dt.")
            dt = _dt
            self._one_step_implicit(dt)
            tmp_rerr = abs(self.get_Sigma_d(self.dN) / self._Sigma_d)

        if tmp_rerr > self.rerr_th:
            if tmp_rerr > self.rerr_th4dt:
                dt /= (tmp_rerr / self.rerr_th4dt) * 2
                self.log_func("dt adjusted temporarily to reduce the relative error: tmp dt = "+f"{dt / self.s2y:.3e}")
                self._one_step_implicit(dt)
                tmp_rerr = abs(self.get_Sigma_d(self.dN) / self._Sigma_d)
            if tmp_rerr > self.rerr_th: # rerr_th may < rerr_th4dt, so only a warning is given, no re-calculations
                self.warn_func("Relative error is somewhat large: sum(dSigma(dN))/Sigma_d = "
                               + f"{tmp_rerr:.3e}. Consider using a smaller timestep "
                               + "(a higher resolution mass grid usually won't help).")
        else:
            # only try dynamic_dt when both conditions are meet (to avoid back and forth)
            if self.dynamic_dt is True and self.dyn_dt_success is False and skip_this_cycle is False:
                dyn_dt = self.dyn_dt * self.s2y
                if tmp_rerr * (dyn_dt / dt) < self.tol_dyndt:
                    tmp_dN = np.copy(self.dN)
                    self._one_step_implicit(dyn_dt)
                    tmp_rerr = abs(self.get_Sigma_d(self.dN) / self._Sigma_d)
                    if tmp_rerr <= self.tol_dyndt:
                        dt = dyn_dt
                        self.dyn_dt_success = True
                        self.log_func(
                            f"dynamic dt used to speed up this run: dyn dt = {dt / self.s2y:.3e}; now continue with it")
                    else:
                        self.log_func(
                            f"dynamic dt attempt failed with rerr={tmp_rerr:.3e}, revert back to original dt.")
                        self.dN = tmp_dN

        # handle possible negative numbers (usually due to large dt)
        tmp_Nk = self.Nk + self.dN
        loop_count = 0
        while np.any(tmp_Nk < 0):
            tmp_Nk[(tmp_Nk < 0) & (self.sigma == 0)] = 0  # if nothing was there, set to zero

            # what we want is the conservation of mass, so instead of Nk, we should check mass in each bin
            # previous code on Nk may lead to larger relative error than expected
            tmp_mass = tmp_Nk * self.m # for checking purpose, no need to include factor 3 and self.dlog_a
            tiny_idx = (tmp_mass < 0) & (abs(tmp_mass) / tmp_mass.sum() < self.neg2o_th)
            tmp_Nk[tiny_idx] = 0  # if contribute little to total mass, reset to zero

            # if negative values persist
            if np.any(tmp_Nk < 0):
                # external solid supply may also cause this by creating a discontinuity
                if not self.closed_box_flag:
                    if not np.any(tmp_Nk * 3 * self.m + self.dsigma_in * dt < 0):
                        # negative values that can be canceled by supply may be ignored safely
                        # here a constant supply is assumed for now; may be improved in the future
                        break
                # if the issue is more severe, reduce dt
                if loop_count > self.negloop_tol:
                    raise RuntimeError(f"Reducing dt by 2^{self.negloop_tol} didn't prevent negative Nk.")
                dt /= 2.0
                self._one_step_implicit(dt)
                tmp_Nk = self.Nk + self.dN
                loop_count += 1
        if loop_count > 0:
            self.log_func("dt reduced to prevent negative Nk: new dt = " + f"{dt / self.s2y:.3e}")
        self.Nk = np.copy(tmp_Nk)

        self.sigma = self.Nk * 3 * self.m
        self.Na = self.Nk * 3
        self.dt = dt / self.s2y  # dt should be in units of sec, self.dt in units of yr
        self.rerr = (self.get_Sigma_d(self.Nk) - self._Sigma_d) / self._Sigma_d

    def enforce_mass_con(self):
        """ Enforce the conservation of total solid mass
        
        It is impossible to avoid rounding errors (see discussion in Garaud+2013).
        Here, we follow their method to enforce mass conservation but we enforce it every timestep.
        """

        self.Nk *= (self._Sigma_d / self.get_Sigma_d(self.Nk))
        self.sigma = self.Nk * 3 * self.m
        self.Na = self.Nk * 3

    def one_step_for_animation(self, dt):
        """ Evolve one step implicitly with post processing for animation use """

        self.one_step_implicit(dt)
        self.enforce_mass_con()
        if not self.closed_box_flag:
            self.update_solids(dt)

    def open_dump_file(self, mode):
        """ open file handler for data dump """

        if 'dat_file_name' in self.kwargs:
            self.dat_file = open(self.kwargs['dat_file_name'], mode)
        else:
            self.dat_file = open(self.dat_file_name, mode)

    def dump_ascii_data(self, first_dump=False):
        """ Dump run data to a file by appending """

        if first_dump:
            self.res4out = np.hstack([self.dlog_a, self.m, self.a])
            if 'dat_file_name' in self.kwargs:
                self.dat_file = open(self.kwargs['dat_file_name'], 'a')
            else:
                self.dat_file = open(self.dat_file_name, 'a')
        else:
            self.res4out = np.hstack([self.t, self.sigma, self.Nk])

        self.dat_file.write(" ".join('{:>12.6e}'.format(x) for x in self.res4out))
        self.dat_file.write('\n')

    def dump_bin_data(self, first_dump=False, open4restart=False):
        """ Dump run data to a file """

        if first_dump:
            self.res4out = np.hstack([self.dlog_a, self.m, self.a])
            if 'dat_file_name' in self.kwargs:
                self.dat_file = open(self.kwargs['dat_file_name'], 'wb')
                self.dat_file.write(self.Ng.to_bytes(4, 'little'))
                self.dat_file.write(self.res4out.tobytes())
                self.dat_file.close()
                self.dat_file = open(self.kwargs['dat_file_name'], 'ab')
            else:
                self.dat_file = open(self.dat_file_name, 'wb')
                self.dat_file.write(self.Ng.to_bytes(4, 'little'))
                self.dat_file.write(self.res4out.tobytes())
                self.dat_file.close()
                self.dat_file = open(self.dat_file_name, 'ab')
        else:
            self.res4out = np.hstack([self.t, self.sigma, self.Nk])
            self.dat_file.write(self.res4out.tobytes())

    def run(self, tlim, dt, out_dt, 
            burnin_dt=1/365.25, no_burnin=False, ramp_speed=1.01, dynamic_dt=False,
            out_log=True, dump='bin'):
        """ 
        Run simulations and dump results

        Parameters
        ----------
        tlim : float
            total run time, in units of year
        dt : float
            time step, in units of year
        out_dt : float
            time interval to dump data, in units of year
        burnin_dt : float
            transitional tiny timesteps at the beginning to burn-in smoothly, default: 1 day
        no_burnin : bool
            skip burn-in steps, useful for testing or if you want fixed dt, default: False
        ramp_speed : float
            how fast to ramp the initial burnin_dt up to dt, default: 1.01 (1% up each cycle after the 1st yr)
        dynamic_dt : bool
            whether to use a larger desired dt whenever relative error is within the input tolerance
            (i.e., self.rerr < self.tol_dyndt) to speed up this run
        out_log : bool
            whether or not to print log/info/warning/errors to file, default: True
            if set to False, these messages will be directed to "print"
        dump : str
            'bin' (default and recommended): dumping binary data
            'ascii': dumping ascii data (may produce large files)

        Notes
        -----
        1. Again, the results of solving the Smoluchowski equation are known to depend on resolution, initial setup,
        and the length of time step (and also the Python environment and the machine used).  Be sure to test
        different setups and find the converging behaviors.

        2. For restarting a simulation, you may directly run this function again with a larger tlim
        (N.B.: restarting simulations may produce results slightly different than direct longer simulations)
        """

        if out_log:
            logger = logging.getLogger("rlogger")
            if 'log_file_name' in self.kwargs:
                fh = logging.FileHandler(filename=self.kwargs['log_file_name'])
            else:
                fh = logging.FileHandler(filename=self.log_file_name)
            fh.setFormatter(logging.Formatter(fmt='%(asctime)s.%(msecs)03d|%(name)s|%(levelname)s| %(message)s',
                                              datefmt='%Y-%m-%d %H:%M:%S'))
            if self.debug_flag:
                logger.setLevel(logging.DEBUG)
            else:
                logger.setLevel(logging.INFO)
            logger.addHandler(fh)
            self.log_func, self.warn_func, self.err_func = logger.info, logger.warning, logger.error
        if dump == 'bin':
            dump_func = self.dump_bin_data  # recommended
        else:
            dump_func = self.dump_ascii_data

        s_time = time.perf_counter()
        if self.t > 0 and self.cycle_count > 0:
            self.log_func(f"===== Simulation restarts now =====")
            self.open_dump_file('ab' if dump == 'bin' else 'a')
            self.log_func(f"cycle={self.cycle_count}, t={self.t:.6e}, dt={dt:.3e}, rerr(Sigma_d)={self.rerr:.3e}")
            self.out_dt = out_dt
            self.next_out_t = self.t + self.out_dt
        else:
            self.log_func(f"===== Simulation begins now =====")
            dump_func(first_dump=True)
            self.cycle_count = 0
            self.log_func(f"cycle={self.cycle_count}, t={self.t:.6e}, dt={dt:.3e}, rerr(Sigma_d)={self.rerr:.3e}")
            self.out_dt = out_dt
            dump_func()
            self.next_out_t = self.t + self.out_dt

        # let's have some slow burn-in steps if not a restart
        # so we can enter the relatively-smooth profile gradually from the initial profile with discontinuities
        tmp_dt = burnin_dt
        if self.cycle_count == 0 and no_burnin is False:
            dt, burnin_dt = burnin_dt, dt
            while self.t + dt < min(tlim, 1):  # alternatively, we can use 4 yrs
                pre_step_t = time.perf_counter()
                self.one_step_implicit(dt * self.s2y)
                self.enforce_mass_con()
                if not self.closed_box_flag:
                    self.update_solids(self.dt * self.s2y) # dt could have changed
                self.t += self.dt
                self.cycle_count += 1
                post_step_t = time.perf_counter()
                self.log_func(f"cycle={self.cycle_count}, t={self.t:.6e}yr, dt={self.dt:.3e}, "
                              + f"rerr(Sigma_d)={self.rerr:.3e}, rt={post_step_t - pre_step_t:.3e}")
                if out_log and self.cycle_count % 100 == 0:
                    fh.flush()
                if self.t > self.next_out_t - dt / 2:
                    dump_func()
                    self.next_out_t = self.t + self.out_dt
            dt, burnin_dt = burnin_dt, dt

            if ramp_speed <= 1.0 or ramp_speed >= 5:
                self.warn_func(f"ramp_speed should be slightly larger than 1.0; got: {ramp_speed}; fall back to 1.01.")
                ramp_speed = 1.01
            # then let the burn-in dt gradually ramp up to match the desired dt
            # N.B., with increasing dt, we might miss the next_out_t and lose all the data 
            # if we only output within (out_t-dt/2, out_t+dt/2)
            while (self.t + tmp_dt < tlim) and (tmp_dt * ramp_speed < dt):
                tmp_dt *= ramp_speed
                pre_step_t = time.perf_counter()
                self.one_step_implicit(tmp_dt * self.s2y)
                self.enforce_mass_con()
                if not self.closed_box_flag:
                    self.update_solids(self.dt * self.s2y) # dt could have changed
                self.t += self.dt
                self.cycle_count += 1
                post_step_t = time.perf_counter()
                self.log_func(f"cycle={self.cycle_count}, t={self.t:.6e}yr, dt={self.dt:.3e}, "
                              + f"rerr(Sigma_d)={self.rerr:.3e}, rt={post_step_t - pre_step_t:.3e}")
                if out_log and self.cycle_count % 100 == 0:
                    fh.flush()
                if self.t > self.next_out_t - tmp_dt / 2:
                    dump_func()
                    self.next_out_t = self.t + self.out_dt

        # turn on dynamic dt if specified
        if dynamic_dt:
            if self.dyn_dt / dt < 4 or self.out_dt / dt < 4:
                self.warn_func(
                    f"Dynamic dt not enabled: dyn_dt/dt={self.dyn_dt / dt:.1f}, out_dt/dt={self.out_dt / dt:.1f}. "
                    f"One of them is smaller than 4, which won't result in a significant speed gain.")
            else:
                # continue to dyn_dt from tmp_dt
                if False:  #no_burnin is False:
                    while (self.t + tmp_dt < tlim) and (tmp_dt * ramp_speed < self.dyn_dt):
                        # tmp_dt += burnin_dt  # alternatively, we can use *= 1.01
                        tmp_dt *= ramp_speed
                        pre_step_t = time.perf_counter()
                        self.one_step_implicit(tmp_dt * self.s2y)
                        self.enforce_mass_con()
                        if not self.closed_box_flag:
                            self.update_solids(self.dt * self.s2y) # dt could have changed
                        self.t += self.dt
                        self.cycle_count += 1
                        post_step_t = time.perf_counter()
                        self.log_func(f"cycle={self.cycle_count}, t={self.t:.6e}yr, dt={self.dt:.3e}, "
                                      + f"rerr(Sigma_d)={self.rerr:.3e}, rt={post_step_t - pre_step_t:.3e}")
                        if out_log and self.cycle_count % 100 == 0:
                            fh.flush()
                        if self.t > self.next_out_t - tmp_dt / 2:
                            dump_func()
                            self.next_out_t = self.t + self.out_dt
                # now turn on dynamic_dt
                self.dynamic_dt = True
                if self.dyn_dt > self.out_dt:
                    self.warn_func(f"The desired dynamic dt is reduced to out_dt (={out_dt:.1f}) to secure data.")
                    self.dyn_dt = self.out_dt
                if self.tol_dyndt > min(self.rerr_th, self.rerr_th4dt):
                    self.tol_dyndt = min(self.rerr_th, self.rerr_th4dt)
                    self.warn_func("The relative error tolerance for using dynamic dt should be smaller than "
                                   + f"min(rerr_th, rerr_th4dt). Adjusted automatically to {self.tol_dyndt}.")

        # Now on to tlim with the input dt
        while self.t + dt < tlim:
            pre_step_t = time.perf_counter()
            self.one_step_implicit(dt * self.s2y)
            self.enforce_mass_con()
            if not self.closed_box_flag:
                self.update_solids(self.dt * self.s2y) # dt could have changed
            self.t += self.dt
            self.cycle_count += 1
            post_step_t = time.perf_counter()
            self.log_func(f"cycle={self.cycle_count}, t={self.t:.6e}yr, dt={self.dt:.3e}, "
                          + f"rerr(Sigma_d)={self.rerr:.3e}, rt={post_step_t - pre_step_t:.3e}")
            if out_log and self.cycle_count % 100 == 0:
                fh.flush()
            if self.t > self.next_out_t - dt / 2:  # to ensure we dump data
                dump_func()
                self.next_out_t = self.t + self.out_dt

        # last time step
        #
        # N.B.: If a large dt is used (large enough that rerr > machine precision, which is usually fine 
        # if rerr < 1e-6), the implicit scheme may give results beyond the linear regime, i.e., dN(dt)/dN(1s) 
        # is not approximately dt/1s. In that case, if the last step uses a very different dt (than the previous loop), 
        # the original quasi-equilibrium distribution will be modified to a different quasi-equilibrium state
        # (usually slightly but noticable). For now, let's just keep the original dt and run over tlim
        # self.dt = tlim - self.t

        pre_step_t = time.perf_counter()
        self.one_step_implicit(dt * self.s2y)
        self.enforce_mass_con()
        if not self.closed_box_flag:
            self.update_solids(self.dt * self.s2y) # dt could have changed
        self.t += self.dt
        self.cycle_count += 1
        post_step_t = time.perf_counter()

        dump_func()
        if self.t > self.next_out_t - dt / 2:  # advance next output time anyway
            self.next_out_t = self.t + self.out_dt

        self.log_func(f"cycle={self.cycle_count}, t={self.t:.6e}yr, dt={self.dt:.3e}, "
                          + f"rerr(Sigma_d)={self.rerr:.3e}, rt={post_step_t - pre_step_t:.3e}")
        elapsed_time = time.perf_counter() - s_time
        self.log_func(f"===== Simulation ends now =====\n" + '*' * 80
                      + f"\nRun stats:\n\tElapsed wall time: {elapsed_time:.6e} sec"
                      + f"\n\tCycles / wall second: {self.cycle_count / elapsed_time:.6e}")
        self.dat_file.close()
        if out_log:
            fh.flush()
            fh.close()
            logger.removeHandler(fh)
            logging.shutdown()
            self.log_func, self.warn_func, self.err_func = print, print, print