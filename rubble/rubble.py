from astropy import units as u
from astropy import constants as c
import numpy as np
import scipy.optimize as spopt
import logging
import time
import sys
import warnings
import torch


class Rubble:
    """ Simulate the local evolution of solid size distributions in Accretion Disks """

    class FlagProperty:
        """ Flag changes automatically update kernels when needed """

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

    """ flag list """
    debug_flag = FlagProperty("debug_flag", "to enable experimental features; default: False")
    # frag_flag = False is often used with customized kernel for testing/benchmarking purposes
    frag_flag = FlagProperty("frag_flag", "to enable fragmentation kernel; default: True")
    bouncing_flag = FlagProperty("bouncing_flag", "to enable bouncing outcomes besides coag/frag; default: True")
    vel_dist_flag = FlagProperty("vel_dist_flag", "to consider collisions with velocity distribution; default: True")
    mass_transfer_flag = FlagProperty("mass_transfer_flag", "to enable mass transfer in frag; default: True")
    simple_St_flag = FlagProperty("simple_St_flag", "to only consider Epstein regime for Stokes number; default: False")
    full_St_flag = FlagProperty("full_St_flag", "to consider all four regimes for Stokes number; default: False")
    uni_gz_flag = FlagProperty("uni_gz_flag", "to use unidirectional ghost zones; default: False")
    f_mod_flag = FlagProperty("f_mod_flag", "to use a modulation factor to disable bins with tiny Nk; default: True")
    feedback_flag = FlagProperty("feedback_flag", "to consider feedback effects on gas diffusivity; default: False")
    closed_box_flag = FlagProperty("closed_box_flag", "to use a closed box w/o solid loss or supply; default: True")
    dyn_env_flag = FlagProperty("dyn_env_flag", "to consider dynamic/live disk environment; default: False")
    # legacy_parRe_flag: use dv(a, amin) as the dust-to-gas relative velocity for calculating particle Reynolds numbers
    #                    See notes in method _calculate_St for more details.
    legacy_parRe_flag = FlagProperty("legacy_parRe_flag",
                                     "to use 2*a*dv(a, amin)/nu_mol as the particle Reynolds number; default: False")
    windmark_flag = FlagProperty("windmark_flag",
                                 "to use Windmark+2012 mass-dependent collision model; default: False")

    """ methods """

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
        """

        self.log, self.warn, self.err = print, print, print  # set up logging
        self.kwargs = kwargs  # store all kwargs

        """ Step 0, choose device for torch tensor """
        # in case cupy functions are needed, in-place conversion can be done fast and straightforward
        self.default_dtype = self.kwargs.get("default_dtype", torch.float64)
        self.device = self.__init_torch_device()

        """ Step 1, build grid for size/mass distributions (symbol: a/m) """
        self.log_amin, self.log_amax, self.num_grid = np.log10(amin), np.log10(amax), num_grid
        self.num_dec = self.log_amax - self.log_amin                   # number of decades in size
        self.dlog_a = self.num_dec / (self.num_grid - 1)               # size step in log space
        self.comm_ratio = (10**self.num_dec)**(1 / (self.num_grid-1))  # the common ratio of the size sequence
        # one ghost zone is needed on each side (gz means ghost zone; l/r means left/right)
        self.log_algz, self.log_argz = self.log_amin-self.dlog_a, self.log_amax+self.dlog_a  # left/right ghost zone
        # N.B., make sure to use float64 and the correct device
        self.a = torch.logspace(self.log_algz, self.log_argz, self.num_grid+2,
                                device=self.device)                    # cell centers of size grid [cm]
        self.log_a = torch.log10(self.a)                               # log cell centers of size grid
        self.log_a_edge = torch.linspace(self.log_algz-self.dlog_a/2, self.log_argz+self.dlog_a/2, self.num_grid+3,
                                         device=self.device)
        self.a_edge = 10**self.log_a_edge                              # cell edges of size grid

        # establish the corresponding mass grid
        self.rho_m = rho_m                                             # material density of solids
        self.dlog_m = self.num_dec * 3 / (self.num_grid - 1)           # step in log space along mass
        self.m = 4*np.pi/3 * rho_m * self.a**3                         # cell center of mass grid [g]
        self.log_m = torch.log10(self.m)                               # log cell center of mass grid
        self.log_m_edge = 4*np.pi/3 * rho_m * self.a_edge**3           # log cell edge of mass grid
        self.m_edge = 4*np.pi/3 * rho_m * self.a_edge**3               # cell edge of mass grid

        # then grids on vertically integrated quantities and other dust properties
        self.sigma = torch.zeros_like(self.a)                          # surface density, [g/cm^2]
        self.dsigma_in = torch.zeros_like(self.a)                      # source term for sigma, [g/cm^2]
        self.dsigma_out = torch.zeros_like(self.a)                     # sink term for sigma, [g/cm^2]
        self.Nk = torch.zeros_like(self.a)                             # number density per log mass bin [1/cm^2]
        self.dN = torch.zeros_like(self.a)                             # dN per time step
        self.Na = torch.zeros_like(self.a)                             # number density per log mass bin [1/cm^2]
        self.St = torch.zeros_like(self.a)                             # Stokes number
        self.St_regimes = torch.zeros_like(self.a)                     # St regimes (Epstein; Stokes; 1<Re<800; Re>800)
        self.H_d = torch.zeros_like(self.a)                            # dust scale height
        self.eps = torch.zeros_like(self.a)                            # midplane dust-to-gas density ratio
        self.dv2gas = torch.zeros_like(self.a)                         # dust-to-gas relative velocity
        self.Re_d = torch.zeros_like(self.a)                           # particle Reynolds-number = 2 a u / nu_mol
        self.Hratio_loss = torch.zeros_like(self.a)                    # mass loss fraction due to accretion funnels

        """ Step 2, initialize scalar physical parameters """
        self.Sigma_d = Sigma_d                                         # total dust surface density [g/cm^2]
        self._Sigma_d = Sigma_d                                        # numerical Sigma_d after discretization
        self.q = q                                                     # index for size dist., dN/da propto a^{-q}
        self.p = (q + 2) / 3                                           # index for mass dist., i.e., dM/da propto a^{-p}
        self.s = q - 4                                                 # index of sigma per log(a)
        self.u_b = self.kwargs.get('u_b', 5.0)                         # velocity thresholds for bouncing [cm/s]
        self.u_f = self.kwargs.get('u_f', 100.0)                       # velocity thresholds for frag [cm/s]
        self.xi = self.kwargs.get('xi', 1.83)                          # index of fragment distribution
        self.chi = self.kwargs.get('chi', 1)                           # mass [m_projectile] excavated from target
        self.mratio_cratering = self.kwargs.get('mratio_cratering', 10)  # minimum mass ratio for cratering to happen
        self.idx_m_c = 0                                               # how many grid points for mratio_cratering
        self.chi_MT = self.kwargs.get('chi_MT', -0.1)                  # mass [m_projectile] transferred to target
        self.mratio_c2MT = self.kwargs.get('mratio_MT', 15)            # transition mass ratio from cratering to MT
        self.mratio_MT = self.kwargs.get('mratio_MT', 50)              # minimum mass ratio for full mass transfer
        self.idx_m_MT = 0                                              # how many grid points for martio_MT
        self.St_12 = 0                                                 # particles below this are tightly-coupled
        self.eps_tot = 0.01                                            # total midplane dust-to-gas density ratio
        self.FB_eps_cap = 100                                          # max eps to cap the feedback effects

        # Windmark+2012 collision model parameters (only used when windmark_flag=True)
        # REFs: Windmark, Birnstiel, Ormel & Dullemond, A&A 540, A73 (2012) [W12a]
        #       Windmark, Birnstiel, Ormel & Dullemond, A&A 544, L16  (2012) [W12b]
        # sticking/bouncing thresholds: dv_stick = (m_p/m_s)^{-5/18} [cm/s], Eq. 1-2 in W12a
        self.W12_m_s = self.kwargs.get('W12_m_s', 3.0e-12)             # sticking normalizing mass [g]
        self.W12_m_b = self.kwargs.get('W12_m_b', 3.3e-3)              # bouncing normalizing mass [g]
        # fragmentation: v_mu = (m/m_mu)^{-gamma} [cm/s], Eq. 7 in W12a
        self.W12_gamma = self.kwargs.get('W12_gamma', 0.16)            # frag threshold exponent
        self.W12_m_10 = self.kwargs.get('W12_m_10', 3.67e7)            # frag onset mass [g] (mu=1.0)
        self.W12_m_05 = self.kwargs.get('W12_m_05', 9.49e11)           # half-mass frag mass [g] (mu=0.5)
        # largest remnant fraction: mu(m,v) = C * m^alpha * v^beta, Eq. 8-10 in W12a
        self.W12_mu_alpha = np.log(2) / np.log(self.W12_m_10 / self.W12_m_05)  # = -0.068
        self.W12_mu_beta = self.W12_mu_alpha / self.W12_gamma                   # = -0.43
        self.W12_mu_C = self.W12_m_10 ** (-self.W12_mu_alpha)                   # = 3.27 [g^{-alpha}]
        # mass transfer efficiency: eps_ac = a_coeff + b_coeff * (v_{1.0,beitz}/v_{1.0}) * dv, Eq. 13 in W12a
        self.W12_eps_ac_a = self.kwargs.get('W12_eps_ac_a', -6.8e-3)   # intercept
        self.W12_eps_ac_b = self.kwargs.get('W12_eps_ac_b', 2.8e-4)    # slope [s/cm]
        self.W12_eps_ac_min = self.kwargs.get('W12_eps_ac_min', 0.0)    # min accretion efficiency (floor)
        self.W12_eps_ac_max = self.kwargs.get('W12_eps_ac_max', 0.5)   # max accretion efficiency
        self.W12_v10_beitz = 13.0                                      # frag onset for 4.1g particles [cm/s]
        # erosion: m_er/m_p = a * (m_p/m_0)^k * dv + b, Eq. 15 in W12a (calibrated by Teiser & Wurm)
        self.W12_ero_a = self.kwargs.get('W12_ero_a', 1.1e-5)          # erosion coefficient
        self.W12_ero_b = self.kwargs.get('W12_ero_b', -0.4)            # erosion offset
        self.W12_ero_k = self.kwargs.get('W12_ero_k', 0.15)            # erosion mass exponent
        self.W12_m_0 = self.kwargs.get('W12_m_0', 3.5e-12)             # monomer mass [g]
        # fragment power-law index: n(m)dm propto m^{-kappa} dm, Eq. 16 in W12a
        self.W12_kappa = self.kwargs.get('W12_kappa', 9/8)             # = 1.125
        # sticking-to-bouncing transition constants, Eq. 22-23 in W12a
        # p_c = 1 - k1 * log10(dv) - k2, with k2 mass-dependent (computed per pair)
        self.W12_k1 = (18.0 / 5) / np.log10(self.W12_m_b / self.W12_m_s)  # = 0.40
        self.W12_log_ms = np.log10(self.W12_m_s)
        self.W12_log_mb_over_ms = np.log10(self.W12_m_b / self.W12_m_s)

        # some constants
        self.sqrt_2_pi = np.sqrt(2 * np.pi)
        self.inv_sqrt_2 = 1 / np.sqrt(2)
        self.cgs_k_B = c.k_B.cgs.value
        self.cgs_sigma_sb = c.sigma_sb.cgs.value
        self.cgs_mu = 2.3 * c.m_p.cgs.value                            # mean molecular weight
        self.s2y = u.yr.to(u.s)                                        # ratio to convert seconds to years
        self.sigma_H2 = 2.0e-15                                        # cross section for H2 [cm^2] (what if T<70K?)
        self.S_annulus = 2*np.pi * 1.0 * 0.1 * (u.au.to(u.cm))**2      # surface an 0.1 AU wide annulus at 1 AU
        self.feedback_K = 1                                            # 1/(1 + <eps_tot>)^{K}, for feedback effects
        # zero out more parameters that are initialized in  __init_disk_parameters()
        self.Sigma_g, self.Sigma_dot, self.alpha, self.T = 0, 0, 0, 0
        self.Omega, self.H, self.rho_g0, self.lambda_mpf = 0, 0, 0, 0
        self.c_s, self.nu_mol, self.u_gas_TM, self.Re = 0, 0, 0, 0
        self.Mdot, self.nu_disk, self.Raccu, self.Zacc = 0, 0, 0, 0
        self.kappa, self.kappa_cap, self.delta_kappa_cap = 0, 0, 0
        self.coeff_kappa_geom, self.coeff_Q_e, dyn_env_burn_in = 0, 0, 0
        self.T_kappa = self.kwargs.get("T_kappa", "latent_heat")
        self.T_i, self.kappa_i, self.H2_C, self.sili_C, self.sili_SLH = 0, 0, 0, 0, 0
        self.Sigma_v, self.P_v_eq, self.P_v_eq0, self.T_a, self.gamma = 0, 0, 0, 0, 0
        self.dSigma_e, self.dSigma_c, self.dSigma_ec, self.rho_v0 = 0, 0, 0, 0
        self.dSigma_v2d, self.dSigma_d2v = torch.zeros_like(self.a), torch.zeros_like(self.a)
        self.dotQ_plus, self.temp_lock_time, self.acc_ramp_time = 0, 0, 0
        self.a2_o_m = self.a**2 / self.m
        self.bar2cgs = u.bar.to(u.g/u.cm/u.s**2)
        self.ev_explicit, self.ev_semi_implicit = False, True

        self.__init_disk_parameters()

        """ Step 3, initialize tensors for matrix calculations and indices manipulations """
        # following the subscript, use i as the highest dimension, j second, k third,
        # meaning the changes due to i happens along the axis=0, changes due to j on axis=1, due to k on axis=2
        # e.g, for arr = [1,2,3],                   then arr_i * arr_j will give arr_ij,
        # arr_i is [[1,1,1],    arr_j is [[1,2,3],       meaning arr_ij[i][j] is from m_i and m_j
        #           [2,2,2],              [1,2,3],
        #           [3,3,3]],             [1,2,3]]
        self.m_j = torch.tile(self.m, [self.num_grid+2, 1])
        self.m_i = self.m_j.T
        self.m_sum_ij = self.m + self.m[:, None]
        self.m_prod_ij = self.m * self.m[:, None]

        # only indexing='ij' will produce the same shape indices
        """ make some of them non-persistent to save memory
        tmp_index = torch.arange(self.num_grid+2, dtype=torch.long, device=self.device)
        self.mesh2D_i, self.mesh2D_j = torch.meshgrid(tmp_index, tmp_index, indexing='ij')
        self.mesh3D_i, self.mesh3D_j, self.mesh3D_k = torch.meshgrid(tmp_index, tmp_index, tmp_index, indexing='ij')
        self.idx_ij_same = self.mesh3D_i == self.mesh3D_j
        self.idx_jk_diff = self.mesh3D_j != self.mesh3D_k
        """
        self.idx_grid = torch.arange(self.num_grid + 2, dtype=torch.long, device=self.device)
        # RL: found that mesh3D_i/j/k basically occupy ZERO memory (instead, np.meshgrid will!)
        mesh3D_i, mesh3D_j, mesh3D_k = torch.meshgrid(self.idx_grid, self.idx_grid, self.idx_grid, indexing='ij')
        # RL: dtype of idx_ij_same etc. is torch.bool, so 1/8 smaller in cuda memory
        self.idx_ij_same = mesh3D_i == mesh3D_j
        self.idx_jk_diff = mesh3D_j != mesh3D_k

        # intermediate tensors used in simulation
        self.dv_BM = torch.zeros_like(self.m_j)                        # relative velocity due to Brownian motion
        self.dv_TM = torch.zeros_like(self.m_j)                        # relative velocity due to gas turbulence
        self.dv = torch.zeros_like(self.m_j)                           # total relative velocity, du_ij
        self.geo_cs = torch.pi * (self.a + self.a[:, None])**2         # geometrical cross section [cm^2]
        self.h_ss_ij = torch.zeros_like(self.m_j)                      # h_i^2 + h_j^2, ss means "sum of squared" [cm^2]
        self.vi_fac = torch.zeros_like(self.m_j)                       # vertically-integration factor, √(2pi h_ss_ij)
        self.p_c = torch.zeros_like(self.m_j)                          # the coagulation probability
        self.p_b = torch.zeros_like(self.m_j)                          # the bouncing probability
        self.p_f = torch.zeros_like(self.m_j)                          # the fragmentation probability

        self.K = torch.zeros_like(self.m_j)                            # the coagulation kernel
        self.L = torch.zeros_like(self.m_j)                            # the fragmentation kernel
        self.f_mod = torch.zeros_like(self.m_j)                        # modulation factor to disable bins with tiny Nk

        # C is the epsilon matrix to distribute coagulated mass
        self.C = torch.zeros(self.num_grid+2, self.num_grid+2, self.num_grid+2, device=self.device)
        self.gF = torch.zeros_like(self.C)                             # the gain coeff of power-law dist. of fragments
        self.lF = torch.ones_like(self.C)                              # the loss coeff of power-law dist. of fragments

        # ultimate matrixes used in the implicit step
        self.I = torch.eye(self.num_grid+2, device=self.device)  # identity matrix
        self.S = torch.zeros_like(self.a)                              # source func
        self.J = torch.zeros_like(self.m_j)                            # Jacobian of the source function
        self.M = torch.zeros_like(self.C)                              # kernel of the Smoluchowski equation
        self.tM = torch.zeros_like(self.C)                             # the vertically-integrated kernel, t: tilde

        """ Step 4, initialize numerical variables """
        self.t = 0                                                     # run time [yr]
        self.dt = 0                                                    # time step [yr]
        self.cycle_count = 0                                           # number of cycles
        self.rerr = 0                                                  # relative error in total dust surface density
        self.out_dt = 0                                                # time interval to output results
        self.next_out_t = 0                                            # next time to output results
        self.res4out = None                                            # stacked array for output
        self._root_finding_tmp = 0                                     # used in root finding

        self.rerr_th = self.kwargs.get('rerr_th', 1e-6)                # threshold for relative error to issue warnings
        self.rerr_th4dt = self.kwargs.get('rerr_th4dt', 1e-6)          # threshold for relative error to lower timestep
        self.neg2o_th = self.kwargs.get('neg2o_th', 1e-15)             # threshold (w.r.t. Sigma_d) to zero out Nk[Nk<0]
        self.negloop_tol = self.kwargs.get('negloop_tol', 10)          # max. No. of loops to half dt to avoid Nk<0

        self.dynamic_dt = False                                        # use a larger desired dt if possible
        self.dyn_dt = self.kwargs.get('dyn_dt', 1)                     # desired dt (only used if it speeds up runs)
        self.tol_dyndt = self.kwargs.get('tol_dyndt', 1e-7)            # tolerance for relative error to use dyn_dt
        self.dyn_dt_success = False                                    # if previous dyn_dt succeed, continue using it

        self.run_name = run_name                                       # name of this simulation run
        self.dat_file = None                                           # file handler for writing data
        self.dat_file_name = run_name + ".dat"                         # filename for writing data
        self.log_file = None                                           # file handler for writing logs
        self.log_file_name = run_name + ".log"                         # filename for writing logs

        """ Step 5, set flags """
        self.static_kernel_flag = self.kwargs.get('static_kernel', True)  # assuming kernel to be static
        self.flag_activated = False                                    # simply set flag values and skip flag_updated()
        self._flag_dict = {}                                           # an internal flag dict for bookkeeping
        self.debug_flag = self.kwargs.get('debug', False)              # to enable experimental features
        self.frag_flag = self.kwargs.get('frag', True)                 # to consider fragmentation
        self.bouncing_flag = self.kwargs.get('bouncing', True)         # to include bouncing
        self.vel_dist_flag = self.kwargs.get('VD', True)               # to include velocity distribution
        self.mass_transfer_flag = self.kwargs.get('MT', True)          # to calculate mass transfer
        self.simple_St_flag = self.kwargs.get('simSt', False)          # to only use Epstein regime
        self.full_St_flag = self.kwargs.get('fullSt', False)           # to consider all four regimes for Stokes number
        self.uni_gz_flag = self.kwargs.get('uni_gz', False)            # to use unidirectional ghost zones
        self.f_mod_flag = self.kwargs.get('f_mod', False)              # to modulate bins with tiny Nk
        self.feedback_flag = False                                     # to consider dust feedback on diffusivity
        self.closed_box_flag = self.kwargs.get('CB', True)             # to use a closed box (so no loss/supply)
        self.dyn_env_flag = self.kwargs.get('dyn_env', False)          # to consider non-static gas disk environment
        self.legacy_parRe_flag = self.kwargs.get("legacy_parRe", False)  # to use legacy particle Reynolds number
        self.windmark_flag = self.kwargs.get("windmark", False)        # to use Windmark+2012 collision model
        self.tmp_debug_flag = self.kwargs.get("tmp_debug", False)

        """ Step 6, prepare coefficients and more initializations """
        self.piecewise_coagulation_coeff()
        self.__init_powerlaw_fragmentation_coeff()
        self.powerlaw_fragmentation_coeff()
        self.distribute_solids()
        if self.closed_box_flag is False or self.dyn_env_flag is True:
            self.__init_update_solids()  # initialize accretion part, based on Mdot, Raccu, H, Z
            # this will update self.S_annulus, which will be used in self.update_kernels
        self.__init_kernel_constants()
        self.__init_calculate_St()
        # with pointers, calls will be directed to the correct functions
        self.calculate_St = self._calculate_St
        self.update_kernels = self._update_kernels
        self.update_kernels()

        self.flag_activated = True  # some flags may change static_kernel_flag
        self.feedback_flag = self.kwargs.get('FB', False)              # requires initialization of H_d, St, eps

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def __init_torch_device(self):
        """ initialize torch device (e.g., cuda:0 and num_threads) """

        torch.set_default_dtype(self.default_dtype)
        device = None  # None actually default to cpu
        if "device" in self.kwargs:  # e.g., if the user needs to specify which GPU to use
            device = self.kwargs.get("device")
            try:
                _ = torch.empty(2, device=device)
            except Exception as e:
                self.err("[device error]:", e)
                sys.exit(19)  # linux error code for "No such device"
        else:
            if torch.cuda.is_available() and not self.kwargs.get("disable_cuda", False):
                # torch.device('cuda') will use the default CUDA device (the same as cuda:0 in the default setup).
                # However, if you are using a context manager (e.g. with torch.cuda.device(1):),
                # torch.device('cuda') will refer to the specified device.
                device = torch.device('cuda')  # assume only using 1 GPU
            else:
                device = torch.device('cpu')

        if device in [None, torch.device('cpu')]:
            if 'num_threads' in self.kwargs:
                torch.set_num_threads(self.kwargs.get('num_threads'))

        self.log("Selected torch device: ", device, "; default_dtype: ", torch.get_default_dtype())

        # PyTorch optimized linalg at 1.13, where solving a matrix with high condition number yields low accuracy
        if torch.__version__ >= "1.13":
            self._get_dN = self._get_dN_after_torch_1_13
            self._one_step_implicit = self._one_step_implicit_after_torch_1_13
            self._torch_113_flag = True  # RL: was wrong, now fixed
        else:
            self._get_dN = self._get_dN_before_torch_1_13
            self._one_step_implicit = self._one_step_implicit_before_torch_1_13
            self._torch_113_flag = False

        return device

    def __init_disk_parameters(self):
        """ Set disk parameters, de-clutter other functions """

        self.Sigma_g = self.kwargs.get('Sigma_g', self.Sigma_d*100)    # total gas surface density [g/cm^2]
        self.Sigma_dot = 0.0                                           # accretion rate in surface density
        self.alpha = self.kwargs.get('alpha', 1e-3)                    # alpha-prescription, may != diffusion coeff
        self.T = self.kwargs.get('T', 280)                             # disk temperature [K]
        self.Omega = 5.631352229752323e-07                             # Keplerian orbital frequency [1/s] (@0.5au)
        self.H = self.kwargs.get('H', 178010309011.3974)               # gas scale height [cm] (2.088e11 if gamma=1.4)

        # more derivations
        self.rho_g0 = self.Sigma_g / (self.sqrt_2_pi * self.H)         # midplane gas density
        self.lambda_mpf = 1 / (self.rho_g0 / self.cgs_mu * self.sigma_H2)  # mean free path of the gas [cm]
        self.c_s = (self.cgs_k_B * self.T / self.cgs_mu) ** 0.5        # gas sound speed [cm/s] (1.1861e5 if gamma=1.4)
        self.nu_mol = 0.5 * np.sqrt(8/np.pi) * self.c_s * self.lambda_mpf  # molecular viscosity

        # needed for calculating relative velocities
        self.u_gas_TM = self.c_s * np.sqrt(3/2 * self.alpha)           # mean square turbulent gas velocity
        # gas Reynolds number (= ratio of turbulent viscosity, nu_t = alpha c_s H, over molecular viscosity)
        self.Re = self.alpha * self.Sigma_g * self.sigma_H2 / (2 * self.cgs_mu)
        self.St_12 = 1/1.6 * self.Re ** (-1/2)                         # critical Stokes number for tightly coupled

        # dust opacity and evaporation/condensation related
        self.kappa = self.kwargs.get('kappa', 1)                       # total opacity, in units of cm^2/g
        self.kappa_cap = self.kwargs.get('kappa_cap', 100)             # maximum kappa allowed
        # user input; it is possible sound speed is determined in other ways
        if 'input_c_s' in self.kwargs:
            self.c_s = self.kwargs['input_c_s']
        if 'input_mu' in self.kwargs:
            self.cgs_mu = self.kwargs['input_mu']

        # calculate coefficients used in update_disk_parameters
        self.coeff_kappa_geom = (3 / (4 * self.rho_m * self.a)) * self.dlog_a
        self.coeff_Q_e = 0.3 * 2 * np.pi * self.a / 0.29
        self.delta_kappa_cap = self.kwargs.get("delta_kappa_cap", 0.01)
        self.dyn_env_burn_in = self.kwargs.get('dyn_env_burn_in', 10)  # burn in time (yrs) before activate dyn_env
        self.temp_lock_time = self.kwargs.get("temp_lock_time", 10)    # burn in time (yrs) before allowing T to change
        self.acc_ramp_time = self.kwargs.get("acc_ramp_time", 0)       # burn in time (yrs) to ramp up to Mdot linearly
        self.T_kappa = self.kwargs.get("T_kappa", "power_law")
        if self.T_kappa == "latent_heat":
            self.T_i = self.T
            self.kappa_i = self.kappa
            self.Sigma_g_i = self.Sigma_g
            self.H2_C = 1.7e8  # erg / g / K
            self.sili_C = 1.25e7  # erg / g / K
            self.sili_SLH = 2.0e11  # 1.5e11  # erg / g
            self.Omega = ((c.G * c.M_sun / (self.kwargs.get('Raccu', 0.1) * c.au)**3)**0.5).value
            self.nu_disk = self.alpha * c.R.cgs.value * self.T / (2.3 * self.Omega)

    def update_disk_parameters(self):
        """ Update disk parameters if they depend on dust properties (e.g., opacity)
            N.B.: This function should be called after some burn in evolution such
                  that dust distribution is far from monodisperse (see dyn_env_burn_in).
        """

        # Step 1, update kappa using current disk parameters (based on Yi-Xian's derivations)
        old_kappa = self.kappa
        kappa_geom = self.coeff_kappa_geom * (self.sigma / self.Sigma_g)
        Q_e = torch.minimum(self.coeff_Q_e * torch.scalar_tensor(self.T), torch.scalar_tensor(2.0))
        self.kappa = torch.sum(kappa_geom * Q_e).item()
        # to avoid sudden huge change, we limit how far away kappa gets -- 1% at most by default
        self.kappa = min((1 + self.delta_kappa_cap) * old_kappa, self.kappa)
        self.kappa = max((1 - self.delta_kappa_cap) * old_kappa, self.kappa)
        # apply kappa limiter
        self.kappa = min(self.kappa, self.kappa_cap)
        # add gas kappa (does not help when T->0 due to Z->0)
        self.kappa += 1e-8 * self.rho_g0**(2/3) * self.T**(3)

        # Step 2, update disk parameters that depends on kappa
        if self.T_kappa == "power_law":
            # here we assume T propto kappa^(1/5) for a radiative disk profile
            kappa_ratio = (self.kappa / old_kappa)
            self.T *= kappa_ratio**(0.2)
            self.H *= kappa_ratio**(0.1)
            self.c_s *= kappa_ratio**(0.1)
            self.rho_g0 *= kappa_ratio ** (-0.3)
            self.Sigma_g *= kappa_ratio**(-0.2)
            self.a_critD *= kappa_ratio**(-0.1)  # propto 1/c_s if letting dR/R = H/R
        elif self.T_kappa == "latent_heat":
            # alternatively, consider latent heat
            """
            try:
                eq_heat_trans = lambda T: c.sigma_sb.cgs.value / self.Sigma_g**2 * (self.T_i**4/self.kappa_i - T**4/self.kappa) - (self.H2_C + (self._Sigma_d/self.Sigma_g) * self.sili_SLH/20 * np.exp(-(T - 2000)**2/20**2)) * (T-self.T) / (self.dt * self.s2y)
                tmp_sln = spopt.root_scalar(eq_heat_trans, x0=self.T-10, x1=self.T+10, method='brentq', bracket=[1000, 3000])
                if tmp_sln.converged and np.isfinite(tmp_sln.root):
                    dT = tmp_sln.root - self.T
                else:
                    raise RuntimeError("solution not converged")
            except Exception as e:
                self.warn_func("failed to find dT, fall back to bad method")
            """

            #dT = c.sigma_sb.cgs.value / self.Sigma_g * (self.T_i**4 * self.Sigma_g * self.kappa_i/(1 + self.Sigma_g**2 * self.kappa_i**2) - self.T**4 * self.Sigma_g * self.kappa/(1 + self.Sigma_g**2 * self.kappa**2)) / (self.H2_C + (self._Sigma_d/self.Sigma_g) * self.sili_SLH/20 * np.exp(-(self.T - 2000)**2/20**2) ) * (self.dt * self.s2y)

            #dT = (c.sigma_sb.cgs.value
            #      * (self.T_i**4 * self.Sigma_g * self.kappa_i/(1 + self.Sigma_g_i**2 * self.kappa_i**2)
            #         - self.T**4 * self.Sigma_g * self.kappa/(1 + self.Sigma_g**2 * self.kappa**2))
            #      * (self.dt * self.s2y)
            #      - (self.dSigma_ec * self.sili_SLH)) / (self.Sigma_g * self.H2_C)

            # set temp_lock_time to infinity to test if Sigma_v goes to an equilibrium if T does not move
            if self.t >= self.temp_lock_time:
                if self.dotQ_plus == 0:
                    self.nu_disk = self.alpha * c.R.cgs.value * self.T / (2.3 * self.Omega)
                    self.dotQ_plus = 2 * self.cgs_sigma_sb * self.T**4 * self.kappa/2 / (
                            1 + (self.Sigma_g*self.kappa/2) ** 2)

                    self.warn(f"Temperature lock deactivated. From disk model, dotQ_plus ="
                              f" {2.25 * self.nu_disk * self.Omega**2:.3e}. From the current "
                              f"dust distribution, kappa={self.kappa:.3e}, dotQ_plus={self.dotQ_plus:.3e}, "
                              f"which will be scaled with T.")

                #dT = (2.25 * self.nu_disk * self.Omega**2
                dT = (self.dotQ_plus * (self.T / self.T_i)
                      - 2 * self.cgs_sigma_sb * self.T**4 * self.kappa/2 / (1 + (self.Sigma_g*self.kappa/2)**2)
                      ) * (self.dt * self.s2y) / self.H2_C \
                      - (self.dSigma_ec * self.sili_SLH / self.Sigma_g) / self.H2_C

                #self.log(f"kappa = {self.kappa:.2f}, dT = {dT:.2e}")
                # limiter
                if abs(dT / self.T) > 0.025:
                    dT = self.T * 0.025 * (dT / abs(dT))
                if (self.T + dT) / self.T_i < 0.1:
                    dT = 0.1 * self.T_i - self.T
                T_ratio = (self.T + dT) / self.T
                self.T += dT
                self.c_s *= T_ratio**(0.5)
                self.H *= T_ratio**(0.5)
                self.rho_g0 *= T_ratio ** (-0.5)
                self.a_critD *= T_ratio**(-0.5)

        # Step 3, update derived parameters that depends on disk parameters
        self.lambda_mpf = 1 / (self.rho_g0 / self.cgs_mu * self.sigma_H2)
        self.nu_mol = 0.5 * np.sqrt(8 / np.pi) * self.c_s * self.lambda_mpf

    def get_Sigma_d(self, any_N):
        """ Integrate the vertically integrated surface number density per log mass to total dust surface density """

        return torch.sum(any_N * 3 * self.m * self.dlog_a).item()

    def distribute_solids(self):
        """ Distribute the solids into all the grid by power law """

        if 'delta_dist' in self.kwargs:
            a_idx = torch.argmin(abs(self.a - self.kwargs['delta_dist']))
            if a_idx < 1:
                raise ValueError(f"delta distribution outside simulation domain, a_min on grid is {self.a[1]:.3e}")
            if a_idx > self.num_grid:
                raise ValueError(f"delta distribution outside simulation domain, a_max on grid is {self.a[-2]:.3e}")
            self.sigma[a_idx] = self.Sigma_d / self.dlog_a
        elif 'ranged_dist' in self.kwargs:
            a_idx_i = torch.argmin(abs(self.a - self.kwargs['ranged_dist'][0]))
            a_idx_f = torch.argmin(abs(self.a - self.kwargs['ranged_dist'][1]))
            if a_idx_f < a_idx_i:  # order reversed
                a_idx_i, a_idx_f = a_idx_f, a_idx_i
            if a_idx_i < 1:
                raise ValueError(f"a_min in ranged_dist is too small, a_min on grid is {self.a[1]:.3e}")
            if a_idx_i > self.num_grid:  # outside right boundary
                raise ValueError(f"a_min in ranged_dist is too large, a_max on grid is {self.a[-2]:.3e}")
            if a_idx_f > self.num_grid:  # outside right boundary
                raise ValueError(f"a_max in ranged_dist is too large, a_max on grid is {self.a[-2]:.3e}")
            if a_idx_f == a_idx_i:
                a_idx_f += 1
            tmp_sigma = torch.zeros_like(self.a)
            tmp_sigma[a_idx_i:a_idx_f + 1] = self.a[a_idx_i:a_idx_f + 1] ** (-self.s)
            C_norm = self.Sigma_d / torch.sum(tmp_sigma * self.dlog_a)
            self.sigma = tmp_sigma * C_norm
        elif 'input_dist' in self.kwargs:
            try:
                self.sigma = self.kwargs['input_dist']
                self.Sigma_d = self.get_Sigma_d(self.sigma / (3 * self.m))
            except Exception as e:
                self.warn("fail to take the input distribution, revert back to default. " + e.__str__())
                tmp_sigma = self.a[1:-1] ** (-self.s)
                C_norm = self.Sigma_d / torch.sum(tmp_sigma * self.dlog_a)
                self.sigma[1:-1] = tmp_sigma * C_norm
        else:
            tmp_sigma = self.a[1:-1] ** (-self.s)
            C_norm = self.Sigma_d / torch.sum(tmp_sigma * self.dlog_a)
            self.sigma[1:-1] = tmp_sigma * C_norm
        self.Nk = self.sigma / (3 * self.m)
        self.Na = self.sigma / self.m
        self._Sigma_d = self.get_Sigma_d(self.Nk)

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
        mesh2D_i, mesh2D_j = torch.meshgrid(self.idx_grid, self.idx_grid, indexing='ij')

        if not coag2nearest:
            # formula from Brauer+2008
            merger_n = torch.searchsorted(self.m, self.m_sum_ij)       # idx for m_n, searchsorted give where to insert
            merger_m = merger_n - 1                                    # idx for m_m,
            epsilon = torch.zeros_like(self.m_j)                       # epsilon, see reference
            epsilon2 = torch.zeros_like(self.m_j)                      # 1 - epsilon for non-ghost zone
            ngz_mask = merger_m <= self.num_grid                       # non-ghost-zone mask

            epsilon[merger_m > self.num_grid] = \
                self.m_sum_ij[merger_m > self.num_grid] / self.m[merger_m[merger_m > self.num_grid]]
            epsilon[ngz_mask] = \
                (self.m[merger_n[ngz_mask]] - self.m_sum_ij[ngz_mask]) \
                / (self.m[merger_n[ngz_mask]] - self.m[merger_m[ngz_mask]])

            epsilon2[ngz_mask] = 1 - epsilon[ngz_mask]

            self.C[mesh2D_i.flatten(), mesh2D_j.flatten(), merger_m.flatten()] = epsilon.flatten()
            self.C[mesh2D_i[ngz_mask], mesh2D_j[ngz_mask], merger_n[ngz_mask]] = epsilon2[ngz_mask]

        else:
            nth_right_edge = torch.searchsorted(self.m_edge, self.m_sum_ij)
            merger = nth_right_edge - 1                                # idx for m_nearest
            epsilon = torch.zeros_like(self.m_j)                       # epsilon, see reference
            ngz_mask = merger <= self.num_grid                         # non-ghost-zone mask

            epsilon[merger > self.num_grid] = self.m_sum_ij[merger > self.num_grid] / self.m[-1]
            epsilon[ngz_mask] = self.m_sum_ij[ngz_mask] / self.m[merger[ngz_mask]]

            merger[merger > self.num_grid] = self.num_grid + 1         # make sure k stays in bound
            self.C[mesh2D_i.flatten(), mesh2D_j.flatten(), merger.flatten()] = epsilon.flatten()

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def __init_powerlaw_fragmentation_coeff(self):
        """ Initialize numerical parameters needed for calculating coefficients for fragmentation kernels """

        # power law index of fragment distribution, ref: Brauer+2008
        if self.xi > 2:
            self.warn(f"The power law index of fragment distribution, xi = {self.xi} > 2, which means most mass "
                      f"goes to smaller fragments. Make sure this is what you want.")

        # mass excavated from the larger particle due to cratering
        if self.chi >= 10:
            self.warn(f"The amount of mass excavated from target by projectile due to cratering (happends when "
                      f"mass ratio in a collision >= 10), in units of the projectile mass, is chi = {self.chi} "
                      f">= 10, which means a complete fragmentation also happen to the target sometimes. "
                      f"To keep it simple, chi has been reduced to 9.99.")
            self.chi = 9.99

        # mass transfer from the projectile to the target (N.B.: mass transfer is kind of reversed cratering)
        if self.mass_transfer_flag:
            if self.chi_MT > 0:  # using negative values to add mass back to targets
                self.chi_MT *= -1
            if self.chi_MT < -1.0:
                self.warn(f"Mass transfer fraction {self.chi_MT} cannot exceed 1.0. The maximum mass available to "
                          f"transfer is 100% in units of the projectile mass.  To make it physical, chi_MT has "
                          f"been reduced to 0.99")
                self.chi_MT = -0.99
            if self.mratio_MT < 20:
                self.warn(f"The minimum mass ratio for full mass transfer effects (mratio_MT = {self.mratio_MT}) "
                          f"is smaller than 20. Mass transfer is more likely to happen when the mass difference"
                          f" is larger. Be careful here.")
                if self.mratio_MT < 1:
                    raise ValueError(
                        f"The minimum mass ratio for full mass transfer effects (mratio_MT = {self.mratio_MT}) "
                        f"is smaller than 1. Please use a larger and reasonable value.")
            if self.mratio_cratering > self.mratio_MT:
                self.warn(f"Mass ratio for cratering to take place (mratio_cratering = {self.mratio_cratering}) "
                          f"is larger than the minimum mass ratio for full mass transfer effects "
                          f"(mratio_MT = {self.mratio_MT}).  mratio_cratering has been changed to mratio_MT - 10.")
                self.mratio_cratering = max(1, self.mratio_MT - 10)
            if self.mratio_c2MT > self.mratio_MT:
                self.warn(f"Mass ratio to begin transition from cratering to mass transfer (mratio_c2MT = "
                          f"{self.mratio_c2MT}) is smaller than the minimum mass ratio for cratering to happen "
                          f"(mratio_MT = {self.mratio_MT}).  mratio_c2MT has been changed to mratio_MT - 5.")
                self.mratio_c2MT = max(1, self.mratio_MT - 5)
            if self.mratio_c2MT < self.mratio_cratering:
                self.warn(f"Mass ratio to begin transition from cratering to mass transfer (mratio_c2MT = "
                          f"{self.mratio_c2MT}) is larger than the minimum mass ratio for full mass transfer "
                          f"effects (mratio_cratering = {self.mratio_cratering}).  mratio_c2MT has been "
                          f"changed to mratio_cratering.")
                self.mratio_c2MT = self.mratio_cratering
        else:
            self.chi_MT = self.chi  # revert to cratering

        # how many grid points for mratio_cratering and for mratio_MT
        self.idx_m_c = int(np.ceil(np.log10(self.mratio_cratering) / self.dlog_m))
        self.idx_m_MT = int(np.ceil(np.log10(self.mratio_MT) / self.dlog_m))

    def powerlaw_fragmentation_coeff(self):
        """ Calculate the coefficients needed to distribute fragments by a power law

        This function also includes the PRESCRIBED cratering effects and mass transfer effects and a smooth
        transition between these two effects.

        This function went through code refactoring to control the peak memory usage on GPU, where code fragments
        have been shuffled, reducing the readability.  One may go back to old commits to read the original code.

        REFs: Birnstiel+2010, Windmark+2012
        """

        C_norm = torch.zeros_like(self.a)
        tmp_Nk = torch.tril(self.m_j**(-self.xi + 1), -1)              # fragments of i into j (= i-1, ..., 0)
        C_norm[1:] = self.m[1:] / torch.sum(tmp_Nk * self.m_j, 1)[1:]  # this only skip the first row, still i-1 to 0

        if False:
            """ Below are a few alternate options to distribute fragments """
            # A. this one somehow slows the program dramatically!!!
            tmp_Nk = torch.tril(self.m_j**(-self.xi+1))                # fragments of i into j (= i, ..., 0)
            C_norm = self.m / torch.sum(tmp_Nk * self.m_j, 1)
            # B. this one also slows the program dramatically!!!
            tmp_Nk = torch.tril(self.m_j**(-self.xi+1))                # fragments of i into j (= i, ..., 1)
            C_norm[1:] = self.m[1:] / torch.sum(tmp_Nk[:, 1:] * self.m_j[:, 1:], 1)[1:]
            tmp_Nk[:, 0] = 0

        frag_Nk = tmp_Nk * C_norm[:, None]                             # how unit mass at i will be distributed to j

        # localize variables for simplicity and improve readability
        chi, chi_MT = self.chi, self.chi_MT
        mratio_cratering = self.mratio_cratering
        mratio_c2MT, mratio_MT = self.mratio_c2MT, self.mratio_MT
        idx_m_c, idx_m_MT = self.idx_m_c, self.idx_m_MT

        mesh3D_i, mesh3D_j, mesh3D_k = torch.meshgrid(self.idx_grid, self.idx_grid, self.idx_grid, indexing='ij')

        # ***** both frag case *****
        idx_both_frag = (mesh3D_i > mesh3D_k) & (mesh3D_j > mesh3D_k)

        idx_ij_diff = mesh3D_i - mesh3D_j
        tmp_idx = idx_both_frag & (idx_ij_diff >= idx_m_MT)
        self.gF[tmp_idx] = frag_Nk[mesh3D_j[tmp_idx], mesh3D_k[tmp_idx]] * (1 + chi_MT)

        tmp_idx = idx_both_frag & ((idx_ij_diff >= idx_m_c) & (idx_ij_diff < idx_m_MT))
        mi_over_mj = (self.m[mesh3D_i] / self.m[mesh3D_j])  # will have a RAM spike, but reserved can be used later

        # using a cosine curve to smoothly transition from chi to chi_MT
        chi_cratering = ((chi + chi_MT) / 2 + (chi - chi_MT) / 2
                         * torch.cos((mi_over_mj - mratio_c2MT) * torch.pi / (mratio_MT - mratio_c2MT)))
        chi_cratering[mi_over_mj < mratio_c2MT] = chi
        self.gF[tmp_idx] = frag_Nk[mesh3D_j[tmp_idx], mesh3D_k[tmp_idx]] * (1 + chi_cratering[tmp_idx])
        del idx_ij_diff, mi_over_mj, tmp_idx

        idx_ji_diff = mesh3D_j - mesh3D_i
        idx_j_MT = idx_ji_diff >= idx_m_MT                             # find ji for full MT (m_j/m_i > mratio_MT)
        idx_j_cratering = (idx_ji_diff >= idx_m_c) & (idx_ji_diff < idx_m_MT)  # find ji for cratering & transition
        del idx_ji_diff
        tmp_idx = idx_both_frag & idx_j_MT
        self.gF[tmp_idx] = frag_Nk[mesh3D_i[tmp_idx], mesh3D_k[tmp_idx]] * (1 + chi_MT)

        tmp_idx = idx_both_frag & idx_j_cratering
        mj_over_mi = (self.m[mesh3D_j] / self.m[mesh3D_i])  # will have a RAM spike, but reserved can be used later
        chi_cratering = ((chi + chi_MT) / 2 + (chi - chi_MT) / 2
                         * torch.cos((mj_over_mi - mratio_c2MT) * torch.pi / (mratio_MT - mratio_c2MT)))
        chi_cratering[mj_over_mi < mratio_c2MT] = chi
        self.gF[tmp_idx] = frag_Nk[mesh3D_i[tmp_idx], mesh3D_k[tmp_idx]] * (1 + chi_cratering[tmp_idx])

        # self.lF = np.copy(self.ones3D)  # should already be ones
        self.lF[idx_j_cratering] *= (chi_cratering[idx_j_cratering] * self.m[mesh3D_i[idx_j_cratering]]
                                     / self.m[mesh3D_j[idx_j_cratering]])
        self.lF[idx_j_MT] *= chi_MT * self.m[mesh3D_i[idx_j_MT]] / self.m[mesh3D_j[idx_j_MT]]
        self.lF[self.idx_jk_diff] = 0
        self.lF[self.idx_ij_same] *= 0.5
        del idx_j_MT, idx_j_cratering, mj_over_mi, tmp_idx

        idx_i_too_large = mesh3D_i - mesh3D_j >= idx_m_c
        idx_j_too_large = mesh3D_j - mesh3D_i >= idx_m_c
        idx_ij_close = (~(idx_i_too_large ^ idx_j_too_large))          # find ij for complete fragmentation
        del idx_i_too_large, idx_j_too_large

        tmp_idx = idx_both_frag & idx_ij_close
        self.gF[tmp_idx] = (frag_Nk[mesh3D_i[tmp_idx], mesh3D_k[tmp_idx]]
                            + frag_Nk[mesh3D_j[tmp_idx], mesh3D_k[tmp_idx]])

        # ***** i frag *****
        idx_i_frag = (mesh3D_i > mesh3D_k) & (mesh3D_j <= mesh3D_k)
        tmp_idx = idx_i_frag & idx_ij_close
        self.gF[tmp_idx] = frag_Nk[mesh3D_i[tmp_idx], mesh3D_k[tmp_idx]]

        # ***** j frag *****
        idx_j_frag = (mesh3D_i <= mesh3D_k) & (mesh3D_j > mesh3D_k)
        tmp_idx = idx_j_frag & idx_ij_close
        self.gF[tmp_idx] = frag_Nk[mesh3D_j[tmp_idx], mesh3D_k[tmp_idx]]
        del idx_ij_close, idx_i_frag, idx_j_frag, tmp_idx

        # can't simplify this by self.gF[idx_ij_same] *= 0.5 b/c some self.gF[i][j][k] = 0 when i == j
        self.gF[self.idx_ij_same][self.gF[self.idx_ij_same] == 0.0] = 1.0
        self.gF[self.idx_ij_same] *= 0.5

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def calculate_dv(self):
        """ Calculate the relative velocities between particles

        Currently, we only consider relative velocities from Brownian motions and gas turbulence.

        REFs: Ormel & Cuzzi 2007, Birnstiel+2010
        """

        # ***** Brownian motions *****
        self.dv_BM = torch.sqrt(8 * self.cgs_k_B * self.T * self.m_sum_ij / (torch.pi * self.m_prod_ij))

        # ***** turbulent relative velocities: three regimes *****
        # localize variables for simplicity and improve readability
        u_gas_TM, Re, St_12 = self.u_gas_TM, self.Re, self.St_12
        St_j = torch.tile(self.St, [self.num_grid+2, 1])
        St_i = St_j.T

        # first, tightly coupled particles
        St_tiny = (St_i < St_12) & (St_j < St_12)
        self.dv_TM[St_tiny] = Re ** (1 / 4) * abs(St_i[St_tiny] - St_j[St_tiny])
        # then, if the larger particle is in intermediate regime
        St_inter = ((St_i >= St_12) & (St_i < 1)) | ((St_j >= St_12) & (St_j < 1))
        St_star12 = torch.maximum(St_i[St_inter], St_j[St_inter])
        epsilon = St_i[St_inter] / St_j[St_inter]
        epsilon[epsilon > 1] = 1 / epsilon[epsilon > 1]
        self.dv_TM[St_inter] = torch.sqrt(St_star12) * torch.sqrt(
            (2 * 1.6 - (1 + epsilon) + 2 / (1 + epsilon) * (1 / (1 + 1.6) + epsilon ** 3 / (1.6 + epsilon))))
        # third, if the larger particle is very large
        St_large = (St_i >= 1) | (St_j >= 1)
        self.dv_TM[St_large] = torch.sqrt(1 / (1 + St_i[St_large]) + 1 / (1 + St_j[St_large]))

        if self.kwargs.get("W12_VD", False):
            # for debug use for now, simplify the turbulent relative velocities to two regimes
            # for both particles with St < 1, use the formula below (from Windmark+2012b)
            self.dv_TM[~St_large] = torch.sqrt(torch.maximum(St_i[~St_large], St_j[~St_large]) * 3.0)

        if self.feedback_flag:
            # reduce the turbulent relative velocities by the weighted midplane dust-to-gas density ratio
            # effectively, using alpha_D = alpha / (1 + <eps>)^K, and K defaults to 1
            # N.B., H_d are initialized to zeros, so this flag must be False in the first call
            u_gas_TM *= 1 / np.sqrt(min(self.FB_eps_cap, (1 +
                (self.sigma/self.H_d).sum().item()/self.sqrt_2_pi*self.dlog_a / self.rho_g0)**self.feedback_K))

        # move this step to below so we can repeat less
        # self.dv_TM = self.dv_TM * u_gas_TM
        # print(f"u_gas_TM={u_gas_TM:.3e}, St_12={St_12:.3e}, Re={Re:.3e}")

        # sum up all contributions to the relative velocities
        self.dv = torch.sqrt(self.dv_BM ** 2 + (self.dv_TM * u_gas_TM) ** 2)
        # if you want to Gaussian smooth the 2D dv array, use this
        # sigma = [3, 3] # [sigma_y, sigma_x]
        # self.dv = sp.ndimage.filters.gaussian_filter(self.dv, sigma, mode='constant')

    def __init_calculate_St(self):
        """ Calculate the Stokes number of particles that does not depend on dv """

        # first iteration, assume Re_d < 1 and then calculate St based on lambda_mpf
        if self.simple_St_flag:
            # only use Epstein regime
            self.St_regimes = torch.ones_like(self.St)
            self.St = self.rho_m * self.a / self.Sigma_g * torch.pi / 2
        else:
            # default: use Epstein regime + Stokes regime
            self.St_regimes[self.a < self.lambda_mpf * 9 / 4] = 1
            self.St_regimes[self.a >= self.lambda_mpf * 9 / 4] = 2
            self.St[self.St_regimes == 1] = (self.rho_m * self.a / self.Sigma_g * torch.pi / 2)[self.St_regimes == 1]
            self.St[self.St_regimes == 2] = (self.sqrt_2_pi / 9 * (self.rho_m * self.sigma_H2 * self.a ** 2)
                                             / self.cgs_mu / self.H)[self.St_regimes == 2]

        if self.full_St_flag and not self.legacy_parRe_flag:
            St_tiny = (self.St < self.St_12)
            self.dv2gas[St_tiny] = self.Re ** (1 / 4) * self.St[St_tiny]
            St_inter = (self.St >= self.St_12)
            # np.sqrt((2 * 1.6 - 1 + 2 * (1 / (1 + 1.6)))) = 1.7231456030268508
            self.dv2gas[St_inter] = torch.sqrt(self.St[St_inter]) * 1.7231456030268508

            self.Re_d = 2 * self.a * self.dv2gas / self.nu_mol
            # self.St_regimes[self.a < self.lambda_mpf * 9 / 4] = 1
            self.St_regimes[self.a >= self.lambda_mpf * 9 / 4] = 2
            self.St_regimes[(self.Re_d >= 1) & (self.Re_d < 800)] = 3
            self.St_regimes[self.Re_d >= 800] = 4

            # self.St[self.St_regimes == 1] = (self.rho_m * self.a / self.Sigma_g * np.pi / 2)[self.St_regimes == 1]
            self.St[self.St_regimes == 2] = (2 * self.rho_m * self.a ** 2 / (9 * self.nu_mol * self.rho_g0)
                                             * self.c_s / self.H)[self.St_regimes == 2]
            self.St[self.St_regimes == 3] = (2 ** 0.6 * self.rho_m * self.a ** 1.6
                                             / (9 * self.nu_mol ** 0.6 * self.rho_g0 * self.dv2gas ** 0.4)
                                             * self.c_s / self.H)[self.St_regimes == 3]
            self.St[self.St_regimes == 4] = ((6 * self.rho_m * self.a / (self.rho_g0 * self.dv2gas))
                                             * self.c_s / self.H)[self.St_regimes == 4]

        self.calculate_dv()

    def _calculate_St(self):
        """ Calculate the Stokes number of particles (and dv, dv2gas)

        Notes:
            In the method paper (LCL22), we consider dust in a global pressure maximum, where the commonly-used
        differential drift, settling, or orbital velocity does not contribute to relative dust-to-gas velocity.
        However, such a dv is required to calculate the particle Reynolds number Re_d = 2*a*dv/nu_mol. For simplicity,
        we adopted dv(a, amin) to compute Re_d. This "simplification" actually introduced extra dependence on amin
        and coupled dv and St (since Stokes number depends on dv for turbulent regimes), leading to further
        complications when feedback_flag is ON (and causing the major performance hit). In addition, dv(a, amin)
        incorporates the Brownian motions of amin and becomes huge if amin is tiny, shifting a(St=1) to a larger size.
        What is physically expected for the relative d2g velocity is macroscopic, like differential drift. Here, we
        neglect the microscopic Brownian motions and reconsider dv from the turbulent relative velocity between dust
        with size a and gas with St = 0. In this physically motivated way, we eliminate the dependence on amin and
        decouple dv and St. The resulting St is basically the same for small particles, and turbulence regimes shift
        toward larger particles. Besides, due to the steeper slope of Stokes regime, a(St=1) is a factor of a few
        smaller; St for a>10cm is also a factor of a few larger, making breakthrough via mass transfer faster and
        easier if there are enough large particles.
        """

        # after decoupling dv[0] and St, there is no need to update St unless Sigma_g changes
        if self.dyn_env_flag is True:
            self.__init_calculate_St()

        # self.calculate_dv()
        # only u_gas_TM changes with eps_tot
        if self.feedback_flag:
            u_gas_TM = self.u_gas_TM / np.sqrt(min(self.FB_eps_cap, (1 +
                  (self.sigma/self.H_d).sum().item()/self.sqrt_2_pi*self.dlog_a / self.rho_g0)**self.feedback_K))
            self.dv = torch.sqrt(self.dv_BM ** 2 + (self.dv_TM * u_gas_TM) ** 2)

    def _calculate_St_legacy(self):
        """ Calculate the Stokes number of particles (and dv)

        REFs: Section 3.5 and Eqs. 10, 14, 57 in Birnstiel+2010

        Thoughts:
            (1) This calculate_St should actually be the initialization of St, where full_St requires first
              estimating 2-regime St which gives first dv. However, after that, calculations of dv and St should be
              able to depend on previous step such that 2-regime_St is not required for the next full_St calculations.
            ===> Tests show that simplified St-calculations lead to St differences of the order of a few percent
            (2) Since calculating St and dv is expensive, we could choose to update them every a few time-step,
              since the changes are usually very small.
            ===> should consider this option in the new version
            (3) We should decouple St and dv[0], which will reduce the number of iterations on St calculations.
              To do so, we can calculate a new array of delta_v between dust and gas by using St=0,
              without going into calculate_dv that is designed for relative velocities between particles.
            ===> The dependence on dv[0] comes from calculating Re_d. Thus, the method used in this function is
              in some sense robust b/c changing amin to a smaller value (e.g., 1e-4 to 1e-5) almost does not affect
              the values of St for dust with sizes > original a_min. This is b/c St for Epstein regime and
              most of the Stokes regime (i.e., regime 1 and 2) does not depend on dv[0] but that is where dv[0]
              changes the most. For larger sizes, dv[0] merely changes since it is dominated by the larger St, which
              in turn makes St almost the same.
            ===> Another reason to decouple St and dv[0] is that, the formula for dv_BM increases drastically when
              amin becomes much smaller (dv[0] ~ 10^4 cm/s when amin ~ 1e-7 cm), which increases Re_d and makes
              a(St=1) smaller. In other words, Brownian motion relative speeds now becomes the dominant velocity in
              dv[0] up to ~10^4 cm. The physics behind seems to be off -- the need for "u" in calculating St for large
              particles in the turbulent regime (and the regime transition from Stokes to turbulent) is focusing on
              the macroscopic relative speed between gas and large particles, not some extrapolation from micro-view.
              So physically, "u" would be the relative dust-gas drift speed in disk if we consider PPDs in general.
              However, since we are looking at a special location, dv_TM(St, 0) should be a more suitable choice.
            ===> Tests show that, if we decouple St and dv[0] and use dv2gas based on calculating dv_TM(St, 0),
              dv2gas is 1 to 2 orders of magnitude smaller than dv[0] (given amin ~ 1e-4 cm). The resulting St values
              for small particles (Epstein and early Stokes regime) are the same. For larger particles, St increases
              by a factor of a few. Consequently, a(St=1) decreases by a factor of ~3. Also, dv between very large
              particles (~ 10^3 cm) drops faster (since 1/St is smaller), probably indicating easier breakthrough.
        """

        """
        # first iteration, assume Re_d < 1 and then calculate St based on lambda_mpf
        if self.simple_St_flag:
            # only use Epstein regime
            self.St_regimes = torch.ones_like(self.St)
            self.St = self.rho_m * self.a / self.Sigma_g * torch.pi / 2
        else:
            # default: use Epstein regime + Stokes regime
            self.St_regimes[self.a < self.lambda_mpf * 9 / 4] = 1
            self.St_regimes[self.a >= self.lambda_mpf * 9 / 4] = 2
            self.St[self.St_regimes == 1] = (self.rho_m * self.a / self.Sigma_g * torch.pi / 2)[self.St_regimes == 1]
            self.St[self.St_regimes == 2] = (self.sqrt_2_pi / 9 * (self.rho_m * self.sigma_H2 * self.a ** 2)
                                            / self.cgs_mu / self.H)[self.St_regimes == 2]
        """

        if self.dyn_env_flag is True:
            self.__init_calculate_St()

        if self.feedback_flag:
            u_gas_TM = self.u_gas_TM / np.sqrt(min(self.FB_eps_cap, (1 +
                  (self.sigma/self.H_d).sum().item()/self.sqrt_2_pi*self.dlog_a / self.rho_g0)**self.feedback_K))
            self.dv = torch.sqrt(self.dv_BM ** 2 + (self.dv_TM * u_gas_TM) ** 2)

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
                + "velocities.  You may consider setting Rubble.full_St_flag = False to only use Epstein regime "
                + "and Stokes regime, neglecting turbulent regime.")
        if self.Re_d.max() > 1:
            # tmp_St_regimes = torch.clone(self.St_regimes)  # not used anymore after fixing St formula typo from B11
            self.St_regimes[self.a < self.lambda_mpf * 9 / 4] = 1
            self.St_regimes[self.a >= self.lambda_mpf * 9 / 4] = 2
            self.St_regimes[(self.Re_d >= 1) & (self.Re_d < 800)] = 3
            self.St_regimes[self.Re_d >= 800] = 4

            self.St[self.St_regimes == 1] = (self.rho_m * self.a / self.Sigma_g * np.pi / 2)[self.St_regimes == 1]
            self.St[self.St_regimes == 2] = (2 * self.rho_m * self.a ** 2 / (9 * self.nu_mol * self.rho_g0)
                                             * self.c_s / self.H)[self.St_regimes == 2]
            self.St[self.St_regimes == 3] = (2 ** 0.6 * self.rho_m * self.a ** 1.6
                                             / (9 * self.nu_mol ** 0.6 * self.rho_g0 * self.dv[0] ** 0.4)
                                             * self.c_s / self.H)[self.St_regimes == 3]
            self.St[self.St_regimes == 4] = ((6 * self.rho_m * self.a / (self.rho_g0 * self.dv[0]))
                                             * self.c_s / self.H)[self.St_regimes == 4]
            self.calculate_dv()

    def __init_kernel_constants(self):
        """ Pre-calculate frequently used constants """

        self.fac1_vd = np.sqrt(54 / torch.pi) / 3
        self.fac2_vd = np.sqrt(1.5)

    def _windmark_compute_outcomes(self):
        """ Compute Windmark+2012 collision outcomes: mass-dependent thresholds, mu, eps_ac, m_er

        This method computes 2D tensors (indexed by [i,j]) that characterize the collision outcome
        for each particle pair based on the center-of-mass energy division scheme.

        REFs: Windmark, Birnstiel, Ormel & Dullemond, A&A 540, A73 (2012)
        """

        # projectile = min(m_i, m_j), target = max(m_i, m_j)
        m_p_ij = torch.minimum(self.m_i, self.m_j)
        m_t_ij = torch.maximum(self.m_i, self.m_j)

        # mass-dependent sticking and bouncing thresholds, Eq. 1-2 in W12a
        # dv_stick = (m_p / m_s)^{-5/18} [cm/s]
        # dv_bounce = (m_p / m_b)^{-5/18} [cm/s]
        self._W12_dv_stick = (m_p_ij / self.W12_m_s) ** (-5.0 / 18)
        self._W12_dv_bounce = (m_p_ij / self.W12_m_b) ** (-5.0 / 18)

        # center-of-mass velocities, Eq. 3-4 in W12a
        # v_p = dv / (1 + m_p/m_t), v_t = dv / (1 + m_t/m_p)
        v_p = self.dv / (1 + m_p_ij / m_t_ij)
        v_t = self.dv / (1 + m_t_ij / m_p_ij)

        # largest remnant fraction for each particle, Eq. 8 in W12a
        # mu(m,v) = C * m^alpha * v^beta, where v is center-of-mass velocity
        # mu >= 1 means the particle is intact; mu < 1 means it fragments
        # clamp v_p, v_t to avoid zero-velocity issues (mu -> inf when v -> 0)
        v_p_safe = torch.clamp(v_p, min=1e-30)
        v_t_safe = torch.clamp(v_t, min=1e-30)
        self._W12_mu_p = self.W12_mu_C * m_p_ij ** self.W12_mu_alpha * v_p_safe ** self.W12_mu_beta
        self._W12_mu_t = self.W12_mu_C * m_t_ij ** self.W12_mu_alpha * v_t_safe ** self.W12_mu_beta
        # clamp mu to [0, some large value] -- values >= 1 mean intact
        self._W12_mu_p = torch.clamp(self._W12_mu_p, min=0)
        self._W12_mu_t = torch.clamp(self._W12_mu_t, min=0)

        # fragmentation threshold velocity for the projectile mass, Eq. 7 in W12a
        # v_{1.0} = (m_p / m_{1.0})^{-gamma}
        v10_p = (m_p_ij / self.W12_m_10) ** (-self.W12_gamma)

        # mass transfer accretion efficiency, Eq. 13 in W12a
        # eps_ac = a + b * (v_{1.0,beitz} / v_{1.0}) * dv
        self._W12_eps_ac = (self.W12_eps_ac_a
                            + self.W12_eps_ac_b * (self.W12_v10_beitz / v10_p) * self.dv)
        self._W12_eps_ac = torch.clamp(self._W12_eps_ac, min=self.W12_eps_ac_min, max=self.W12_eps_ac_max)

        # erosion mass ratio, Eq. 15 in W12a (calibrated by Teiser & Wurm 2009)
        # m_er / m_p = a * (m_p / m_0)^k * dv + b
        self._W12_ero_ratio = (self.W12_ero_a * (m_p_ij / self.W12_m_0) ** self.W12_ero_k * self.dv
                               + self.W12_ero_b)
        self._W12_ero_ratio = torch.clamp(self._W12_ero_ratio, min=0)

    def _windmark_fragmentation_coeff(self):
        """ Compute gF/lF using Windmark+2012 velocity+mass dependent collision outcomes

        Uses mu_p, mu_t (largest remnant fractions), eps_ac (mass transfer efficiency),
        and m_er (erosion) computed by _windmark_compute_outcomes() to build the 3D
        fragmentation gain (gF) and loss (lF) coefficient tensors.

        Fragment distribution follows n(m)dm propto m^{-kappa} dm with kappa = 9/8.
        Each fragmenting particle produces a largest remnant of mass mu*m and a power-law
        tail containing (1-mu)*m of mass.

        REFs: Windmark, Birnstiel, Ormel & Dullemond, A&A 540, A73 (2012), Sect. 3.4-3.5
        """

        self.gF.fill_(0)
        self.lF.fill_(1)

        N = self.num_grid + 2

        # ----- base fragment distribution (velocity-independent, cached) -----
        # frag_Nk[i, k]: how unit mass of particle i is distributed to bin k when fully fragmented
        # using W12 power-law index kappa = 9/8 (different from the default xi = 1.83)
        if not hasattr(self, '_W12_frag_Nk'):
            C_norm = torch.zeros_like(self.a)
            tmp_Nk = torch.tril(self.m_j ** (-self.W12_kappa + 1), -1)
            C_norm[1:] = self.m[1:] / torch.sum(tmp_Nk * self.m_j, 1)[1:]
            self._W12_frag_Nk = tmp_Nk * C_norm[:, None]  # frag_Nk[i, k] for k < i
        frag_Nk = self._W12_frag_Nk

        # ----- 2D collision outcome tensors -----
        mu_p = self._W12_mu_p  # [N, N]
        mu_t = self._W12_mu_t  # [N, N]
        eps_ac = self._W12_eps_ac  # [N, N]
        ero_ratio = self._W12_ero_ratio  # [N, N]

        # map mu_p/mu_t (defined for projectile=min, target=max) back to particle indices i and j
        # _W12_mu_p is always the mu of the smaller particle (projectile)
        # _W12_mu_t is always the mu of the larger particle (target)
        # when m_i >= m_j (j is projectile): mu_of_i = mu_t, mu_of_j = mu_p
        # when m_i <  m_j (i is projectile): mu_of_i = mu_p, mu_of_j = mu_t
        i_ge_j = self.m_i >= self.m_j  # [N, N]
        mu_i = torch.where(i_ge_j, mu_t, mu_p)  # mu for particle i
        mu_j = torch.where(i_ge_j, mu_p, mu_t)  # mu for particle j

        # collision type masks [N, N]
        proj_frags = mu_p < 1  # projectile fragments
        targ_frags = mu_t < 1  # target fragments
        is_frag = proj_frags & targ_frags  # destructive: both fragment
        is_MT = proj_frags & (~targ_frags)  # mass transfer: only projectile fragments

        mesh3D_i, mesh3D_j, mesh3D_k = torch.meshgrid(self.idx_grid, self.idx_grid, self.idx_grid, indexing='ij')

        # ===== Case 1: Destructive fragmentation (both particles fragment) =====
        # each particle produces: largest remnant (mu*m) + power-law tail ((1-mu)*m)
        is_frag_3d = is_frag[mesh3D_i, mesh3D_j]
        mu_i_3d = mu_i[mesh3D_i, mesh3D_j]
        mu_j_3d = mu_j[mesh3D_i, mesh3D_j]

        # -- power-law from particle i: bins k < i --
        idx_i_pw = is_frag_3d & (mesh3D_k < mesh3D_i)
        self.gF[idx_i_pw] = (frag_Nk[mesh3D_i[idx_i_pw], mesh3D_k[idx_i_pw]]
                              * (1 - mu_i_3d[idx_i_pw]))
        # -- power-law from particle j: bins k < j --
        idx_j_pw = is_frag_3d & (mesh3D_k < mesh3D_j)
        self.gF[idx_j_pw] += (frag_Nk[mesh3D_j[idx_j_pw], mesh3D_k[idx_j_pw]]
                               * (1 - mu_j_3d[idx_j_pw]))

        # -- remnant placement --
        m_rem_i = mu_i * self.m[:, None]  # [N, N]
        m_rem_j = mu_j * self.m[None, :]  # [N, N]
        k_rem_i = torch.searchsorted(self.m, m_rem_i.reshape(-1)).reshape(N, N).clamp(0, N - 1)
        k_rem_j = torch.searchsorted(self.m, m_rem_j.reshape(-1)).reshape(N, N).clamp(0, N - 1)

        frag_ij = is_frag.nonzero(as_tuple=False)
        if frag_ij.shape[0] > 0:
            fi, fj = frag_ij[:, 0], frag_ij[:, 1]
            ki = k_rem_i[fi, fj]
            self.gF[fi, fj, ki] += mu_i[fi, fj] * self.m[fi] / self.m[ki]
            kj = k_rem_j[fi, fj]
            self.gF[fi, fj, kj] += mu_j[fi, fj] * self.m[fj] / self.m[kj]

        # ===== Case 2: Mass transfer (only projectile fragments, target intact) =====
        # projectile fragments: (1 - eps_ac) goes to fragments (remnant + power-law), eps_ac goes to target
        # target may also be eroded: m_er is removed from target and distributed as fragments
        is_MT_3d = is_MT[mesh3D_i, mesh3D_j]
        eps_ac_3d = eps_ac[mesh3D_i, mesh3D_j]
        mu_p_3d = mu_p[mesh3D_i, mesh3D_j]
        i_ge_j_3d = i_ge_j[mesh3D_i, mesh3D_j]

        # projectile index for each (i,j,k)
        proj_idx = torch.where(i_ge_j_3d, mesh3D_j, mesh3D_i)

        # -- projectile power-law fragments: (1-eps_ac)*(1-mu_p) fraction, bins k < proj_idx --
        idx_MT_pw = is_MT_3d & (mesh3D_k < proj_idx)
        if idx_MT_pw.any():
            pk = proj_idx[idx_MT_pw]
            self.gF[idx_MT_pw] += (frag_Nk[pk, mesh3D_k[idx_MT_pw]]
                                   * (1 - eps_ac_3d[idx_MT_pw])
                                   * (1 - mu_p_3d[idx_MT_pw]))

        # -- projectile remnant and eroded mass from target --
        MT_ij = is_MT.nonzero(as_tuple=False)
        if MT_ij.shape[0] > 0:
            mi_mt, mj_mt = MT_ij[:, 0], MT_ij[:, 1]
            proj_is_j = i_ge_j[mi_mt, mj_mt]
            m_proj = torch.where(proj_is_j, self.m[mj_mt], self.m[mi_mt])
            m_targ = torch.where(proj_is_j, self.m[mi_mt], self.m[mj_mt])

            mu_p_mt = mu_p[mi_mt, mj_mt]
            eps_ac_mt = eps_ac[mi_mt, mj_mt]
            ero_r_mt = ero_ratio[mi_mt, mj_mt]

            # projectile remnant: mu_p * (1 - eps_ac) * m_proj
            m_rem_proj = mu_p_mt * (1 - eps_ac_mt) * m_proj
            k_rem_proj = torch.searchsorted(self.m, m_rem_proj).clamp(0, N - 1)
            self.gF[mi_mt, mj_mt, k_rem_proj] += (mu_p_mt * (1 - eps_ac_mt)
                                                    * m_proj / self.m[k_rem_proj])

            # accreted mass onto target: eps_ac * m_proj added to target bin
            targ_idx_mt = torch.where(proj_is_j, mi_mt, mj_mt)
            self.gF[mi_mt, mj_mt, targ_idx_mt] += eps_ac_mt * m_proj / self.m[targ_idx_mt]

            # eroded mass from target: distributed as power-law using frag_Nk[k_er_max]
            # vectorized: for each MT pair, the erosion fragments follow frag_Nk[k_er_max, :]
            # but we need to normalize so total eroded mass = m_er for each pair
            m_er = ero_r_mt * m_proj  # [P]
            k_er_max = torch.searchsorted(self.m, m_er).clamp(0, N - 1)  # [P]

            # gather the fragment distribution for each pair's k_er_max
            er_dist_all = frag_Nk[k_er_max]  # [P, N]
            # zero out bins k >= k_er_max for each pair
            k_range = torch.arange(N, device=self.device)  # [N]
            er_dist_all = er_dist_all * (k_range[None, :] < k_er_max[:, None])  # [P, N]
            # normalize: total mass in distribution = sum(er_dist * m)
            er_total = (er_dist_all * self.m[None, :]).sum(dim=1)  # [P]
            # scale factor: m_er / er_total, avoid div by zero
            valid = (er_total > 0) & (m_er > 0) & (k_er_max > 0)
            scale = torch.zeros_like(m_er)
            scale[valid] = m_er[valid] / er_total[valid]
            # add to gF: gF[mi_mt[p], mj_mt[p], k] += scaled_er[p, k]
            scaled_er = er_dist_all * scale[:, None]  # [P, N]
            # expand indices to [P, N] for scatter-add into 3D gF
            P = MT_ij.shape[0]
            ii_exp = mi_mt[:, None].expand(P, N)  # [P, N]
            jj_exp = mj_mt[:, None].expand(P, N)  # [P, N]
            kk_exp = k_range[None, :].expand(P, N)  # [P, N]
            nonzero = scaled_er != 0
            if nonzero.any():
                self.gF.index_put_((ii_exp[nonzero], jj_exp[nonzero], kk_exp[nonzero]),
                                   scaled_er[nonzero], accumulate=True)

        # ===== loss coefficient lF =====
        # destructive frag: both particles lose all mass -> lF = 1 (default on j==k diagonal)
        # mass transfer: projectile loses all mass (lF=1), target loses m_er
        if MT_ij.shape[0] > 0:
            targ_idx = torch.where(proj_is_j, mi_mt, mj_mt)
            frac_loss = ero_r_mt * m_proj / m_targ
            self.lF[mi_mt, mj_mt, targ_idx] = frac_loss

        # standard cleanup
        self.lF[self.idx_jk_diff] = 0
        self.lF[self.idx_ij_same] *= 0.5

        # halve gF for i==j collisions
        self.gF[self.idx_ij_same][self.gF[self.idx_ij_same] == 0.0] = 1.0
        self.gF[self.idx_ij_same] *= 0.5

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def calculate_kernels(self):
        """ Calculate the kernel for coagulation and fragmentation (ghost zones only takes mass)

        REFs: Brauer+2008, Birnstiel+2010, 2011, Windmark+2012
        """

        # localize variables for simplicity and improve readability
        if self.windmark_flag:
            # Windmark+2012: mass-dependent thresholds (2D tensors)
            # In W12, sticking is below dv_stick, bouncing between dv_stick and dv_bounce,
            # and fragmentation/mass-transfer is determined by mu above dv_bounce.
            # For the probability computation, we map:
            #   u_b -> dv_stick (below which 100% sticking)
            #   u_f -> dv_bounce (above which fragmentation/MT occurs)
            u_b = self._W12_dv_stick
            u_f = self._W12_dv_bounce
        else:
            u_b, u_f = self.u_b, self.u_f

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

            p_f = (1 + self.fac1_vd * torch.exp(-3/2 * (u_f/self.dv)**2) * (u_f/self.dv)
                   - torch.special.erf(self.fac2_vd * u_f/self.dv))

            # we may try FURTHER and put a bouncing barrier in
            if self.bouncing_flag:
                p_c = (-self.fac1_vd * torch.exp(-3/2 * (u_b/self.dv)**2) * (u_b/self.dv)
                       + torch.special.erf(self.fac2_vd * u_b/self.dv))
            else:
                p_c = 1 - p_f
        else:
            delta_u = 0.2 * u_f                                        # transition width, ref for 0.2: Birnstiel+2011
            soften_u_f = u_f - delta_u                                 # p_f = 0 when du_ij < soften_u_f
            p_f = torch.zeros_like(self.m_j)                           # set all to zero
            p_f[self.dv > u_f] = 1.0                                   # set where du_ij > u_f to 1
            p_f_mask = (self.dv > soften_u_f) & (self.dv < u_f)
            if self.windmark_flag:
                # u_f and delta_u are 2D tensors, need to index them by the same mask
                p_f[p_f_mask] = 1 - (u_f[p_f_mask] - self.dv[p_f_mask]) / delta_u[p_f_mask]
            else:
                p_f[p_f_mask] = 1 - (u_f - self.dv[p_f_mask]) / delta_u  # set else values

            if self.bouncing_flag:
                p_c = torch.zeros_like(self.m_j)
                p_c[self.dv < u_b] = 1.0
                if self.windmark_flag:
                    # Windmark+2012 Eq. 22-23: logarithmic sticking-to-bouncing transition
                    # p_c = 1 - k1 * log10(dv) - k2, for dv_stick < dv < dv_bounce
                    # k2 = log10(m_p / m_s) / log10(m_b / m_s), depends on projectile mass
                    m_p_ij = torch.minimum(self.m_i, self.m_j)
                    k2 = (torch.log10(m_p_ij) - self.W12_log_ms) / self.W12_log_mb_over_ms
                    trans_mask = (self.dv >= u_b) & (self.dv < u_f)
                    dv_safe = torch.clamp(self.dv, min=1e-30)
                    p_c_trans = 1 - self.W12_k1 * torch.log10(dv_safe) - k2
                    p_c[trans_mask] = p_c_trans[trans_mask].clamp(0, 1)
            else:
                p_c = 1 - p_f

        self.p_f.fill_(0)
        self.p_c.fill_(0)
        self.p_b.fill_(0)
        if self.uni_gz_flag == True:
            # unidirectional ghost zones, where the left one also coagulate and the right one also fragment
            self.p_f[1:, 1:] = p_f[1:, 1:]
            self.p_c[:-1, :-1] = p_c[:-1, :-1]
            self.p_b = (1 - self.p_f - self.p_c)
        elif self.uni_gz_flag == False:
            # set the probabilities to zero for any p_{ij} that involves m_0 and m_last (i.e., inactive ghost zones)
            self.p_f[1:-1, 1:-1] = p_f[1:-1, 1:-1]
            self.p_c[1:-1, 1:-1] = p_c[1:-1, 1:-1]
            self.p_b[1:-1, 1:-1] = (1 - self.p_f - self.p_c)[1:-1, 1:-1]
        elif self.uni_gz_flag == 2:
            # set the probabilities to zero for any p_{ij} that involves m_last (i.e., active left + inactive right)
            self.p_f[1:-1, 1:-1] = p_f[1:-1, 1:-1]
            self.p_c[:-1, :-1] = p_c[:-1, :-1]
            self.p_b[:-1, :-1] = (1 - self.p_f - self.p_c)[:-1, :-1]
        elif self.uni_gz_flag == 3:
            # set the probabilities to zero for any p_{ij} that involves m_0 (i.e., inactive left + active right)
            self.p_f[1:, 1:] = p_f[1:, 1:]
            self.p_c[1:-1, 1:-1] = p_c[1:-1, 1:-1]
            self.p_b[1:, 1:] = (1 - self.p_f - self.p_c)[1:, 1:]
        elif self.uni_gz_flag == 4:
            # fully active ghost zones
            self.p_f = p_f
            self.p_f[0, 0] = 0
            self.p_c = p_c
            self.p_c[-1, -1] = 0
            self.p_b = 1 - self.p_f - self.p_c
        else:
            raise ValueError(f"uni_gz_flag must be one of: True (both gz active), False (both gz inactive), "
                             f"2 (active left gz + inactive right gz), or 3 (inactive left gz + active right gz).")

        # Windmark+2012: when both particles survive (mu_p >= 1 and mu_t >= 1),
        # the collision is bouncing, not fragmentation — move p_f to p_b for those pairs
        if self.windmark_flag:
            both_intact = (self._W12_mu_p >= 1) & (self._W12_mu_t >= 1)
            self.p_b[both_intact] += self.p_f[both_intact]
            self.p_f[both_intact] = 0

        # clamp probabilities to avoid tiny negatives from numerical noise (e.g., in erf computation)
        self.p_c.clamp_(min=0)
        self.p_f.clamp_(min=0)
        self.p_b.clamp_(min=0)

        # Note: since K is symmetric, K.T = K, and dot(K_{ik}, n_i) = dot(K_{ki}, n_i) = dot(n_i, K_{ik})
        # kernel = self.dv * self.geo_cs                               # general kernel = du_ij * geo_cs
        self.K = self.dv * self.geo_cs * self.p_c                      # coag kernel, K_ij
        self.L = self.dv * self.geo_cs * self.p_f                      # frag kernel, L_ij

        if self.f_mod_flag is True:
            # use the modulation function to limit the interactions between mass bins that have low particle numbers
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', '', RuntimeWarning)
                tmp_Nk = self.Nk * self.S_annulus
                self.f_mod = torch.exp(-1 / tmp_Nk[:, None] - 1 / tmp_Nk)
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

        self.M = 0.5 * self.C * self.K[:, :, None]                     # multiply C and K with matching i and j
        self.M[self.idx_ij_same] *= 0.5                                # remove collision events counted twice

        # self.Nk.dot(self.Nk.dot(self.M2)) is equiv to self.Nk * self.K.dot(self.Nk), the original explicit way
        tmp_M = self.K[:, :, None] * torch.ones_like(self.C)
        tmp_M[self.idx_jk_diff] = 0.0                                  # b/c M2 has a factor of delta(j-k)
        tmp_M[self.idx_ij_same] *= 0.5                                 # remove collision events counted twice
        self.M -= tmp_M

        # the 3rd/4th term is from fragmentation
        if self.frag_flag:
            self.M += 0.5 * self.L[:, :, None] * self.gF               # multiply gF and L with matching i and j
            # self.M3[self.mesh3D_i == self.mesh3D_j] *= 0.5           # self.gF already considered this 0.5 factor

            self.M -= self.L[:, :, None] * self.lF                     # multiply lF and L with matching i and j
            # self.M4[self.idx_jk_diff] = 0.0                          # self.lF alrady considered this
            # self.M4[self.idx_ij_same] *= 0.5                         # self.lF alrady considered this

        # sum up all the parts
        # self.M = self.M1 - self.M2 + self.M3 - self.M4

        # now convert to vertically integrated kernel
        self.tM = self.M / self.vi_fac[:, :, None]

    def _update_kernels(self, update_coeff=False):

        if self.f_mod_flag is True and self.feedback_flag is False and self.dyn_env_flag is False \
                and self.windmark_flag is False and self.cycle_count > 0:
            # use the modulation function to limit the interactions between mass bins that have low particle numbers
            # if w/o feedback effects, we don't need to go through the entire update_kernels procedure, only new f_mod
            # and new tM are needed. Thus, we squeeze them here
            if self.cycle_count == 1:
                # Previously, self.K/L has been over-written with *= self.f_mod. We need to re-generate them when going
                # into this branch for the first time
                # kernel = self.dv * self.geo_cs
                self.L = self.dv * self.geo_cs * self.p_f
                self.K = self.dv * self.geo_cs * self.p_c
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', '', RuntimeWarning)
                tmp_Nk = self.Nk * self.S_annulus
                self.f_mod = torch.exp(-1 / tmp_Nk[:, None] - 1 / tmp_Nk)
            # for f_mod only, we cannot just modify K/L since they won't be re-generated
            tmp_K = self.K * self.f_mod
            tmp_L = self.L * self.f_mod

            self.M = 0.5 * self.C * tmp_K[:, :, None]  # multiply C and K with matching i and j
            self.M[self.idx_ij_same] *= 0.5  # less collision events in single particle species
            tmp_M = tmp_K[:, :, None] * torch.ones_like(self.C)
            tmp_M[self.idx_jk_diff] = 0.0  # b/c M2 has a factor of delta(j-k)
            tmp_M[self.idx_ij_same] *= 0.5  # less collision events in single particle species
            self.M -= tmp_M

            if self.frag_flag:
                self.M += 0.5 * tmp_L[:, :, None] * self.gF  # multiply gF and L with matching i and j
                self.M -= tmp_L[:, :, None] * self.lF  # multiply lF and L with matching i and j

            # now convert to vertically integrated kernel
            self.tM = self.M / self.vi_fac[:, :, None]
            return None

        # first, update disk parameters if needed in the future
        if self.dyn_env_flag:
            if self.t >= self.dyn_env_burn_in:
                self.update_disk_parameters()

        # then, calculate solid properties
        self.calculate_St()
        if self.feedback_flag:
            # make a closer guess on the total midplane dust-to-gas density ratio
            self.eps = self.sigma * self.dlog_a / self.sqrt_2_pi / self.H_d / self.rho_g0
            self.eps_tot = self.eps.sum().item()

            # use root finding to self-consistently calculate the weighted midplane dust-to-gas ratio
            self._root_finding_tmp = (self.sigma * self.dlog_a / self.sqrt_2_pi / self.rho_g0 / self.H).cpu().numpy()
            _St = self.St.cpu().numpy()
            """
            N.B.: in fact St also depends on eps through dv, but we assume St won't change too much (especially for
            particles with small St) and solve for eps that makes H_d and eps self-consistent.

            One caveat when eps_tot>>1 though: 
            each time update_kernels() is called, St in the high mass tail varies,
            leading to *slightly* different St, dv, and thus *slightly* different kernels (K varies more, L less)

            spopt.root_scalar only utilize CPU, if torch.sum uses GPU, the communications between CPU&GPU is
            very expensive, especially when root_scalar requires multiple (>10) function calls.
            Benchmarks show that tmp_sln on CPU solely cost 80e-6s, tmp_sln on CPU+GPU cost 1.84ms. An alternative
            coding choice is to use et/x0/x1 as scalar_tensor and eliminate the call item(), which however cost
            2.14ms, somewhat more.
            """
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    tmp_sln = spopt.root_scalar(lambda et: et - np.sum(self._root_finding_tmp * (
                            1 + _St / self.alpha * min(self.FB_eps_cap,
                                                       (1 + et) ** self.feedback_K)) ** 0.5).item(),
                                                x0=self.eps_tot, x1=self.eps_tot * 5, method='brentq',
                                                bracket=[self.eps_tot * 0.2, self.eps_tot * 100])
                    if tmp_sln.converged and np.isfinite(tmp_sln.root):
                        _root_finding_succeed = True
                        self.H_d = self.H / torch.sqrt(1 + self.St / self.alpha
                                                       * min(self.FB_eps_cap, (1 + tmp_sln.root)**self.feedback_K))
                        self.eps = self.sigma * self.dlog_a / self.sqrt_2_pi / self.H_d / self.rho_g0
                        self.eps_tot = self.eps.sum().item()
                    else:
                        raise RuntimeError("solution not converged")
            except Exception as e:
                self.warn("Root finding for the total midplane dust-to-gas density ratio failed. "
                          + "\nError message: " + e.__str__() + "\nFall back to use five iterations.")
                for idx_fb in range(4):
                    # if eps.sum() is already larger than the desired capped value, skip
                    # if (1 + self.eps.sum())**self.feedback_K > self.FB_eps_cap:
                    #     break
                    # on a second thought, even if eps.sum() > cap, one more loop is needed to make H_d consistent
                    # manually finding closer solution
                    self.H_d = self.H / torch.sqrt(1 + self.St / self.alpha
                                                   * min(self.FB_eps_cap, (1 + self.eps.sum().item())**self.feedback_K))
                    self.eps = self.sigma * self.dlog_a / self.sqrt_2_pi / self.H_d / self.rho_g0
                    self.eps_tot = self.eps.sum().item()
        else:
            # using Eq. 28 in Youdin & Lithwick 2007, ignore the factor: np.sqrt((1 + self.St) / (1 + 2*self.St))
            self.H_d = self.H / torch.sqrt(1 + self.St / self.alpha)
            # the ignored factor may further reduce H_d and lead to a larger solution to midplane eps_tot
            self.eps = self.sigma * self.dlog_a / self.sqrt_2_pi / self.H_d / self.rho_g0
            self.eps_tot = self.eps.sum().item()

        if self.kwargs.get("B10_Hd", False):
            # for debug use only, calculate the solid scale height based on Eq. 51 in Birnstiel+2010
            # this formula mainly focuses on smaller particles and results in super small H_d for St >> 1
            # which may be improved by adding limits from the consideration of KH effects
            self.H_d = self.H * torch.minimum(
                torch.scalar_tensor(1.0),
                torch.sqrt(self.alpha / (torch.minimum(self.St, torch.scalar_tensor(0.5)) * (1 + self.St ** 2))))
            self.eps = self.sigma * self.dlog_a / self.sqrt_2_pi / self.H_d / self.rho_g0

        self.h_ss_ij = self.H_d ** 2 + self.H_d[:, None] ** 2
        self.vi_fac = torch.sqrt(2 * torch.pi * self.h_ss_ij)

        if self.windmark_flag:
            # Windmark+2012: recompute velocity-dependent outcomes and gF/lF every kernel update
            self._windmark_compute_outcomes()
            self._windmark_fragmentation_coeff()
        elif update_coeff:
            # currently, no flag requires updateing coagulation coeff
            # self.piecewise_coagulation_coeff()
            self.gF.fill_(0)  # reset the gain coeff
            self.lF.fill_(1)  # reset the loss coeff
            self.__init_powerlaw_fragmentation_coeff()
            self.powerlaw_fragmentation_coeff()

        # finally, re-evaluate kernels
        self.calculate_kernels()

    def _update_kernels_legacy(self, update_coeff=False):
        """ Update collisional kernels """

        if self.f_mod_flag is True and self.feedback_flag is False and self.dyn_env_flag is False \
                and self.windmark_flag is False and self.cycle_count > 0:
            # use the modulation function to limit the interactions between mass bins that have low particle numbers
            # if w/o feedback effects, we don't need to go through the entire update_kernels procedure, only new f_mod
            # and new tM are needed. Thus, we squeeze them here
            if self.cycle_count == 1:
                # Previously, self.K/L has been over-written with *= self.f_mod. We need to re-generate them when going
                # into this branch for the first time
                # kernel = self.dv * self.geo_cs
                self.L = self.dv * self.geo_cs * self.p_f
                self.K = self.dv * self.geo_cs * self.p_c
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', '', RuntimeWarning)
                tmp_Nk = self.Nk * self.S_annulus
                self.f_mod = torch.exp(-1 / tmp_Nk[:, None] - 1 / tmp_Nk)
            # for f_mod only, we cannot just modify K/L since they won't be re-generated
            tmp_K = self.K * self.f_mod
            tmp_L = self.L * self.f_mod

            self.M = 0.5 * self.C * tmp_K[:, :, None]  # multiply C and K with matching i and j
            self.M[self.idx_ij_same] *= 0.5  # less collision events in single particle species
            tmp_M = tmp_K[:, :, None] * torch.ones_like(self.C)
            tmp_M[self.idx_jk_diff] = 0.0  # b/c M2 has a factor of delta(j-k)
            tmp_M[self.idx_ij_same] *= 0.5  # less collision events in single particle species
            self.M -= tmp_M

            if self.frag_flag:
                self.M += 0.5 * tmp_L[:, :, None] * self.gF  # multiply gF and L with matching i and j
                self.M -= tmp_L[:, :, None] * self.lF  # multiply lF and L with matching i and j

            # now convert to vertically integrated kernel
            self.tM = self.M / self.vi_fac[:, :, None]
            return None

        # first, update disk parameters if needed in the future
        if self.dyn_env_flag:
            if self.t >= self.dyn_env_burn_in:
                self.update_disk_parameters()

        # then, calculate solid properties
        if self.feedback_flag:
            # make a closer guess on the total midplane dust-to-gas density ratio
            self.H_d = self.H / torch.sqrt(1 + self.St / self.alpha
                                           * min(self.FB_eps_cap, (1 + self.eps.sum().item())**self.feedback_K))
            self.eps = self.sigma * self.dlog_a / self.sqrt_2_pi / self.H_d / self.rho_g0
            self.eps_tot = self.eps.sum().item()
            self.calculate_St()

            # use root finding to self-consistently calculate the weighted midplane dust-to-gas ratio
            self._root_finding_tmp = (self.sigma * self.dlog_a / self.sqrt_2_pi / self.rho_g0 / self.H).cpu().numpy()
            _St = self.St.cpu().numpy()
            """
            N.B.: in fact St also depends on eps through dv, but we assume St won't change too much (especially for
            particles with small St) and solve for eps that makes H_d and eps self-consistent.

            One caveat when eps_tot>>1 though: 
            each time update_kernels() is called, St in the high mass tail varies,
            leading to *slightly* different St, dv, and thus *slightly* different kernels (K varies more, L less)

            spopt.root_scalar only utilize CPU, if torch.sum uses GPU, the communications between CPU&GPU is
            very expensive, especially when root_scalar requires multiple (>10) function calls.
            Benchmarks show that tmp_sln on CPU solely cost 80e-6s, tmp_sln on CPU+GPU cost 1.84ms. An alternative
            coding choice is to use et/x0/x1 as scalar_tensor and eliminate the call item(), which however cost
            2.14ms, somewhat more.
            """
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    tmp_sln = spopt.root_scalar(lambda et: et - np.sum(self._root_finding_tmp * (
                            1 + _St / self.alpha * min(self.FB_eps_cap,
                                                       (1 + et) ** self.feedback_K)) ** 0.5).item(),
                                                x0=self.eps_tot, x1=self.eps_tot * 5, method='brentq',
                                                bracket=[self.eps_tot * 0.2, self.eps_tot * 100])
                    if tmp_sln.converged and np.isfinite(tmp_sln.root):
                        _root_finding_succeed = True
                        self.H_d = self.H / torch.sqrt(1 + self.St / self.alpha
                                                       * min(self.FB_eps_cap, (1 + tmp_sln.root)**self.feedback_K))
                        self.eps = self.sigma * self.dlog_a / self.sqrt_2_pi / self.H_d / self.rho_g0
                        self.eps_tot = self.eps.sum().item()
                        self.calculate_St()
                    else:
                        raise RuntimeError("solution not converged")
            except Exception as e:
                self.warn("Root finding for the total midplane dust-to-gas density ratio failed. "
                          + "\nError message: " + e.__str__() + "\nFall back to use five iterations.")
                for idx_fb in range(4):
                    # if eps.sum() is already larger than the desired capped value, skip
                    # if (1 + self.eps.sum())**self.feedback_K > self.FB_eps_cap:
                    #     break
                    # on a second thought, even if eps.sum() > cap, one more loop is needed to make H_d consistent
                    # manually finding closer solution
                    self.H_d = self.H / torch.sqrt(1 + self.St / self.alpha
                                                   * min(self.FB_eps_cap, (1 + self.eps.sum().item())**self.feedback_K))
                    self.eps = self.sigma * self.dlog_a / self.sqrt_2_pi / self.H_d / self.rho_g0
                    self.eps_tot = self.eps.sum().item()
                    self.calculate_St()
        else:
            self.calculate_St()
            # using Eq. 28 in Youdin & Lithwick 2007, ignore the factor: np.sqrt((1 + self.St) / (1 + 2*self.St))
            self.H_d = self.H / torch.sqrt(1 + self.St / self.alpha)
            # the ignored factor may further reduce H_d and lead to a larger solution to midplane eps_tot
            self.eps = self.sigma * self.dlog_a / self.sqrt_2_pi / self.H_d / self.rho_g0
            self.eps_tot = self.eps.sum().item()

        if self.kwargs.get("B10_Hd", False):
            # for debug use only, calculate the solid scale height based on Eq. 51 in Birnstiel+2010
            # this formula mainly focuses on smaller particles and results in super small H_d for St >> 1
            # which may be improved by adding limits from the consideration of KH effects
            self.H_d = self.H * torch.minimum(
                torch.scalar_tensor(1.0),
                torch.sqrt(self.alpha / (torch.minimum(self.St, torch.scalar_tensor(0.5)) * (1 + self.St ** 2))))
            self.eps = self.sigma * self.dlog_a / self.sqrt_2_pi / self.H_d / self.rho_g0

        self.h_ss_ij = self.H_d ** 2 + self.H_d[:, None] ** 2
        self.vi_fac = torch.sqrt(2 * torch.pi * self.h_ss_ij)

        if self.windmark_flag:
            self._windmark_compute_outcomes()
            self._windmark_fragmentation_coeff()
        elif update_coeff:
            # currently, no flag requires updateing coagulation coeff
            # self.piecewise_coagulation_coeff()
            self.gF.fill_(0)  # reset the gain coeff
            self.lF.fill_(1)  # reset the loss coeff
            self.__init_powerlaw_fragmentation_coeff()
            self.powerlaw_fragmentation_coeff()

        # finally, re-evaluate kernels
        self.calculate_kernels()

    def flag_updated(self, flag_name):
        """ Update flag-dependent kernels whenever a flag changes """

        if self.flag_activated:
            if flag_name in ["f_mod_flag", "feedback_flag", "dyn_env_flag", "windmark_flag"]:
                if (self.f_mod_flag is True or self.feedback_flag is True
                        or self.dyn_env_flag is True or self.windmark_flag is True):
                    self.static_kernel_flag = False
                else:
                    self.static_kernel_flag = True
            if flag_name in ["closed_box_flag", "dyn_env_flag"]:
                if self.closed_box_flag is False or self.dyn_env_flag is True:
                    self.__init_update_solids()
            if flag_name in ["simple_St_flag", "full_St_flag"]:
                self.__init_calculate_St()
            if flag_name in ["legacy_parRe_flag"]:
                if self.legacy_parRe_flag:
                    self.__init_calculate_St()
                    self.calculate_St = self._calculate_St_legacy
                    self.update_kernels = self._update_kernels_legacy
                else:
                    self.__init_calculate_St()
                    self.calculate_St = self._calculate_St
                    self.update_kernels = self._update_kernels

            # N.B., for legacy_parRe_flag = True, St/dv[0]/H_d/eps change modestly every time update_kernels is called,
            # due to the fact that St[St_regimes == 3|4] and dv[0] are mutually dependent to each other.
            # However, flag_updated calls update_kernels a lot after flag_activated is switched to True.
            # To produce consistent results that do not depend on how (i.e., in which order) flags are set,
            # it is important to reset St calculations to the initial state. The price is slightly more discrepancy
            # w.r.t. legacy results in the method paper (on the order of 1e-6).
            if self.legacy_parRe_flag is True:
                self.__init_calculate_St()
            # finally, update kernels
            if flag_name in ["mass_transfer_flag", "windmark_flag"]:
                self.update_kernels(update_coeff=True)
            else:
                self.update_kernels()
        else:
            # do not update kernel during the initial setup
            pass

    def show_flags(self):
        """ print all the flags to show status """

        print(f"{'debug_flag:':32}", self.debug_flag)
        print(f"{'frag_flag:':32}", self.frag_flag)
        print(f"{'bouncing_flag:':32}", self.bouncing_flag)
        print(f"{'vel_dist_flag:':32}", self.vel_dist_flag)
        print(f"{'mass_transfer_flag:':32}", self.mass_transfer_flag)
        print(f"{'simple_St_flag:':32}", self.simple_St_flag)
        print(f"{'full_St_flag:':32}", self.full_St_flag)
        print(f"{'uni_gz_flag:':32}", self.uni_gz_flag)
        print(f"{'f_mod_flag:':32}", self.f_mod_flag)
        print(f"{'feedback_flag:':32}", self.feedback_flag)
        print(f"{'closed_box_flag:':32}", self.closed_box_flag)
        print(f"{'dyn_env_flag:':32}", self.dyn_env_flag)
        print(f"{'windmark_flag:':32}", self.windmark_flag)
        print(f"")
        print(f"{'static_kernel_flag:':32}", self.static_kernel_flag)
        print(f"{'flag_activated:':32}", self.flag_activated)

    def _user_setup(self, test='BMonly'):
        """ Customized setup for testing purposes (mainly for debugging) """

        # generate tmp mesh to save memory usage
        mesh2D_i, mesh2D_j = torch.meshgrid(self.idx_grid, self.idx_grid, indexing='ij')

        if test == 'BMonly':
            # for Brownian motion only (and only same-sized solids collide)
            self.frag_flag = False
            self.bouncing_flag = False
            self.vel_dist_flag = False
            self.mass_transfer_flag = False
            self.simple_St_flag = True
            self.dv = self.dv_BM
            self.dv[mesh2D_i != mesh2D_j] = 0
            self.calculate_kernels()
        elif test == 'BM+turb':
            # same-sized BM plus the relative turbulence velocities between same-sized solids
            self.frag_flag = False
            self.bouncing_flag = False
            self.vel_dist_flag = False
            self.mass_transfer_flag = False
            self.simple_St_flag = True
            self.dv = self.dv_BM
            self.dv[mesh2D_i != mesh2D_j] = 0
            dv_TM = self.c_s * torch.sqrt(2 * self.alpha * self.St)
            dv_TM[self.St > 1] = self.c_s * torch.sqrt(2 * self.alpha / self.St[self.St > 1])
            self.dv[mesh2D_i == mesh2D_j] = (self.dv[mesh2D_i == mesh2D_j]**2 + dv_TM**2)**0.5
            self.calculate_kernels()
        elif test == 'BM+turb+fulldv':
            # full collisions with BM and turbulence from a simpler description
            self.frag_flag = False
            self.bouncing_flag = False
            self.vel_dist_flag = False
            self.mass_transfer_flag = False
            self.simple_St_flag = True
            self.dv = self.dv_BM
            v_TM = self.c_s * torch.sqrt(self.alpha * self.St)
            v_TM[self.St > 1] = self.c_s * torch.sqrt(self.alpha / self.St[self.St > 1])
            self.dv += torch.sqrt(v_TM**2 + v_TM[:, None]**2)
            self.calculate_kernels()
        elif test == 'constK':
            # constant kernel
            # manually set the number of m_0 to 1.0 (using Nk[1] b/c Nk[0] is for ghost zone)
            self.Nk[1] = self.kwargs.get("n_0", 1)
            self.sigma[1] = self.Nk[1] * 3 * self.m[1]
            self.Na[1] = self.Nk[1] * 3
            self._Sigma_d = self.get_Sigma_d(self.Nk)
            self.Sigma_d = self.get_Sigma_d(self.Nk)

            self.frag_flag = False
            self.bouncing_flag = False
            self.vel_dist_flag = False
            self.mass_transfer_flag = False
            self.simple_St_flag = True
            self.static_kernel_flag = True

            self.dv.fill_(1.0)
            self.geo_cs.fill_(self.kwargs.get("alpha_c", 1.0))
            self.vi_fac.fill_(1.0)
            self.calculate_kernels()
        elif test == 'sumK':
            # sum kernel
            # manually set the number of m_0 to 1.0 (using Nk[1] b/c Nk[0] is for ghost zone)
            self.Nk[1] = self.kwargs.get("n_0", 1)
            self.sigma[1] = self.Nk[1] * 3 * self.m[1]
            self.Na[1] = self.Nk[1] * 3
            self._Sigma_d = self.get_Sigma_d(self.Nk)
            self.Sigma_d = self.get_Sigma_d(self.Nk)

            self.frag_flag = False
            self.bouncing_flag = False
            self.vel_dist_flag = False
            self.mass_transfer_flag = False
            self.simple_St_flag = True
            self.static_kernel_flag = True

            self.dv.fill_(1.0)
            self.geo_cs = self.kwargs.get('beta_c', 1.0) * self.m_sum_ij
            self.vi_fac.fill_(1.0)
            self.calculate_kernels()
        elif test == "productK":
            # product kernel
            # manually set the number of m_0 to 1.0 (using Nk[1] b/c Nk[0] is for ghost zone)
            self.Nk[1] = self.kwargs.get("n_0", 1)
            self.sigma[1] = self.Nk[1] * 3 * self.m[1]
            self.Na[1] = self.Nk[1] * 3
            self._Sigma_d = self.get_Sigma_d(self.Nk)
            self.Sigma_d = self.get_Sigma_d(self.Nk)

            self.frag_flag = False
            self.bouncing_flag = False
            self.vel_dist_flag = False
            self.mass_transfer_flag = False
            self.simple_St_flag = True
            self.uni_gz_flag = 4  # let the last ghost zone join coagulation with others
            self.static_kernel_flag = True

            self.dv.fill_(1.0)
            self.geo_cs = self.kwargs.get('gamma_c', 1.0) * self.m_prod_ij
            self.vi_fac.fill_(1.0)
            self.calculate_kernels()
        else:
            raise ValueError(f"Unknown test case: {test}")

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def __init_update_solids(self):
        """ Initialized accretion info and calculate the loss/supply rate of solids """

        self.Mdot = self.kwargs.get('Mdot', 3e-9)                      # [Msun/yr]
        self.Raccu = self.kwargs.get('Raccu', 0.1)                     # [AU]
        self.Zacc = self.kwargs.get('Z', 0.01)                         # dust-to-gas ratio
        self.a_critD = self.kwargs.get('a_critD', 0.01)  # critical dust size that will be lifted [cm]

        self.S_annulus = 2 * torch.pi * self.Raccu * 0.1*self.Raccu * (u.au.to(u.cm))**2
        self.Sigma_dot = self.Mdot*((u.Msun/u.yr).to(u.g/u.s)) / (2*torch.pi * self.Raccu*(u.au.to(u.cm)) * self.H)

        a_min_in = self.kwargs.get('a_min_in', self.a[1])              # smallest solids drifting in [cm]
        a_max_in = self.kwargs.get('a_max_in', 10)                     # largest solids drifting in [cm]
        if a_min_in > a_max_in:
            self.warn(f"The size range of solids drifting in seems off: {a_min_in} > {a_max_in}. Reversed.")
            a_min_in, a_max_in = a_max_in, a_min_in
        a_idx_i = torch.argmin(abs(self.a - a_min_in)).item()
        a_idx_f = torch.argmin(abs(self.a - a_max_in)).item()
        a_idx_i = max(a_idx_i, 1)
        a_idx_i = min(a_idx_i, self.num_grid+1)
        a_idx_f = max(a_idx_f, 1)
        a_idx_i = min(a_idx_i, self.num_grid+1)

        tmp_sigma = torch.zeros_like(self.a)
        tmp_sigma[a_idx_i:a_idx_f+1] = self.a[a_idx_i:a_idx_f+1]**(0.5)  # MRN dist
        C_norm = self.Zacc * self.Sigma_dot / torch.sum(tmp_sigma * self.dlog_a)
        self.dsigma_in = tmp_sigma * C_norm

        self.Sigma_v = self.kwargs.get("Sigma_v", 1e-2 * self._Sigma_d)
        # Below we use Clausius-Clapeyron relation to describe the pressure-temperature profile,
        # where the original format is P_v_eq = P_v_eq0 * exp(-T_a'/T), but we use (DOI: 10.1088/2041-8205/767/1/L12)
        # log_10[P_v_eq(SiO2, liq)] = P_v_eq0 - log_10[exp(-T_a'/T)] = 8.203 − 25898.9/T
        # Note that we re-define T_a to be log_10(e) * T_a', where T_a'(SiO2, liq) ~ 6.0e4 K
        self.P_v_eq0 = self.kwargs.get('P_v_eq0', 10**8.203)           # the saturated pressure at T_a'
        self.T_a = self.kwargs.get('T_a', 25898.9)                     # np.log10(np.e) * T_a'

        if self.dyn_env_flag and self.T_kappa == "latent_heat":
            # calculate the initial kappa
            kappa_geom = self.coeff_kappa_geom * (self.sigma / self.Sigma_g)
            Q_e = torch.minimum(self.coeff_Q_e * torch.scalar_tensor(self.T), torch.scalar_tensor(2.0))
            self.kappa = torch.sum(kappa_geom * Q_e).item()
            self.log(f"The initial kappa = {self.kappa:.2e}.")

            if self.debug_flag:
                self.log("Now we adjust dust surface density to make kappa unity.")
                self.sigma /= self.kappa
                self.Nk = self.sigma / (3 * self.m)
                self.Na = self.sigma / self.m
                self.Sigma_d = self.get_Sigma_d(self.Nk)
                self._Sigma_d = self.Sigma_d
                kappa_geom = self.coeff_kappa_geom * (self.sigma / self.Sigma_g)
                self.kappa = torch.sum(kappa_geom * Q_e).item()
                self.kappa_i = self.kappa

                """ RL: previously we adjust alpha, Sigma_g to accommodate kappa's change, above we instead adjust kappa
                self.Raccu = self.kwargs.get('Raccu', 0.1)  # [AU]
                self.Mdot = self.kwargs.get('Mdot', 3e-9)  # [Msun/yr]
                alpha_Tkappa = (self.T/373)**(-5) * self.Raccu**(-9/2) * (self.Mdot/10**(-7.5))**(2) * (self.kappa/1)

                self.Sigma_g *= (alpha_Tkappa*0.01 / self.alpha)**(-0.7)
                self.alpha = alpha_Tkappa * 0.01  # the scale is on alpha_{-2}
                
                # update quantities depending on alpha
                self.rho_g0 = self.Sigma_g / (self.sqrt_2_pi * self.H)
                self.lambda_mpf = 1 / (self.rho_g0 / self.cgs_mu * self.sigma_H2)
                self.nu_mol = 0.5 * np.sqrt(8/np.pi) * self.c_s * self.lambda_mpf

                self.u_gas_TM = self.c_s * np.sqrt(3/2 * self.alpha)
                self.Re = self.alpha * self.Sigma_g * self.sigma_H2 / (2 * self.cgs_mu)
                self.St_12 = 1/1.6 * self.Re ** (-1/2)
                self.nu_disk = self.alpha * c.R.cgs.value * self.T / (2.3 * self.Omega)

                self.kappa_i = self.kappa
                self.Sigma_g_i = self.Sigma_g
                """

                # since dotQ_plus != 0 now, it won't be changed when T begins to vary
                # dotQ_plus is only used for calculating dT, where Sigma_g is eliminated on both side
                self.dotQ_plus = 2.25 * self.nu_disk * self.Omega**2  # * self.Sigma_g
                self.warn(f"Derived dotQ_plus[/Sigma_g] = {self.dotQ_plus:.3e}.")

    def update_solids(self, dt):
        """ Update the particle distribution every time step if needed """

        if self.closed_box_flag is False:
            # first, calculate sigma loss due to accretion funnels
            # this could be the last step; we keep it here to make results compatible with previous versions
            self.Hratio_loss = 1 - torch.special.erf(self.inv_sqrt_2 / (self.H_d / self.H))
            self.Hratio_loss[self.a > self.a_critD] = 0
            self.dsigma_out = self.sigma / self.Sigma_g * self.Hratio_loss * self.Sigma_dot
            if self.t < self.acc_ramp_time:  # False even if both = 0
                acc_ramp_ratio = self.t / self.acc_ramp_time
            else:
                acc_ramp_ratio = 1.0
            self.sigma -= torch.minimum(acc_ramp_ratio * self.dsigma_out * dt, self.sigma)

            # second, add dust supply from outer disk
            # this step is performed beforehand so the supply also go through evaporation/condensation
            self.sigma += acc_ramp_ratio * self.dsigma_in * dt
            # alternatively, the amount of supply may be regulated by the simplified treatment below
            # if self.dyn_env_flag and self.t >= self.dyn_env_burn_in and self.T > 2000:
            # temp_fac = np.exp(-(self.T - 2000) / (2010 - 2000))
            # self.sigma += self.dsigma_in * dt * temp_fac

            # update _Sigma_d that will be used in the third step
            self.Nk = self.sigma / (3 * self.m)
            self.Na = self.sigma / self.m
            self._Sigma_d = torch.sum(self.sigma * self.dlog_a).item()

        # third, calculate evaporation/condensation rate
        if self.dyn_env_flag and self.t >= self.dyn_env_burn_in:

            c_s_v = np.sqrt((self.cgs_k_B * self.T) / (self.cgs_mu * 60 / 2.3))
            self.P_v_eq = self.P_v_eq0 * 10**(- self.T_a / self.T) * self.bar2cgs
            R_e = 8 * self.sqrt_2_pi * self.a2_o_m / c_s_v * self.P_v_eq
            R_c = 8 * c_s_v * self.a2_o_m / self.H

            if self.ev_explicit:
                limited_dsigma_e = torch.minimum(R_e * self.sigma * dt, self.sigma)
                dot_Sigma_v2c = torch.sum(R_c * self.sigma * self.dlog_a * self.Sigma_v).item()
                if dot_Sigma_v2c * dt > self.Sigma_v:
                    tmp_dt = 0.99 * self.Sigma_v / dot_Sigma_v2c  # use 0.99 to avoid numerical error on summation
                    limited_dsigma_c = R_c * self.sigma * self.Sigma_v * tmp_dt
                else:
                    limited_dsigma_c = R_c * self.sigma * self.Sigma_v * dt
                self.sigma += limited_dsigma_c - limited_dsigma_e
                self.dSigma_e = torch.sum(limited_dsigma_e * self.dlog_a).item()
                self.dSigma_c = torch.sum(limited_dsigma_c * self.dlog_a).item()
                self.Sigma_v += self.dSigma_e - self.dSigma_c
                self.log(f"dSigma_e = {self.dSigma_e:.4e}, dSigma_c = {self.dSigma_c:.4e}, "
                         + f"Sigma_v = {self.Sigma_v}, Sigma_d = {torch.sum(self.sigma * self.dlog_a).item()}")

            if self.ev_semi_implicit:   # ev = evaporation/condensation
                R_c_prime = R_c * self.sigma * self.dlog_a
                ev_idx_non0 = self.sigma > 0  # use indices so it works for discontinuous distributions
                ev_non0 = torch.count_nonzero(ev_idx_non0).item()
                ev_A = torch.zeros([ev_non0+2, ev_non0+1], dtype=self.default_dtype, device=self.device)
                ev_B = torch.zeros(ev_non0+2, dtype=self.default_dtype, device=self.device)
                old_Sigma_dv = self._Sigma_d + self.Sigma_v

                ev_A[0, 0] = 1 + R_c_prime[ev_idx_non0].sum().item() * dt
                ev_A[1:-1, 1:] = torch.diag_embed(1 + R_e[ev_idx_non0] * dt)
                ev_A[0, 1:] = -R_e[ev_idx_non0] * dt
                ev_A[1:-1, 0] = -R_c_prime[ev_idx_non0] * dt
                ev_A[-1].fill_(100)  # use 100 to give more weight to mass conservation
                ev_B[0] = self.Sigma_v
                ev_B[1:-1] = self.sigma[ev_idx_non0] * self.dlog_a
                ev_B[-1] = 100 * (self.Sigma_v + self._Sigma_d)

                if True:
                    _A = torch.clone(ev_A[:-1, :])
                    if self._torch_113_flag:
                        _A = _A.mT.contiguous().mT
                    ev_sln = torch.linalg.solve(_A, ev_B[:-1])

                    tmp_sigma = torch.zeros_like(self.sigma)
                    tmp_sigma[ev_idx_non0] = ev_sln[1:] / self.dlog_a
                    new_Sigma_dv = (torch.sum(tmp_sigma * self.dlog_a) + ev_sln[0]).item()
                    tmp_rerr = abs((new_Sigma_dv - old_Sigma_dv) / old_Sigma_dv)
                    if tmp_rerr > self.rerr_th:
                        self.warn("Relative error is somewhat large in evaporation/condensation calculations: "
                                  f"Cond(_A) = {torch.linalg.cond(_A):.3e}; "
                                  f"Delta Sigma_d+v / Sigma_d+v = {tmp_rerr:.3e}. Consider using a smaller timestep.")
                else:
                    # it seems mass bins with tiny sigma would unphysically gain mass when using lstsq
                    ev_sln = torch.linalg.lstsq(ev_A, ev_B)

                    tmp_sigma = torch.zeros_like(self.sigma)
                    tmp_sigma[ev_idx_non0] = ev_sln.solution[1:] / self.dlog_a
                    new_Sigma_dv = (torch.sum(tmp_sigma * self.dlog_a) + ev_sln.solution[0]).item()
                    tmp_rerr = abs((new_Sigma_dv - old_Sigma_dv) / old_Sigma_dv)
                    if tmp_rerr > self.rerr_th4dt:
                        self.log("weight for mass conservation adjusted to reduce relative error"
                                 + "in evaporation/condensation calculations.")
                        ev_A[-1].fill_(1000)
                        ev_B[-1] = 1000 * (self.Sigma_v + self._Sigma_d)
                        ev_sln = torch.linalg.lstsq(ev_A, ev_B)
                        tmp_sigma[ev_idx_non0] = ev_sln.solution[1:] / self.dlog_a
                        new_Sigma_dv = (torch.sum(tmp_sigma * self.dlog_a) + ev_sln.solution[0]).item()
                        tmp_rerr = abs((new_Sigma_dv - old_Sigma_dv) / old_Sigma_dv)
                    if tmp_rerr > self.rerr_th:
                        self.warn("Relative error is somewhat large in evaporation/condensation calculations: "
                                  f"Delta Sigma_d+v / Sigma_d+v = {tmp_rerr:.3e}. Consider using a smaller timestep.")

                # deal with possible negative numbers (usually outside a_supp)
                # also, set a surface density floor, so condensation is possible after almost all dust sublimates
                if torch.any(tmp_sigma < 0):
                    tmp_sigma[(tmp_sigma < 0) & (self.Nk == 0)] = 0  # if nothing was there, set to zero

                    # tiny_idx = (tmp_sigma < 0) & (abs(tmp_sigma) / tmp_sigma.sum() < 1e-8)
                    # The line above may fail if most solids have been vaporized such that tmp_sigma is already tiny
                    # Instead, we use the total surface density (we adopt 1e-10 since lstsq gives limited accuracy)
                    tiny_idx = (tmp_sigma < 0) & (torch.abs(tmp_sigma * self.dlog_a) / old_Sigma_dv < 1e-10)
                    tmp_sigma[tiny_idx] = 0

                # if there was something, set to surface density floor;
                # such a floor corresponds to sigma(1e-4 cm) = 4.6e-03, Sigma_d ~ 0.001 g/cm^2 at 0.1 au
                tiny_idx = (tmp_sigma == 0) & (self.a < 1e-2)
                tmp_sigma[tiny_idx] = torch.maximum(tmp_sigma[tiny_idx],
                                                    torch.scalar_tensor((old_Sigma_dv / self.dlog_a) * 1e-15))

                new_Sigma_dv = (torch.sum(tmp_sigma * self.dlog_a) + ev_sln[0]).item()  #.solution
                if torch.any(tmp_sigma < 0):
                    self.warn("Negative sigma remains. Calculations may encounter problems.")

                enforce_mass_con_ratio = old_Sigma_dv / new_Sigma_dv
                self.Sigma_v = ev_sln[0].item() * enforce_mass_con_ratio  #.solution
                self.sigma = tmp_sigma * enforce_mass_con_ratio
                if self.Sigma_v < 0:
                    self.warn("Vapor surface density becomes negative. Enforce it back to zero now. "
                              + "Smaller timestep is recommended.")
                    tmp_Sigma_d = torch.sum(self.sigma * self.dlog_a).item()
                    enforce_mass_con_ratio = (tmp_Sigma_d + self.Sigma_v) / tmp_Sigma_d
                    self.sigma *= enforce_mass_con_ratio
                    self.Sigma_v = 0
                # _Sigma_d hasn't updated yet (positive if evaporate more, negative if condense more)
                self.dSigma_ec = self._Sigma_d - (old_Sigma_dv - self.Sigma_v)
                #self.log(f"Sigma_v = {self.Sigma_v}, Sigma_d = {torch.sum(self.sigma * self.dlog_a).item()}")
                self.dSigma_v2d = R_c_prime * dt * self.Sigma_v
                self.dSigma_d2v = R_e * dt * self.sigma * self.dlog_a

            # finally, update Nk and total surface density
            self.Nk = self.sigma / (3 * self.m)
            self.Na = self.sigma / self.m
            self._Sigma_d = torch.sum(self.sigma * self.dlog_a).item()
            if self.closed_box_flag is False and self.tmp_debug_flag is False:
                # accrete Sigma_v accordingly
                dSigma_v_acc = self.Sigma_v * (self.Sigma_dot * dt) / self.Sigma_g
                self.Sigma_v -= min(dSigma_v_acc, self.Sigma_v)

    def _get_dN_before_torch_1_13(self, dt):
        """ Get the current dN for debug use """

        self.S = self.Nk @ (self.Nk @ self.tM)  # M_ijk N_i N_j equals to M_jik N_i N_j
        self.J = self.Nk @ (self.tM + torch.swapaxes(self.tM, 0, 1))
        return torch.linalg.solve(self.I / dt - self.J.transpose(0, 1), self.S)

    def _get_dN_after_torch_1_13(self, dt):
        """ Get the current dN for debug use """

        self.S = self.Nk @ (self.Nk @ self.tM)  # M_ijk N_i N_j equals to M_jik N_i N_j
        self.J = self.Nk @ (self.tM + torch.swapaxes(self.tM, 0, 1))
        _A = self.I / dt - self.J.transpose(0, 1)
        _A = _A.mT.contiguous().mT  # see https://github.com/pytorch/pytorch/issues/90453
        return torch.linalg.solve(_A, self.S)

    def _one_step_implicit_before_torch_1_13(self, dt):
        """ Group matrix solving part for future optimization

        Notes: When mass grid spans more than 16 orders of magnitude, it gives ill-conditioned matrix warning
        (use np.linalg.cond to check the condition number) due to fragmentation (b/c the fragments of larger solids
        during a collision is distributed across the entire mass grid smaller than themselves).

        Solving an ill-conditioned linear system will still give "exact" results (numpy/scipy, mpmath, Mathematica, or
        Matlab all give the same results within machine/working precision), but you'll get wildly different results for
        small perturbations of A or b.

        In this function, we filter out the ill-conditioned matrix warning.
        """

        self.S = self.Nk @ (self.Nk @ self.tM)  # M_ijk N_i N_j equals to M_jik N_i N_j
        self.J = self.Nk @ (self.tM + torch.swapaxes(self.tM, 0, 1))
        self.dN = torch.linalg.solve(self.I / dt - self.J.transpose(0, 1), self.S)

    def _one_step_implicit_after_torch_1_13(self, dt):
        self.S = self.Nk @ (self.Nk @ self.tM)  # M_ijk N_i N_j equals to M_jik N_i N_j
        self.J = self.Nk @ (self.tM + torch.swapaxes(self.tM, 0, 1))
        _A = self.I / dt - self.J.transpose(0, 1)
        _A = _A.mT.contiguous().mT  # see https://github.com/pytorch/pytorch/issues/90453
        self.dN = torch.linalg.solve(_A, self.S)

    def one_step_implicit(self, dt):
        """ Evolve the particle distribution for dt with one implicit step"""

        #if self._Sigma_d == 0 and not self.static_kernel_flag:  # in case all solids become vapor
        #    self.update_kernels()
        #    return None

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
            self.log(f"continue using dyn dt failed with rerr={tmp_rerr:.3e}; revert back to original dt.")
            dt = _dt
            self._one_step_implicit(dt)
            tmp_rerr = abs(self.get_Sigma_d(self.dN) / self._Sigma_d)

        if tmp_rerr > self.rerr_th:
            if tmp_rerr > self.rerr_th4dt:
                dt /= (tmp_rerr / self.rerr_th4dt) * 2
                self.log("dt adjusted temporarily to reduce the relative error: tmp dt = "+f"{dt / self.s2y:.3e}")
                self._one_step_implicit(dt)
                tmp_rerr = abs(self.get_Sigma_d(self.dN) / self._Sigma_d)
            if tmp_rerr > self.rerr_th:  # rerr_th may < rerr_th4dt, so only a warning is given, no re-calculations
                self.warn("Relative error is somewhat large: sum(dSigma(dN))/Sigma_d = "
                          f"{tmp_rerr:.3e}. Consider using a smaller timestep "
                          "(a higher resolution mass grid usually won't help).")
        else:
            # only try dynamic_dt when both conditions are meet (to avoid back and forth)
            if self.dynamic_dt is True and self.dyn_dt_success is False and skip_this_cycle is False:
                dyn_dt = self.dyn_dt * self.s2y
                if tmp_rerr * (dyn_dt / dt) < self.tol_dyndt:
                    tmp_dN = torch.clone(self.dN)
                    self._one_step_implicit(dyn_dt)
                    tmp_rerr = abs(self.get_Sigma_d(self.dN) / self._Sigma_d)
                    if tmp_rerr <= self.tol_dyndt:
                        dt = dyn_dt
                        self.dyn_dt_success = True
                        self.log(f"dynamic dt used to speed up this run: dyn dt={dt/self.s2y:.3e}; continue with it")
                    else:
                        self.log(f"dynamic dt attempt failed with rerr={tmp_rerr:.3e}, revert back to original dt.")
                        self.dN = tmp_dN

        # handle possible negative numbers (usually due to large dt)
        tmp_Nk = self.Nk + self.dN
        loop_count = 0
        while torch.any(tmp_Nk < 0):
            tmp_Nk[(tmp_Nk < 0) & (self.sigma == 0)] = 0  # if nothing was there, set to zero

            # what we want is the conservation of mass, so instead of Nk, we should check mass in each bin
            # previous code on Nk may lead to larger relative error than expected
            tmp_mass = tmp_Nk * self.m  # for checking purpose, no need to include factor 3 and self.dlog_a
            tiny_idx = (tmp_mass < 0) & (abs(tmp_mass) / tmp_mass.sum() < self.neg2o_th)
            tmp_Nk[tiny_idx] = 0  # if contribute little to total mass, reset to zero

            # if negative values persist
            if torch.any(tmp_Nk < 0):
                # external solid supply may also cause this by creating a discontinuity
                if not self.closed_box_flag:
                    if not torch.any(tmp_Nk * 3 * self.m + self.dsigma_in * dt < 0):
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
            self.log("dt reduced to prevent negative Nk: new dt = " + f"{dt / self.s2y:.3e}")
        self.Nk = torch.clone(tmp_Nk)

        self.sigma = self.Nk * 3 * self.m
        self.Na = self.Nk * 3
        self.dt = dt / self.s2y  # dt should be in units of sec, self.dt in units of yr
        self.rerr = (self.get_Sigma_d(self.Nk) - self._Sigma_d) / self._Sigma_d

        if not self.static_kernel_flag:
            self.update_kernels()  # put it at the cycle end so self.dt can be used in update_disk_parameters

    def enforce_mass_con(self):
        """ Enforce the conservation of total solid mass

        It is impossible to avoid rounding errors (see discussion in Garaud+2013).
        Here, we follow their method to enforce mass conservation but we enforce it every timestep.
        """
        if self._Sigma_d == 0:  # in case all solids become vapor
            return None
        self.Nk *= (self._Sigma_d / self.get_Sigma_d(self.Nk))
        self.sigma = self.Nk * 3 * self.m
        self.Na = self.Nk * 3

    def one_step_for_animation(self, dt):
        """ Evolve one step implicitly with post processing for animation use """

        self.one_step_implicit(dt)
        self.enforce_mass_con()
        self.update_solids(dt)

    def dump_bin_data(self, first_dump=False, open4restart=False):
        """ Dump run data to a file """

        # not sure if this is the best practice if using GPU
        if first_dump:
            # self.dlog_a and self.t is np.float64, so res4out becomes np.float64
            if self.dyn_env_flag:
                if self.debug_flag:
                    self.res4out = np.hstack([self.dlog_a, self.m.cpu().numpy(), self.a.cpu().numpy(),
                                              np.zeros(self.num_grid + 2), np.zeros(self.num_grid+2),
                                              self.kappa, self.Sigma_g, self.Sigma_v, self.T])
                else:
                    self.res4out = np.hstack([self.dlog_a, self.m.cpu().numpy(), self.a.cpu().numpy(),
                                          self.kappa, self.Sigma_g, self.Sigma_v, self.T])
            else:
                self.res4out = np.hstack([self.dlog_a, self.m.cpu().numpy(), self.a.cpu().numpy()])
            self.dat_file = open(self.dat_file_name, 'wb')
            self.dat_file.write(self.num_grid.to_bytes(4, 'little'))
            # for device='cpu', cpu() won't do anything, no copy
            self.dat_file.write(self.res4out.tobytes())
            self.dat_file.close()
            self.dat_file = open(self.dat_file_name, 'ab')
        else:
            if self.dyn_env_flag:
                if self.debug_flag:
                    self.res4out = np.hstack([self.t, self.sigma.cpu().numpy(), self.Nk.cpu().numpy(),
                                              self.dSigma_v2d.cpu().numpy(), self.dSigma_d2v.cpu().numpy(),
                                              self.kappa, self.Sigma_g, self.Sigma_v, self.T])
                else:
                    self.res4out = np.hstack([self.t, self.sigma.cpu().numpy(), self.Nk.cpu().numpy(),
                                          self.kappa, self.Sigma_g, self.Sigma_v, self.T])
            else:
                self.res4out = np.hstack([self.t, self.sigma.cpu().numpy(), self.Nk.cpu().numpy()])
            self.dat_file.write(self.res4out.tobytes())
            self.dat_file.flush()

    def run(self, tlim, dt, out_dt, cycle_limit=1000000000,
            burnin_dt=1/365.25, no_burnin=False, burnin_out_dt_ratio=1.0,
            ramp_speed=1.01, dynamic_dt=False, out_log=True):
        """
        Run simulations and dump results

        Parameters
        ----------
        tlim : float
            total run time [year]
        dt : float
            time step [year]
        out_dt : float
            time interval to dump data [year]
        cycle_limit : int
            number of max cycles, run will terminate if it is exceeded, default: 1e9
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
            self.log, self.warn, self.err = logger.info, logger.warning, logger.error

        dump_func = self.dump_bin_data  # dump binary data by default; ascii has been deprecated

        s_time = time.perf_counter()
        if self.t > 0 and self.cycle_count > 0:
            self.log(f"===== Simulation restarts now =====")
            self.dat_file = open(self.dat_file_name, 'ab')
            self.log(f"cycle={self.cycle_count}, t={self.t:.6e}, dt={dt:.3e}, rerr(Sigma_d)={self.rerr:.3e}")
            self.out_dt = out_dt
            self.next_out_t = self.t + self.out_dt
        else:
            self.log(f"===== Simulation begins now =====")
            dump_func(first_dump=True)
            self.cycle_count = 0
            self.log(f"cycle={self.cycle_count}, t={self.t:.6e}, dt={dt:.3e}, rerr(Sigma_d)={self.rerr:.3e}")
            self.out_dt = out_dt
            dump_func()
            self.next_out_t = self.t + self.out_dt

        # let's have some slow burn-in steps if not a restart
        # so we can enter the relatively-smooth profile gradually from the initial profile with discontinuities
        tmp_dt = burnin_dt
        if self.cycle_count == 0 and no_burnin is False:
            dt, burnin_dt = burnin_dt, dt
            self.next_out_t = self.t + self.out_dt * burnin_out_dt_ratio
            while self.t + dt < min(tlim, 1) and self.cycle_count < cycle_limit:  # alternatively, we can use 4 yrs
                pre_step_t = time.perf_counter()
                self.one_step_implicit(dt * self.s2y)
                self.enforce_mass_con()
                self.update_solids(self.dt * self.s2y)  # dt could have changed
                self.t += self.dt
                self.cycle_count += 1
                post_step_t = time.perf_counter()
                self.log(f"cycle={self.cycle_count}, t={self.t:.6e}yr, dt={self.dt:.3e}, T={self.T:.1f}, "
                         f"rerr(Sigma_d)={self.rerr:.3e}, rt={post_step_t - pre_step_t:.3e}")
                if out_log and self.cycle_count % 100 == 0:
                    fh.flush()
                if self.t > self.next_out_t - dt / 2:
                    dump_func()
                    self.next_out_t = self.t + self.out_dt * burnin_out_dt_ratio
            dt, burnin_dt = burnin_dt, dt

            if ramp_speed <= 1.0 or ramp_speed >= 5:
                self.warn(f"ramp_speed should be slightly larger than 1.0; got: {ramp_speed}; fall back to 1.01.")
                ramp_speed = 1.01
            # then let the burn-in dt gradually ramp up to match the desired dt
            # N.B., with increasing dt, we might miss the next_out_t and lose all the data
            # if we only output within (out_t-dt/2, out_t+dt/2)
            while (self.t + tmp_dt < tlim) and (tmp_dt * ramp_speed < dt) and (self.cycle_count < cycle_limit):
                tmp_dt *= ramp_speed
                pre_step_t = time.perf_counter()
                self.one_step_implicit(tmp_dt * self.s2y)
                self.enforce_mass_con()
                self.update_solids(self.dt * self.s2y)  # dt could have changed
                self.t += self.dt
                self.cycle_count += 1
                post_step_t = time.perf_counter()
                self.log(f"cycle={self.cycle_count}, t={self.t:.6e}yr, dt={self.dt:.3e}, T={self.T:.1f}, "
                         f"rerr(Sigma_d)={self.rerr:.3e}, rt={post_step_t - pre_step_t:.3e}")
                if out_log and self.cycle_count % 100 == 0:
                    fh.flush()
                if self.t > self.next_out_t - tmp_dt / 2:
                    dump_func()
                    self.next_out_t = self.t + self.out_dt * burnin_out_dt_ratio

        # turn on dynamic dt if specified
        if dynamic_dt:
            if self.dyn_dt / dt < 4 or self.out_dt / dt < 4:
                self.warn(
                    f"Dynamic dt not enabled: dyn_dt/dt={self.dyn_dt / dt:.1f}, out_dt/dt={self.out_dt / dt:.1f}. "
                    f"One of them is smaller than 4, which won't result in a significant speed gain.")
            else:
                # continue to dyn_dt from tmp_dt (RL: disabled, too many burnin steps)
                if False:  # no_burnin is False:
                    while (self.t + tmp_dt < tlim) and (tmp_dt * ramp_speed < self.dyn_dt):
                        # tmp_dt += burnin_dt  # alternatively, we can use *= 1.01
                        tmp_dt *= ramp_speed
                        pre_step_t = time.perf_counter()
                        self.one_step_implicit(tmp_dt * self.s2y)
                        self.enforce_mass_con()
                        self.update_solids(self.dt * self.s2y)  # dt could have changed
                        self.t += self.dt
                        self.cycle_count += 1
                        post_step_t = time.perf_counter()
                        self.log(f"cycle={self.cycle_count}, t={self.t:.6e}yr, dt={self.dt:.3e}, T={self.T:.1f}, "
                                 f"rerr(Sigma_d)={self.rerr:.3e}, rt={post_step_t - pre_step_t:.3e}")
                        if out_log and self.cycle_count % 100 == 0:
                            fh.flush()
                        if self.t > self.next_out_t - tmp_dt / 2:
                            dump_func()
                            self.next_out_t = self.t + self.out_dt
                # now turn on dynamic_dt
                self.dynamic_dt = True
                if self.dyn_dt > self.out_dt:
                    self.warn(f"The desired dynamic dt is reduced to out_dt (={out_dt:.1f}) to secure data.")
                    self.dyn_dt = self.out_dt
                if self.tol_dyndt > min(self.rerr_th, self.rerr_th4dt):
                    self.tol_dyndt = min(self.rerr_th, self.rerr_th4dt)
                    self.warn("The relative error tolerance for using dynamic dt should be smaller than "
                              f"min(rerr_th, rerr_th4dt). Adjusted automatically to {self.tol_dyndt}.")

        # Now on to tlim with the input dt
        while self.t + dt < tlim and self.cycle_count < cycle_limit:
            pre_step_t = time.perf_counter()
            self.one_step_implicit(dt * self.s2y)
            self.enforce_mass_con()
            self.update_solids(self.dt * self.s2y)  # dt could have changed
            self.t += self.dt
            self.cycle_count += 1
            post_step_t = time.perf_counter()
            self.log(f"cycle={self.cycle_count}, t={self.t:.6e}yr, dt={self.dt:.3e}, T={self.T:.1f}, "
                     f"rerr(Sigma_d)={self.rerr:.3e}, rt={post_step_t - pre_step_t:.3e}")
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
        if self.cycle_count < cycle_limit:
            pre_step_t = time.perf_counter()
            self.one_step_implicit(dt * self.s2y)
            self.enforce_mass_con()
            self.update_solids(self.dt * self.s2y)  # dt could have changed
            self.t += self.dt
            self.cycle_count += 1
            post_step_t = time.perf_counter()

            dump_func()
            if self.t > self.next_out_t - dt / 2:  # advance next output time anyway
                self.next_out_t = self.t + self.out_dt
            self.log(f"cycle={self.cycle_count}, t={self.t:.6e}yr, dt={self.dt:.3e}, T={self.T:.1f}, "
                     f"rerr(Sigma_d)={self.rerr:.3e}, rt={post_step_t - pre_step_t:.3e}")

        elapsed_time = time.perf_counter() - s_time
        self.log(f"===== Simulation ends now =====\n" + '*' * 80 +
                 f"\nRun stats:\n\tElapsed wall time: {elapsed_time:.6e} sec"
                 f"\n\tCycles / wall second: {self.cycle_count / elapsed_time:.6e}")
        self.dat_file.close()
        if out_log:
            fh.flush()
            fh.close()
            logger.removeHandler(fh)
            logging.shutdown()
            self.log, self.warn, self.err = print, print, print
