from scipy import signal as spsig
import scipy as sp
import numpy as np
from tqdm import tqdm
from . import BOLDgrid, BOLDsequence

class DeterministicDiffuser2D:

    def __init__(
        self,
        ADC: float,
        B0: float,
        num_dt: int,
        dt: float,
        edge_width: float=0,
    ):
        self.ADC = ADC
        self.B0 = B0
        self.num_dt = num_dt
        self.dt = dt
        self.edge_width = edge_width
        self.kernel = None
        self.N = None

    def signal(
        self, 
        grid: BOLDgrid.Grid2D, 
        sequence: BOLDsequence.Sequence,
        kernel_type: str='ModifiedBessel', 
        cplx: bool=False, 
        permeable: bool=True,
        progressbar: bool=True
    ):        
        self.N = grid.N
        self.size = grid.size
        ip_map = grid.mask > 0
        ep_map = np.logical_not(ip_map)
        phase = grid.phase

        signalEV = np.zeros(self.num_dt+1) + 0j
        signalIV = np.zeros(self.num_dt+1) + 0j
        signal = np.zeros(self.num_dt+1) + 0j
        
        grid_range = np.linspace(-self.size/2, self.size/2, self.N)
        
        #calculate the diffusion distance
        sigma_sq = self.ADC*2*self.dt*0.001
        
        X, Y = np.meshgrid(grid_range, grid_range)
        
        self.make_kernel(X, Y, sigma_sq, kernel_type)

        phase = self.offset_all(phase, sequence.dB0)
        
        edge = int(np.floor(self.N * self.edge_width))
        nnzEV = np.sum(ep_map[edge:-edge-1,edge:-edge-1])
        nnzIV = np.sum(ip_map[edge:-edge-1,edge:-edge-1])
        
        Mx = np.zeros((self.N, self.N))
        My = np.zeros((self.N, self.N))
        Mz = np.zeros((self.N, self.N))
        
        Mz[:,:] = 1
        
        Mx, My, Mz = self.rot_any(Mx, My, Mz, sequence.pulse_axes[0][0], sequence.pulse_axes[0][1], sequence.pulse_angles[0])
        
        signalEV[0] = np.sum( (ep_map * (Mx + 1j*My))[edge:-edge-1,edge:-edge-1] ) / nnzEV
        signalIV[0] = np.sum( (ip_map * (Mx + 1j*My))[edge:-edge-1,edge:-edge-1] ) / nnzIV
        signal[0] = (signalEV[0] * nnzEV + signalIV[0] * nnzIV)/ (nnzEV + nnzIV)
        
        intervals = list(sequence.pulse_time_indices)
        
        current_interval = 0
        text = 'Applying Deterministic Diffusion'
        for j in tqdm(range(1, self.num_dt+1), desc=text, disable=not progressbar):
            if permeable:
                Mx, My = self.diffusion_convolution_permeable(Mx, My, kernel_type) 
            else:
                Mx, My = self.diffusion_convolution_impermeable(Mx, My, ip_map, ep_map, kernel_type) 
            
            Mx, My = self.rot_phase(Mx, My, phase)
            
            if j in intervals:
                current_interval += 1
                angle = sequence.pulse_angles[current_interval]*sequence.dB1fact
                Mx, My, Mz = self.rot_any(Mx, My, Mz, sequence.pulse_axes[current_interval][0], sequence.pulse_axes[current_interval][1], angle)
            
            if sequence.T1IV is not None:
                Mz = self.T1decay(Mz, j-intervals(current_interval), ip_map, sequence.T1IV)
            if sequence.T1EV is not None:
                Mz = self.T1decay(Mz, j-intervals(current_interval), ep_map, sequence.T1EV)    
            if sequence.T2IV is not None:
                Mx, My = self.T2decay(Mx, My, j-intervals(current_interval), ip_map, sequence.T2IV)
            if sequence.T2EV is not None:
                Mx, My = self.T2decay(Mx, My, j-intervals(current_interval), ep_map, sequence.T2EV)
            
            MxEV = Mx * ep_map
            MyEV = My * ep_map
            MxIV = Mx * ip_map
            MyIV = My * ip_map
                
            signalEV[j] = np.sum(MxEV[edge:-edge-1,edge:-edge-1] + 1j*MyEV[edge:-edge-1,edge:-edge-1]) / nnzEV
            signalIV[j] = np.sum(MxIV[edge:-edge-1,edge:-edge-1] + 1j*MyIV[edge:-edge-1,edge:-edge-1]) / nnzIV
            signal[j] = (signalEV[j] * nnzEV + signalIV[j] * nnzIV)/ (nnzEV + nnzIV)    
            
        if not cplx:
            signalEV = np.abs(signalEV)
            signalIV = np.abs(signalIV)
            signal = np.abs(signal)
        
        return signal, signalEV, signalIV,
    
    def make_kernel(
        self, 
        X: np.ndarray, 
        Y: np.ndarray, 
        sigma_sq: float, 
        kernel_type: str
    ):
        if kernel_type == 'Gaussian':
            kernel = np.exp(-((X ** 2 +  Y ** 2) / (2 * sigma_sq)))
            kernel /= np.sum(kernel)
        
        elif kernel_type == 'ModifiedBessel':
            sigma = np.sqrt(sigma_sq)
            dx = self.size/self.N
            Nhw = int(np.ceil(6*sigma/dx))
            kernel_range = np.abs(np.arange(-Nhw, Nhw+1))
            kernel = np.exp(-(sigma/dx)**2)*sp.special.iv(kernel_range, (sigma/dx)**2)   
        
        self.kernel=kernel
    
    def diffusion_convolution_permeable(
        self, 
        Mx: np.ndarray, 
        My: np.ndarray, 
        kernel_type: str
    ):
        if kernel_type == 'Gaussian':
            Mx = spsig.fftconvolve(Mx, self.kernel, mode='same')
            My = spsig.fftconvolve(My, self.kernel, mode='same')
       
        elif kernel_type == 'ModifiedBessel':
            Mx = sp.ndimage.convolve1d(Mx, self.kernel, axis=0, mode='wrap')
            Mx = sp.ndimage.convolve1d(Mx, self.kernel, axis=1, mode='wrap')
            
            My = sp.ndimage.convolve1d(My, self.kernel, axis=0, mode='wrap')
            My = sp.ndimage.convolve1d(My, self.kernel, axis=1, mode='wrap')
        
        return Mx, My
    
    def diffusion_convolution_impermeable(
        self, 
        Mx: np.ndarray, 
        My: np.ndarray, 
        ip_map: np.ndarray, 
        ep_map: np.ndarray,
        kernel_type: str 
    ):
        
        conep, conip = self.diffusion_convolution_permeable(ep_map * 1.0, ip_map * 1.0, kernel_type)

        Wep = conip * ep_map
        Wip = conep * ip_map
        Mxep = Mx * ep_map 
        Myep = My * ep_map
        Mxip = Mx * ip_map
        Myip = My * ip_map
             
        Mxep_diff, Myep_diff = self.diffusion_convolution_permeable(Mxep, Myep, kernel_type)
        Mxep = Mxep_diff * ep_map + Wep * Mxep
        Myep = Myep_diff * ep_map + Wep * Myep
        
        Mxip_diff, Myip_diff = self.diffusion_convolution_permeable(Mxip, Myip, kernel_type)
        Mxip = Mxip_diff * ip_map + Wip * Mxip
        Myip = Myip_diff * ip_map + Wip * Myip

        Mx = Mxep + Mxip
        My = Myep + Myip
        
        return Mx, My

    def offset_all(self, phase: np.ndarray, dB0: float):
        
        phase += 2*np.pi*42.58e6*dB0*(self.dt*0.001)

        return phase

    def rot_any(
        self, 
        Mx: np.ndarray, 
        My: np.ndarray, 
        Mz: np.ndarray, 
        ax_theta: float, 
        ax_phi: float, 
        angle: float
    ): 
        #https://sites.google.com/site/glennmurray/Home/rotation-matrices-and-formulas/rotation-about-an-arbitrary-axis-in-3-dimensions
        
        u = np.sin(ax_theta)*np.cos(ax_phi)
        v = np.sin(ax_theta)*np.sin(ax_phi)
        w = np.cos(ax_theta)
        
        cangle = np.cos(angle)
        sangle = np.sin(angle)
        term = (u*Mx + v*My + w*Mz)*(1-cangle)
        
        Mx_out = u*term + Mx*cangle + (-w*My + v*Mz)*sangle
        My_out = v*term + My*cangle + (w*Mx - u*Mz)*sangle
        Mz_out = w*term + Mz*cangle + (-v*Mx + u*My)*sangle
        
        return Mx_out, My_out, Mz_out
    
    def rot_phase(self, Mx: np.ndarray, My: np.ndarray, phase: np.ndarray):
        
        cphase = np.cos(phase)
        sphase = np.sin(phase)
        
        Mx_out = Mx*cphase - My*sphase
        My_out = Mx*sphase + My*cphase
        
        return Mx_out, My_out
    
    def T1decay(self, Mz: np.ndarray, t: float, mask: np.ndarray, T1: float):       
        E1 = np.exp(-t/T1)
        Mz[mask] = Mz[mask]*E1+(1-E1)
        return Mz
        
    def T2decay(self, Mx: np.ndarray, My: np.ndarray, t: float, mask: np.ndarray, T2: float):
        E2=np.exp(-t/T2)
        Mx[mask] *= E2
        My[mask] *= E2
        return Mx, My