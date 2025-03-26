import h5py
import hdf5plugin
import numpy as np
import os
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree
from ase.io import read
from skimage.measure import block_reduce
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.integrate import quad
import glob
import pandas as pd
import ast
import re

class h5File_ID02:
    def __init__(self,file):
        self.file=file
        self.file_number=self.extract_number()
               
        if file is None:
            print("Please specify a data file path")
        if "_waxs_" in file:
            with h5py.File(file, "r") as f:
                group = list(f.keys())[0]  # Retrieve first group key
                self.title=str(f[group+'/instrument/id02-rayonixhs-waxs/header/Title'][()].decode('utf-8'))
                self.nb_frames = int(f[group + '/instrument/id02-rayonixhs-waxs/acquisition/nb_frames'][()])
                self.acq_time = float(f[group + '/instrument/id02-rayonixhs-waxs/acquisition/exposure_time'][()])
                
                # Retrieve image data
                target = group + '/measurement/data'
                data = np.array(f[target])
                self.data = np.mean(data, axis=0)  # Average over frames
                shape = np.shape(self.data)
                self.num_pixel_x = shape[0]
                self.num_pixel_z = shape[1]
                
                # Retrieve header information
                header = group + '/instrument/id02-rayonixhs-waxs/header'
                self.pixel_size_x = float(f[header + '/PSize_1'][()].decode('utf-8'))
                self.pixel_size_z = float(f[header + '/PSize_2'][()].decode('utf-8'))
                self.wl = float(f[header + '/WaveLength'][()])
                self.x_center = float(f[header + '/Center_1'][()].decode('utf-8'))
                self.z_center = float(f[header + '/Center_2'][()].decode('utf-8'))
                self.D = float(f[header + '/SampleDistance'][()].decode('utf-8'))

                # Retrieve binning info
                header = '/entry_0000/instrument/id02-rayonixhs-waxs/image_operation/binning'
                self.bin_x = f[header + '/x'][()]
                self.bin_y = f[header + '/y'][()]
        
        elif "_eiger2_" in file:
            with h5py.File(file, "r") as f:
                group = list(f.keys())[0]  # Retrieve first group key
                self.nb_frames = int(f[group + '/instrument/id02-eiger2-saxs/acquisition/nb_frames'][()])
                self.acq_time = float(f[group + '/instrument/id02-eiger2-saxs/acquisition/exposure_time'][()])
                
                # Retrieve image data
                target = group + '/measurement/data'
                self.data = np.array(f[target])
                shape = np.shape(self.data)
                self.num_pixel_x = shape[0]
                self.num_pixel_z = shape[1]
                
                # Retrieve header information
                header = '/entry_0000/instrument/id02-eiger2-saxs/header'
                self.pixel_size_x = float(f[header + '/PSize_1'][()].decode('utf-8'))
                self.pixel_size_z = float(f[header + '/PSize_2'][()].decode('utf-8'))
                self.wl = float(f[header + '/WaveLength'][()])
                self.x_center = float(f[header + '/Center_1'][()].decode('utf-8'))
                self.z_center = float(f[header + '/Center_2'][()].decode('utf-8'))
                self.D = f[header + '/SampleDistance'][()]

                # Retrieve binning info
                header = '/entry_0000/instrument/id02-eiger2-saxs/image_operation/binning'
                self.bin_x = f[header + '/x'][()]
                self.bin_y = f[header + '/y'][()]
        self.samplename=self.extract_sample_name()
        self.B=self.extract_magnetic_field()

    def extract_magnetic_field(self):
        # Match patterns like '100mT', '1T', '1.4T', etc.
        match = re.search(r'(\d+(\.\d+)?)(mT|T)', self.title)
        
        if match:
            value, _, unit = match.groups()
            value = float(value)
            
            # Convert T to mT
            if unit == 'T':
                value *= 1000

            return int(value)
        else:
            # If no magnetic field is found, return '0mT'
            return 0
    

    def extract_sample_name(self):
        """Extracts the sample name from a string before the magnetic field value."""
        pattern = re.compile(r"^(.*?)(?:_\d+(?:\.\d+)?(?:mT|T).*)$")
        match = pattern.match(self.title)
        if match:
            return match.group(1)
        else:
            return self.title
        
    def extract_number(self):
        """
        Extract the number from the filename.
        
        Args:
            file_path (str): Path to the file.
        
        Returns:
            int: Extracted number from the filename.
        """
        # Split the filename to get the number
        filename = self.file.split('/')[-1]  
        number = filename.split('_')[2]       
        return int(number)



class Mapping_ID02:
    def __init__(self,
                 file: str = None,
                 cif_file: str = None,
                 reflections: np.ndarray = None,
                 qvalues:np.ndarray = None,
                 threshold: float = 0.0001,
                 binning: int = 2):
        """
        Initialize the Mapping class.
        
        Args:
            file (str): Path to the .h5 file.
            cif_file (str): Path to the CIF file for crystallographic information.
            reflections (np.ndarray): Array containing reflection indices.
            qvalues (np.ndarray): array containing q values at which azimuthal profiles shoudl be extracted (saxs data)
            threshold (float): Relative tolerance for q values.
            binning (int): Factor for downsampling the image data.
        """
        self.file=file
        self.file_number=self.extract_number(file)
        self.threshold = threshold
        self.binning = binning
        
        if file is None:
            print("Please specify a data file path")
        self.drx = False  # Flag to check if diffraction data is available
        
        # Load reflections if provided
        if reflections is not None:
            self.reflections = reflections
            self.drx = True
        # Load qvalues if provided
        if qvalues is not None:
            self.qvalues=qvalues
            self.drx=False
        
        # Load CIF file if provided
        if cif_file is not None:
            self.drx = True
            atoms = read(cif_file)
            self.lattice_parameters = atoms.get_cell()
            self.a, self.b, self.c = self.lattice_parameters.lengths()
            self.alpha, self.beta, self.gamma = self.lattice_parameters.angles()
            self.atom_positions = atoms.get_scaled_positions()
            self.atom_elements = atoms.get_chemical_symbols()
            """
            # Print lattice parameters
            print('Crystal structure loaded from CIF:')
            print('Lattice parameters: a=%.4f, b=%.4f, c=%4f, alpha=%d, beta=%d, gamma=%d' %
                  (self.a, self.b, self.c, round(self.alpha), round(self.beta), round(self.gamma)))
            
            # Print atomic positions
            for i, frac_coord in enumerate(self.atom_positions):
                print(f"Atom {self.atom_elements[i]}: {frac_coord}")
            """
        
        # Load data from .h5 file (either WAXS or EIGER2 detector)
        if "_waxs_" in file:
            with h5py.File(file, "r") as f:
                group = list(f.keys())[0]  # Retrieve first group key
                self.title=str(f[group+'/instrument/id02-rayonixhs-waxs/header/Title'][()].decode('utf-8'))
                self.nb_frames = int(f[group + '/instrument/id02-rayonixhs-waxs/acquisition/nb_frames'][()])
                self.acq_time = float(f[group + '/instrument/id02-rayonixhs-waxs/acquisition/exposure_time'][()])
                
                # Retrieve image data
                target = group + '/measurement/data'
                data = np.array(f[target])
                self.data = np.mean(data, axis=0)  # Average over frames
                shape = np.shape(self.data)
                self.num_pixel_x = shape[0]
                self.num_pixel_z = shape[1]
                
                # Retrieve header information
                header = group + '/instrument/id02-rayonixhs-waxs/header'
                self.pixel_size_x = float(f[header + '/PSize_1'][()].decode('utf-8'))
                self.pixel_size_z = float(f[header + '/PSize_2'][()].decode('utf-8'))
                self.wl = float(f[header + '/WaveLength'][()])
                self.x_center = float(f[header + '/Center_1'][()].decode('utf-8'))
                self.z_center = float(f[header + '/Center_2'][()].decode('utf-8'))
                self.D = float(f[header + '/SampleDistance'][()].decode('utf-8'))

                # Retrieve binning info
                header = '/entry_0000/instrument/id02-rayonixhs-waxs/image_operation/binning'
                self.bin_x = f[header + '/x'][()]
                self.bin_y = f[header + '/y'][()]
        
        elif "_eiger2_" in file:
            with h5py.File(file, "r") as f:
                group = list(f.keys())[0]  # Retrieve first group key
                self.nb_frames = int(f[group + '/instrument/id02-eiger2-saxs/acquisition/nb_frames'][()])
                self.acq_time = float(f[group + '/instrument/id02-eiger2-saxs/acquisition/exposure_time'][()])
                
                # Retrieve image data
                target = group + '/measurement/data'
                self.data = np.array(f[target])
                shape = np.shape(self.data)
                self.num_pixel_x = shape[0]
                self.num_pixel_z = shape[1]
                
                # Retrieve header information
                header = '/entry_0000/instrument/id02-eiger2-saxs/header'
                self.pixel_size_x = float(f[header + '/PSize_1'][()].decode('utf-8'))
                self.pixel_size_z = float(f[header + '/PSize_2'][()].decode('utf-8'))
                self.wl = float(f[header + '/WaveLength'][()])
                self.x_center = float(f[header + '/Center_1'][()].decode('utf-8'))
                self.z_center = float(f[header + '/Center_2'][()].decode('utf-8'))
                self.D = f[header + '/SampleDistance'][()]

                # Retrieve binning info
                header = '/entry_0000/instrument/id02-eiger2-saxs/image_operation/binning'
                self.bin_x = f[header + '/x'][()]
                self.bin_y = f[header + '/y'][()]
        
        # Apply binning if needed
        if self.binning != 1:
            self.x_center /= self.binning
            self.z_center /= self.binning
            self.pixel_size_x *= self.binning
            self.pixel_size_z *= self.binning
            self.num_pixel_x //= self.binning
            self.num_pixel_z //= self.binning
            
            # Downsample the image
            self.data = self.downsample_image()

        # Compute Q_parallel and Q_perpendicular
        self.q_parr = np.zeros(self.data.shape, dtype='float')
        self.q_perp = np.zeros(self.data.shape, dtype='float')
        self.norm_Q = np.zeros(self.data.shape, dtype='float')
        self.thetaB = np.zeros(self.data.shape, dtype='float')
        self.phi=np.zeros(self.data.shape, dtype='float')
        
        for i in range(self.num_pixel_z):
            delta_i = (i - self.x_center) * self.pixel_size_x
            for j in range(self.num_pixel_x):
                delta_j = (j - self.z_center) * self.pixel_size_z
                denom = (self.D ** 2 + delta_i ** 2 + delta_j ** 2) ** (1/2)
                a = 2 * np.pi / self.wl
                self.q_parr[j, i] = a * delta_i / denom  # qx
                self.q_perp[j, i] = (a / denom) * (delta_j ** 2 + (self.D - denom) ** 2) ** (1/2)
                self.norm_Q[j, i] = np.sqrt(self.q_parr[j, i] ** 2 + self.q_perp[j, i] ** 2)
                self.thetaB[j, i] = np.arcsin(self.q_parr[j, i] / self.norm_Q[j,i]) * 180 / np.pi  # in degrees
                # build chi_array
                self.phi[j,i]=np.arctan(((i-self.z_center)/(j-self.x_center)))*(180/np.pi)
                
            
        # Apply median filter for noise reduction
        self.data = median_filter(self.data, size=self.binning)

        self.samplename=self.extract_sample_name()
        self.B=self.extract_magnetic_field() 

    def extract_magnetic_field(self):
        # Match patterns like '100mT', '1T', '1.4T', etc.
        match = re.search(r'(\d+(\.\d+)?)(mT|T)', self.title)
        
        if match:
            value, _, unit = match.groups()
            value = float(value)
            
            # Convert T to mT
            if unit == 'T':
                value *= 1000

            return int(value)
        else:
            # If no magnetic field is found, return '0mT'
            return 0
    

    def extract_sample_name(self):
        """Extracts the sample name from a string before the magnetic field value."""
        pattern = re.compile(r"^(.*?)(?:_\d+(?:\.\d+)?(?:mT|T).*)$")
        match = pattern.match(self.title)
        if match:
            return match.group(1)
        else:
            return self.title
        
    def extract_number(self,file_path):
        """
        Extract the number from the filename.
        
        Args:
            file_path (str): Path to the file.
        
        Returns:
            int: Extracted number from the filename.
        """
        # Split the filename to get the number
        filename = file_path.split('/')[-1]  
        number = filename.split('_')[2]       
        return int(number)


    def downsample_image(self):
        """Downsample the image using block averaging."""
        return block_reduce(self.data, block_size=self.binning,func=np.mean)
    
    def plotcomponents(self):   
        """Plot the 2D data in pixel space and q-space."""
        # Plot the original data
        fig,ax=plt.subplots(2,1)
        
        # create grid
        x_edges=np.linspace(0,self.num_pixel_x,self.num_pixel_x+1)
        z_edges=np.linspace(0,self.num_pixel_z,self.num_pixel_z+1)
        z,x=np.meshgrid(z_edges,x_edges)

        mesh0=ax[0].pcolormesh(z,np.abs(x-self.num_pixel_x), 1e-10*self.q_parr,shading='flat',cmap='jet')
        cbar = fig.colorbar(mesh0, ax=ax[0])
        cbar.set_label(r'$q_{\parallel} (\AA^{-1})$')
        ax[0].set_xlabel('Pixels')
        ax[0].set_ylabel('Pixels')
        ax[0].set_title('$q_{\parallel}$')

        
        mesh1=ax[1].pcolormesh(z,np.abs(x-self.num_pixel_x), 1e-10*self.q_perp,shading='flat',cmap='jet')
        cbar = fig.colorbar(mesh1, ax=ax[1])
        cbar.set_label(r'$q_{\perp} (\AA^{-1})$')
        ax[1].set_xlabel('Pixels')
        ax[1].set_ylabel('Pixels')
        ax[1].set_title('$q_{\perp}$')
        plt.tight_layout()
        plt.show()


        fig,ax=plt.subplots(2,1)
        mesh2=ax[0].pcolormesh(z,np.abs(x-self.num_pixel_x), self.thetaB,shading='flat',cmap='jet')
        cbar = fig.colorbar(mesh2, ax=ax[0])
        cbar.set_label('$\\theta_{B}$ (°)')
        ax[0].set_xlabel('Pixels')
        ax[0].set_ylabel('Pixels')
        ax[0].set_title('$\\theta_{B}$ (°)')

        mesh3=ax[1].pcolormesh(z,np.abs(x-self.num_pixel_x), 1e-10*self.norm_Q,shading='flat',cmap='jet')
        cbar = fig.colorbar(mesh3, ax=ax[1])
        cbar.set_label(r'$||\vec{Q}|| (\AA^{-1})$')
        ax[1].set_xlabel('Pixels')
        ax[1].set_ylabel('Pixels')
        ax[1].set_title(r'$||\vec{Q}|| (\AA^{-1})$')
        plt.tight_layout()
        plt.show()
        pass
    
    
    def plot2D(self):   
        """Plot the 2D data in pixel space and q-space."""
        # Plot the original data
        fig,ax=plt.subplots(2,1)
        
        # Plot original data
        x_edges=np.linspace(0,self.num_pixel_x,self.num_pixel_x+1)
        z_edges=np.linspace(0,self.num_pixel_z,self.num_pixel_z+1)
        z,x=np.meshgrid(z_edges,x_edges)

        mesh0=ax[0].pcolormesh(z,np.abs(x-self.num_pixel_x), self.data,shading='flat',cmap='jet',vmin=0,vmax=100)
        #mesh0 = ax[0].pcolormesh(z,x,self.data, shading='flat', cmap='jet',vmin=0,vmax=100)
        cbar = fig.colorbar(mesh0, ax=ax[0])
        cbar.set_label('Intensity')
        ax[0].set_xlabel('Pixels')
        ax[0].set_ylabel('Pixels')
        ##### CHANGE HERE
        # Convert q_parr and q_perp to 1e-10 scale
        q_parr_scaled = 1e-10 * (self.q_parr)
        q_perp_scaled = 1e-10 * (self.q_perp)

        # Compute edges by averaging neighboring points for non-monotonic arrays
        q_parr_edges = np.zeros((q_parr_scaled.shape[0] + 1, q_parr_scaled.shape[1] + 1))
        q_perp_edges = np.zeros((q_perp_scaled.shape[0] + 1, q_perp_scaled.shape[1] + 1))

        q_parr_edges[:-1, :-1] = q_parr_scaled
        q_parr_edges[:-1, -1] = q_parr_scaled[:, -1]
        q_parr_edges[-1, :-1] = q_parr_scaled[-1, :]
        q_parr_edges[-1, -1] = q_parr_scaled[-1, -1]

        q_perp_edges[:-1, :-1] = q_perp_scaled
        q_perp_edges[:-1, -1] = q_perp_scaled[:, -1]
        q_perp_edges[-1, :-1] = q_perp_scaled[-1, :]
        q_perp_edges[-1, -1] = q_perp_scaled[-1, -1]

        # Use the edges in pcolormesh
        mesh1 = ax[1].pcolormesh(q_parr_edges, q_perp_edges, self.data, shading='flat', cmap='jet', vmin=0, vmax=100)
        ### Old code
        #mesh1=ax[1].pcolormesh(1e-10*np.abs(self.q_parr),1e-10*np.abs(self.q_perp),self.data,shading='nearest',cmap='jet',vmin=0,vmax=100)
        cbar = fig.colorbar(mesh1, ax=ax[1])
        cbar.set_label('Intensity')
        # LABELS ARE OK, do not change
        ax[1].set_xlabel(r'$q_{\parallel} (\AA^{-1})$')
        ax[1].set_ylabel(r'$q_{\perp}(\AA^{-1})$')
        ax[1].set_title('Data plotted against $q_{\parallel}$ and $q_{\perp}$')
        
        plt.tight_layout()
        plt.show()
        pass

    # Functions below have been introduced to extract azimuthal profiles from diffraction image, based on peak indexing

    def d_hkl(self,reflection):
        if self.drx:
            """Compute interplanar spacing d_hkl for given Miller indices."""
            h=reflection[0];k=reflection[1];l=reflection[2]
            # Convert angles from degrees to radians
            if self.alpha!=90 or self.beta!=90:
                raise Exception("Triclinic systems not implemented")
            
            else:
                alpha = np.radians(self.alpha)
                beta = np.radians(self.beta)
                gamma = np.radians(self.gamma)
                
                # Compute the denominator of the general formula for d_hkl
                    
                term1 = h**2 / ((self.a**2)*(np.sin(gamma))**2) + k**2 / ((self.b**2)*(np.sin(gamma))**2) + l**2 / (self.c**2)-2 * (h * k * np.cos(gamma) / (self.a * self.b*(np.sin(gamma))**2))
                

                # Calculate d_hkl
                d_hkl = np.sqrt(1 / (term1))
            return d_hkl
        else:
            print('please specify cif file and reflections')
    
    def theta_hkl(self,reflection):
        """Compute Bragg angle"""
        if self .drx:
            return np.arcsin(self.wl/(2*self.d_hkl(reflection)))
        else:
            print('please specify cif file and reflections')
    
    def q_hkl(self,reflection):
        """
        Computes the q (norm of Q vector) value for a given reflexion
        """
        if self.drx:
            return 4*np.pi*np.sin(self.theta_hkl(reflection))/self.wl
        else:
            print('please specify cif file and reflections')
    
    def pixelindexes_constantq(self,reflection=None,qvalue=None):
        """Find detector pixels corresponding to a given Q value."""
        if self.drx:
            q=self.q_hkl(reflection)
        else:
            q=qvalue
        constantq_pixels_indexes=np.argwhere((np.abs(1e-10*self.norm_Q[:, :] - q)/q) <= self.threshold)
        return constantq_pixels_indexes
        
        
    
    def compute_datavstheta_B(self,reflection=None,qvalue=None):
        if self.drx:
            poi=self.pixelindexes_constantq(reflection)
        else:
            poi=self.pixelindexes_constantq(qvalue)
        theta_b=[]
        data=[]
        for pixel in poi:
            i=pixel[0];j=pixel[1]
            theta_b.append(self.thetaB[i,j])
            #print('thetab',theta_b)
            data.append(self.data[i,j])
        results=list(zip(theta_b,data))
        results=sorted(results)
        theta_b,data=zip(*results)
        theta_b=list(theta_b)
        data=list(data)
        return [theta_b,data]
        

    def compute_datavsphi(self,reflection=None,qvalue=None):
        if self.drx:
            poi=self.pixelindexes_constantq(reflection)
        else:
            poi=self.pixelindexes_constantq(qvalue)   
        phi=[]
        data=[]
        for pixel in poi:
            i=pixel[0];j=pixel[1]
            phi.append(self.phi[i,j])
            #print('thetab',theta_b)
            data.append(self.data[i,j])
        return [phi,data]
        

 
    
    def plot_azim_profiles(self,plotphi=False):
        if self.drx:
            nb_plots=len(self.reflections)
            fig,ax=plt.subplots(nb_plots)
            i=0
            for reflection in self.reflections:

                profile=self.compute_datavstheta_B(reflection)
                thetab=profile[0];data=profile[1]
                profile_phi=self.compute_datavsphi(reflection)
                phi=profile_phi[0];data_phi=profile_phi[1]
                ax[i].plot(thetab,data,'.',label=f'Reflection: {reflection} '+'- vs $\\theta_{B}$')
                if plotphi:
                    ax[i].plot(phi,data_phi,'.',label=f'Reflection: {reflection}- vs '+r'$\phi$')
                ax[i].set_xlabel('Angle (°)')
                ax[i].set_ylabel('Intensity')
                ax[i].legend()
                i+=1
            plt.legend()    
            plt.tight_layout()
            plt.show()
        else:
            nb_plots=len(self.qvalues)
            fig,ax=plt.subplots(nb_plots)
            i=0
            for qvalue in self.qvalues:
                profile=self.compute_datavstheta_B(qvalue)
                thetab=profile[0];data=profile[1]
                profile_phi=self.compute_datavsphi(qvalue)
                phi=profile_phi[0];data_phi=profile_phi[1]
                ax[i].plot(thetab,data,'.',label=f'Q value {qvalue:.2f} '+'- vs $\\theta_{B}$')
                if plotphi:
                    ax[i].plot(phi,data_phi,'.',label=f'Q value {qvalue:.2f}- vs '+r'$\phi$')
                ax[i].set_xlabel('Angle (°)')
                ax[i].set_ylabel('Intensity')
                ax[i].legend()
                i+=1
            plt.legend()    
            plt.tight_layout()
            plt.show()

    def gaussian(self,x, y0,I, mean, sigma,slope):
        x = np.asarray(x)
        return y0+slope*x+I*(1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))
    
    def gaussian_nobckgd(self,x,I,mean,sigma):
        return I*(1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))
    
    def pseudo_voigt(self,x,y0,I, x0, gamma,eta,slope):
        pi=np.pi
        ln2=np.log(2)
        a=(2/gamma)*(ln2/pi)**(1/2)
        b=(4*ln2/(gamma**2))
        return y0+I*(eta*(a*np.exp(-b*(x-x0)**2))+(1-eta)*((1/pi)*((gamma/2)/((x-x0)**2+(gamma/2)**2))))
    
    def azim_profile_fit(self,plotflag=False,printResults=False):
        """
        performs fit of azimuthal profile for each reflection of a single image
        results are stored in a dictionnary {reflection:[y0,I,mean,sigma,aire]}
        """
        results={}  
        for reflection in self.reflections:  
            profile=self.compute_datavstheta_B(reflection)
            x=profile[0];y=savgol_filter(profile[1],5,1)
            # find position of max and define fitting range with "width" variable
            index=np.argmax(y) #index of max Intensity
            if self.B==1000:
                width=45
            if self.B==1400:
                width=22
            if self.B==800:
                width=75
            if self.B<799:
                width=90
            a=x[index]-width/2; b=x[index]+width/2 
            
            # define fitting region in x
            test=np.argwhere((a<x)&(x<b))
            amin=test[0,0] #index where x is close to a
            amax=test[-1,0] # index where x is close to b
            #extract arrays corresponding to the fitting region
            x2fit=x[amin:amax+1]
            x2fit = np.asarray(x2fit)
            xmin=np.min(x2fit);xmax=np.max(x2fit)
            y2fit=y[test[:,0]] 
            y2fit=np.asarray(y2fit)
            # give a first estimation of the fitted parameters
            y0_guess=np.mean(y[amin:amin+5]) # flat horizonthal background
            mean_guess=x[index]
            I_guess=np.max(y)
            sigma_guess=5
            slope_guess=0.001
            init_params=[y0_guess,I_guess,mean_guess,sigma_guess,slope_guess]
            
            # define bounds
            lb_G=[0,0,xmin,0,-np.inf] # sigma low bounds can be negative in the formula 
            ub_G=[np.inf,np.inf,xmax,np.inf,np.inf]
            bounds_G=(lb_G,ub_G) 
            
            # fit the parameters and extract sigmas

            try:
                params, _ =curve_fit(self.gaussian,x2fit,y2fit,p0=init_params,bounds=bounds_G,method='trf')
                y0=params[0]
                I=params[1]
                mean=params[2]    
                sigma=params[3]
                slope=params[4]
                # Calculate R² (coefficient of determination)
                residuals = y2fit - self.gaussian(x2fit, *params)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y2fit - np.mean(y2fit))**2)
                r_squared = 1 - (ss_res / ss_tot)

                # calculate surface areas with refined values
                aire, erreur = quad(self.gaussian_nobckgd, a, b, args=(I, mean, sigma))
                fitflag=True
            except:
                y0=np.nan
                I=np.nan
                mean=np.nan
                sigma=np.nan
                aire=np.nan
                r_squared=np.nan
                fitflag=False
            results[str(reflection)]=[y0,I,mean,sigma,aire,r_squared] 
               
            if plotflag:
                #save plots in an array
                path=os.path.dirname(self.file)
                outputdir=path+'/azim_profile_fittings/'
                os.makedirs(outputdir,exist_ok=True)
                # create figure
                fig=plt.figure()
                ax=fig.add_subplot(111)
                ax.plot(x2fit,y2fit,'.-k',label='Experimental data')
                if fitflag:
                    xfine=np.linspace(np.min(x2fit),np.max(x2fit),200)
                    yfit=self.gaussian(xfine,y0,I,mean,sigma,slope)
                    ax.plot(xfine,yfit,'--b',label=f'Gaussian fit, R²={r_squared}')
                fname=os.path.basename(self.file).split('/')[-1].split('.')[0]
                title=f'{fname}-Reflection={reflection}-B={self.B}mT_fit'
                ax.set_title(title)
                ax.legend()
                #plt.show() 
                figname=f'{outputdir}/{title}.png'   
                print(f'plot saved: {figname}')             
                plt.savefig(figname)
        #print(results) 
        if printResults:
            for reflection in self.reflections:
                print(f'Reflection{reflection}: x0={results[str(reflection)][2]:.1f}, sigma={results[str(reflection)][3]:.1f},area={results[str(reflection)][4]:.1f},R²={results[str(reflection)][5]:.2f}')  
        return results
    
    
class BatchAzimProfileExtraction_ID02():
    def __init__(self, path, cif, reflections,file_filter='*_waxs*_raw.h5',threshold=0.0001,binning=2,plotflag=False):
        """
        path: str path to the directory containing h5 files
        cif: cif file describing the sample crystalline structure (optional)
        reflections: list of reflections for which azimuth profiles are extracted
        file_filter:str wildcard file filter (default=*_waxs*_raw.h5)
        threshold (float): Relative tolerance for q values.
        binning (int): Factor for downsampling the image data. 
        plotflag:bool Flag to plot azimuthal profile fits (optional)
        """
        self.path=path
        self.cif=cif
        self.reflections=reflections
        self.h5_filelist=glob.glob(os.path.join(path,file_filter))
        self.h5_filelist=sorted(self.h5_filelist,key=self.extract_number)
        self.threshold=threshold
        self.binning=binning
        self.plotflag=plotflag
        

    def extract_number(self,file_path):
        """
        Extract the number from the filename.
        
        Args:
            file_path (str): Path to the file.
        
        Returns:
            int: Extracted number from the filename.
        """
        # Split the filename to get the number
        filename = file_path.split('/')[-1]  
        number = filename.split('_')[2]       
        return int(number)
    
    def extract_titles(self):
        titlelist=self.path+'/list_titles.txt'
        line2write=''
        for file in self.h5_filelist:
            with h5py.File(file, "r") as f:
                group = list(f.keys())[0]  # Retrieve first group key
                title=str(f[group+'/instrument/id02-rayonixhs-waxs/header/Title'][()].decode('utf-8'))
            
            line2write+=f'{title}\n'
        with open(titlelist,'w') as f:
            f.write(line2write)
        print(line2write)

    def plot_save_azim_profiles_vs_B(self):
        #let's store azim profiles in a dictionnary
        azimprofiles = {}
        B_array = []
        B=-1
        outputdir=self.path+'/Azimuthal_profiles'
        os.makedirs(outputdir,exist_ok=True)
        # Loop through files and process data
        for file in self.h5_filelist:
            map = Mapping_ID02(file, self.cif, self.reflections, threshold=self.threshold, binning=self.binning)
            samplename=map.samplename
            filenumber=map.file_number
            B_array.append(map.B) # store B values
            # Process each reflection
            for reflection in self.reflections:
                reflection_key = tuple(reflection)

                # Ensure the reflection key exists
                if reflection_key not in azimprofiles:
                    azimprofiles[reflection_key] = {}

                # Store the computed azimuthal profile
                azimprofiles[reflection_key][map.B] = map.compute_datavstheta_B(reflection)
                azimprofile=azimprofiles[reflection_key][map.B]
                filename=outputdir+f'/file_{filenumber:05d}_{samplename}_{map.B}mT_{reflection}_azim_profile.csv'
                np.savetxt(filename,np.column_stack([azimprofile[0], azimprofile[1]]),delimiter=',')
                #B=map.B # allows to keep only ascending B

        # make plots
        # Set up subplots — one per reflection
        fig, ax = plt.subplots(len(self.reflections), figsize=(8, 4 * len(self.reflections)))

        # Ensure ax is always iterable (handles 1 reflection case too)
        if len(self.reflections) == 1:
            ax = [ax]

        # Plot each reflection's data
        for i, reflection in enumerate(self.reflections):
            hkl = tuple(reflection)
            Btemp=-1
            for B in B_array:
                azimprofile = azimprofiles[hkl][B]
                
                if B>=Btemp:
                    # plot with '--' if B is ascending
                    ax[i].plot(azimprofile[0], azimprofile[1],'--',label=f'{B} mT')
                else:
                    ax[i].plot(azimprofile[0], azimprofile[1],'-',label=f'{B} mT')
                Btemp=B
            ax[i].set_xlabel('$\\theta_{B}$')
            ax[i].set_ylabel('Intensity')
            ax[i].set_title(samplename+ str(hkl))
        # Create a single legend outside the subplots
        handles, labels = ax[i].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5), title='Magnetic Field (mT)')

        plt.tight_layout()
        plt.subplots_adjust(right=0.9)  # Shrink plots to make space for the legend
        plt.savefig(outputdir + '/AzimProfiles.png', bbox_inches='tight')
        plt.show()
        

  
    def fit_azimprofiles(self,plot=True,r2_threshold=0.85):
        logfile=self.path+'/BatchAzimProfileExtraction.log'
        line2write=''
        self.batch_results=[]
        for file in self.h5_filelist:
            try:
                filename=os.path.basename(file)
                map=Mapping_ID02(file,self.cif,self.reflections,threshold=self.threshold,binning=self.binning)
                samplename=map.samplename;Bstring=map.B
                results=map.azim_profile_fit(plotflag=self.plotflag)
                
                #results are stored in a dictionnary {reflection:[y0,I,mean,sigma,aire]}
                                
                for reflection in self.reflections:
                    background= results[str(reflection)][0]
                    position= results[str(reflection)][2]
                    sigma =results[str(reflection)][3]
                    area = results[str(reflection)][4]
                    r_squared=results[str(reflection)][5]
                    self.batch_results.append([filename,samplename,Bstring,reflection,background,position,sigma,area, r_squared])
            except:
                #print(f'Failed: {filename}\n')
                line2write+=f'Failed: {filename}\n'
        with open(logfile,'w') as f:
            f.write(line2write)
        
        # Create a DataFrame from the list
        self.df = pd.DataFrame(self.batch_results, columns=['File Name', 'Sample_name', 'B (mT)', 'hkl','background','Gaussian peak position','Gaussian sigma','Gaussian area','R_squared'])
        self.df.to_csv(self.path+'/azim_profiles_refinements.csv', index=False)
        
        # Plot area,sigma as a function of B 
        if plot:
            # Ensure 'hkl' is treated as tuples for reliable comparison
            self.df['hkl'] = self.df['hkl'].apply(lambda x: tuple(ast.literal_eval(str(x))))
            reflections=self.df['hkl'].unique()
            fig,ax=plt.subplots(3,figsize=(8,12))
            markers=['s','h','v','^','.','p','o','d']
            
            for i,hkl in enumerate(reflections):
                # filter by reflection and keep good quality fits 
                subset0 = self.df[self.df['hkl'] == hkl] 
                subset= subset0[self.df['R_squared'] > r2_threshold]
                ax[0].plot(subset['B (mT)'], subset['Gaussian peak position'], marker=markers[i], label=f'Reflection {hkl}')               
                ax[1].plot(subset['B (mT)'], subset['Gaussian sigma'], marker=markers[i], label=f'Reflection {hkl}')
                ax[2].plot(subset['B (mT)'], subset['Gaussian area'], marker=markers[i], label=f'Reflection {hkl}')
            for a in ax:
                a.set_xlabel('B (mT)')
                a.grid=True              
            ax[0].set_ylabel('Position (°)')
            ax[1].set_ylabel('$\\sigma$')
            ax[2].set_ylabel('Peak Area')
            # Create a single legend outside the subplots
            handles, labels = ax[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5), title='Reflection')
            fig.suptitle(samplename)
            plt.tight_layout()
            plt.subplots_adjust(right=0.9)
            plt.savefig(self.path+'/Fitting_results.png')
            plt.show()
        return self.df
