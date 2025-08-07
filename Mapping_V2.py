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
import fabio
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
import pyFAI, pyFAI.detectors

class EdfFile:
    def __init__(self, file):
        """
        file: path to file
        """
        self.file = file
        
        # check if data corresponds to lineEraser case
        if 'vd' in self.file.split('_'):
            self.lineEraser=True
            # Redefine x and y centers from individual images
            file1,file2=self.getindividualfiles_lineEraser(self.file)
            im1=fabio.open(file1)
            header1=im1.header
            im2=fabio.open(file2)
            header2=im2.header
            
            self.x_center=float(header1['Center_1'])
            self.z_center=float(header2['Center_2'])

        else:
            self.lineEraser=False
              
        image = fabio.open(self.file)
        header = image.header
        
        self.data = image.data
        
        shape=self.data.shape
        self.num_pixel_x = shape[0]
        self.num_pixel_z = shape[1]
               
        # Experimental details
        self.wl = float(header['WaveLength'])
        
        if not self.lineEraser:
            self.x_center = float(header['Center_1'])
            self.z_center = float(header['Center_2'])
        self.pixel_size_x = float(header['Pixel_1'])
        self.pixel_size_z = float(header['Pixel_2'])
        self.D = float(header['SampleDistance'])
        self.samplename = header['Comment']
        self.nb_frames=1
        self.file_number=int(file.split('/')[-1].split('.')[0].split('_')[-1].split('-')[0])
        

        self.bin_x=1
        self.bin_y=1
        self.B=self.extract_B_value()
        #print('l.68',self.file_number, self.B,'mT')
        
    
    def getindividualfiles_lineEraser(self,file):
        directory=os.path.dirname(file)
        filename=file.split('/')[-1].split('.')[0]
        prefix=filename.split('_')[0]+'_0'
        filenumbers=filename.split('_')[3]
        file1=f'{directory}/{prefix}_{int(filenumbers.split("-")[0]):05d}.edf'
        file2=f'{directory}/{prefix}_{int(filenumbers.split("-")[1]):05d}.edf'
        return file1,file2
    
    def extract_B_value(self):
        match = re.search(r'(\d+)\s*mT', self.file)
        if match:
            return int(match.group(1))  
        return 0


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
                self.title=str(f[group+'/instrument/id02-eiger2-saxs/header/Title'][()].decode('utf-8'))
                self.nb_frames = int(f[group + '/instrument/id02-eiger2-saxs/acquisition/nb_frames'][()])
                self.acq_time = float(f[group + '/instrument/id02-eiger2-saxs/acquisition/exposure_time'][()])
                
                # Retrieve image data
                target = group + '/measurement/data'
                self.data = np.array(f[target])
                shape = np.shape(self.data)
                if len(shape)==2:
                    self.num_pixel_x = shape[0]
                    self.num_pixel_z = shape[1]
                elif len(shape)==3:
                    self.num_pixel_x=shape[1]
                    self.num_pixel_z=shape[2]
                else:
                    print(f"Data in file {self.file} should have 2 or 3 dimensions")
                
                # Retrieve header information
                header = '/entry_0000/instrument/id02-eiger2-saxs/header'
                self.pixel_size_x = float(f[header + '/PSize_1'][()].decode('utf-8'))
                self.pixel_size_z = float(f[header + '/PSize_2'][()].decode('utf-8'))
                self.wl = float(f[header + '/WaveLength'][()])
                self.x_center = float(f[header + '/Center_1'][()].decode('utf-8'))
                self.z_center = float(f[header + '/Center_2'][()].decode('utf-8'))
                self.D = float(f[header + '/SampleDistance'][()])

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



class Mapping:
    def __init__(self,
                 file: str = None,
                 cif_file: str = None,
                 instrument='ID02',
                 reflections: np.ndarray = None,
                 qvalues:np.ndarray = None,
                 threshold: float = 0.0001,
                 binning: int = 2,
                 mask=None):
        """
        Initialize the Mapping class.
        
        Args:
            file (str): Path to the .h5 file.
            cif_file (str): Path to the CIF file for crystallographic information.
            reflections (np.ndarray): Array containing reflection indices.
            qvalues (np.ndarray): array containing q values at which azimuthal profiles shoudl be extracted (saxs data)
            threshold (float): Relative tolerance for q values.
            binning (int): Factor for downsampling the image data.
            mask: path to mask file for pyFAI method (optional)
        """
        self.filepath=file
        self.path=file.split('/')[:-1]
        if instrument=='ID02':
            self.file=h5File_ID02(file)
        elif instrument=='LGC':
            self.file=EdfFile(file)
        self.file_number=self.file.file_number
        self.threshold = threshold
        self.binning = binning
        if mask is not None:
            if mask.split('.')[-1]=='edf':
                maskimage=fabio.open(mask)
                self.mask=mask
                self.maskdata=maskimage.data
                self.maskdata=self.bin_mask()
            else:
                print('Mask Format Error: Mask files should be provided in edf format')
        
        
        if file is None:
            print("Please specify a data file path")
        self.drx = False  # Flag to check if diffraction data is available
        
        # Set Flags to distinguish between SAXS and WAXS data
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
            self.alpha, self.beta_lattice, self.gamma = self.lattice_parameters.angles()
            self.atom_positions = atoms.get_scaled_positions()
            self.atom_elements = atoms.get_chemical_symbols()
            """
            # Print lattice parameters
            print('Crystal structure loaded from CIF:')
            print('Lattice parameters: a=%.4f, b=%.4f, c=%4f, alpha=%d, beta=%d, gamma=%d' %
                  (self.a, self.b, self.c, round(self.alpha), round(self.beta_lattice), round(self.gamma)))
            
            # Print atomic positions
            for i, frac_coord in enumerate(self.atom_positions):
                print(f"Atom {self.atom_elements[i]}: {frac_coord}")
            """
            
        
        # Load data from file (either LGC, WAXS or EIGER2 detector)
        self.nb_frames=self.file.nb_frames
        #self.acq_time=self.file.acq_time
        self.data=self.file.data
        self.num_pixel_x=self.file.num_pixel_x
        self.num_pixel_z=self.file.num_pixel_z
        self.pixel_size_x=self.file.pixel_size_x
        self.pixel_size_z=self.file.pixel_size_z
        self.wl=self.file.wl
        self.x_center=self.file.x_center
        self.z_center=self.file.z_center
        self.D=self.file.D
        self.bin_x=self.file.bin_x
        self.bin_y=self.file.bin_y
        self.samplename=self.file.samplename
        self.B=self.file.B 
       
             
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
        # Average data if needed
        if len(self.data.shape)==3:
            self.data=np.mean(self.data,axis=0)
        # Compute Q_parallel and Q_perpendicular
        self.q_parr = np.zeros(self.data.shape, dtype='float')
        self.q_perp = np.zeros(self.data.shape, dtype='float')
        self.qx = np.zeros(self.data.shape,dtype='float')
        self.qy = np.zeros(self.data.shape,dtype='float')
        self.qz = np.zeros(self.data.shape,dtype='float')
        self.norm_Q = np.zeros(self.data.shape, dtype='float')
        self.thetaB = np.zeros(self.data.shape, dtype='float')
        self.beta = np.zeros(self.data.shape, dtype='float')
        self.phi=np.zeros(self.data.shape, dtype='float')
        
        for i in range(self.num_pixel_z):
            delta_i = (i - self.x_center) * self.pixel_size_x
            for j in range(self.num_pixel_x):
                delta_j = (j - self.z_center) * self.pixel_size_z
                denom = (self.D ** 2 + delta_i ** 2 + delta_j ** 2) ** (1/2)
                a = 2 * np.pi / self.wl
                self.q_parr[j,i] = a * delta_i / denom  # qx
                self.q_perp[j,i] = (a / denom) * (delta_j ** 2 + (self.D - denom) ** 2) ** (1/2)
                self.qy[j,i]=(a/denom)*(self.D-denom)
                self.qz[j,i]=(a/denom)*delta_j
                self.norm_Q[j,i] = np.sqrt(self.q_parr[j,i] ** 2 + self.q_perp[j,i] ** 2)
                self.thetaB[j,i] = np.arcsin(self.q_parr[j,i] / self.norm_Q[j,i]) * 180 / np.pi  # in degrees
                self.beta[j,i] = np.arccos(self.q_parr[j,i] / self.norm_Q[j,i]) * 180 / np.pi  # in degrees
                # build phi_array
                try:
                    self.phi[j,i]=90-(np.arctan(((i-self.z_center)/(j-self.x_center)))*(180/np.pi))
                    phi_error=False
                except:
                    phi_error=True
                    pass
        if phi_error:
            print("phi values could not be computed (divide by zero)")
            print("Azimuthal profiles should be plotted against beta only")
        self.qx = self.q_parr  
              
        # Apply median filter for noise reduction
        self.data = median_filter(self.data, size=self.binning)

        
    def bin_mask(self):
        h, w = self.maskdata.shape
        bin_factor=self.binning
        # Ensure mask shape is divisible by bin_factor
        h_binned = h // bin_factor
        w_binned = w // bin_factor
        mask = self.maskdata[:h_binned * bin_factor, :w_binned * bin_factor]  # crop to divisible size

        # Reshape and apply max to propagate masked pixels
        reshaped = mask.reshape(h_binned, bin_factor, w_binned, bin_factor)
        binned_mask = reshaped.max(axis=(1, 3))  
        return binned_mask
    
    def export2sasview(self):
        directory=os.path.dirname(self.filepath)
        outputfilename=f"{directory}/{self.samplename}_{self.file_number}.dat"
                         
        # write file        
        line2write = "#Data columns Qx - Qy - I(Qx,Qy) \n"
        line2write+="#ASCII data"+"\n"

        qx_array = np.zeros(self.num_pixel_x)
        qy_array = np.zeros(self.num_pixel_z)

        for k in range(self.num_pixel_x):
            for l in range(self.num_pixel_z):
                denominateur=(self.D**2+((k-self.x_center)*self.pixel_size_x)**2+((l-self.z_center)*self.pixel_size_z)**2)**(1/2)
                qx_array[k] = (2*np.pi/self.wl)*((k-self.x_center)*self.pixel_size_x/denominateur)
                qy_array[l] = (2*np.pi/self.wl)*((l-self.z_center)*self.pixel_size_z/denominateur)

        for k in range (self.num_pixel_z-1):
            for l in range(self.num_pixel_x-1):
                
                #replace masked pixels by NaN
                if self.maskdata[l][k]==1:
                    I2write=np.nan
                else:
                    I2write=self.data[l][k]
                #line2write += "%f"%float(qx_array[l])+'\t %f'%float(qy_array[k])+"\t %f"%I2write +"\n"
                line2write += "%f"%float(self.qx[l][k]*1e-10)+'\t %f'%float(self.qz[l][k]*1e-10)+"\t %f"%I2write +"\n"
        with open(outputfilename,'w') as f:
            f.write(line2write)
        
        print(f"File {self.filepath} exported to sasview in {outputfilename}")


    
    
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
        
    
    def extract_number(self,file_path,prefix='WAXS'):
        """
        Extract the number from the filename.
        
        Args:
            file_path (str): Path to the file.
        
        Returns:
            int: Extracted number from the filename.
        """
        # Split the filename to get the number
        
        filename = file_path.split('/')[-1]  
        extension=filename.split('.')[-1]
        #print('l960',extension)
        basename=filename.split('.')[0]
        if extension =='h5':
            number = filename.split('_')[2]
        if extension =='edf':
            #print('l963',basename.split('_')[-1])
            number= basename.split('_')[-1].split('-')[0]   
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
        cbar.set_label('$\\beta$ (°)')
        ax[0].set_xlabel('Pixels')
        ax[0].set_ylabel('Pixels')
        ax[0].set_title('$\\beta$ (°)')

        mesh3=ax[1].pcolormesh(z,np.abs(x-self.num_pixel_x), 1e-10*self.norm_Q,shading='flat',cmap='jet')
        cbar = fig.colorbar(mesh3, ax=ax[1])
        cbar.set_label(r'$||\vec{Q}|| (\AA^{-1})$')
        ax[1].set_xlabel('Pixels')
        ax[1].set_ylabel('Pixels')
        ax[1].set_title(r'$||\vec{Q}|| (\AA^{-1})$')
        plt.tight_layout()
        plt.show()
        pass
    
    def plot_savedata(self,path):   
        """Plot the 2D data in pixel space and q-space."""
        # Création de la figure et des sous-graphiques
        fig, ax = plt.subplots(figsize=(8, 10))  # Ajuste la taille pour une meilleure lisibilité

        # Définition des axes pour l'espace pixel
        x_edges = np.linspace(0, self.num_pixel_x, self.num_pixel_x + 1)
        z_edges = np.linspace(0, self.num_pixel_z, self.num_pixel_z + 1)
        z, x = np.meshgrid(z_edges, x_edges)

        # Tracé des données en espace pixel
        mesh0 = ax[0].pcolormesh(z, np.abs(x - self.num_pixel_x), self.data, shading='flat', cmap='jet', vmin=0, vmax=100)
        cbar = fig.colorbar(mesh0, ax=ax[0])
        cbar.set_label('Intensity')
        ax[0].set_xlabel('Pixels')
        ax[0].set_ylabel('Pixels')
        ax[0].set_aspect('equal')  # Fixe un rapport d'aspect
        figname=f'{path}_Img{self.file_number:05d}_{self.samplename}_{self.B}mT.png'
        plt.tight_layout()
        plt.savefig(figname)
        plt.show()


    def plot2D(self):   
        """Plot the 2D data in pixel space and q-space."""
        # Création de la figure et des sous-graphiques
        fig, ax = plt.subplots(2, 1, figsize=(8, 10))  # Ajuste la taille pour une meilleure lisibilité

        # Définition des axes pour l'espace pixel
        x_edges = np.linspace(0, self.num_pixel_x, self.num_pixel_x + 1)
        z_edges = np.linspace(0, self.num_pixel_z, self.num_pixel_z + 1)
        z, x = np.meshgrid(z_edges, x_edges)

        # Tracé des données en espace pixel
        mesh0 = ax[0].pcolormesh(z, np.abs(x - self.num_pixel_x), self.data, shading='flat', cmap='jet', vmin=0, vmax=100)
        cbar = fig.colorbar(mesh0, ax=ax[0])
        cbar.set_label('Intensity')
        ax[0].set_xlabel('Pixels')
        ax[0].set_ylabel('Pixels')
        ax[0].set_aspect('equal')  # Fixe un rapport d'aspect 1:1

        # Conversion des coordonnées en q-space
        q_parr_scaled = 1e-10 * self.q_parr
        q_perp_scaled = 1e-10 * self.q_perp

        # Création des bords des mailles
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

        # Tracé des données en q-space
        mesh1 = ax[1].pcolormesh(q_parr_edges, q_perp_edges, self.data, shading='flat', cmap='jet', vmin=0, vmax=100)
        cbar = fig.colorbar(mesh1, ax=ax[1])
        cbar.set_label('Intensity')
        ax[1].set_xlabel(r'$q_{\parallel} (\AA^{-1})$')
        ax[1].set_ylabel(r'$q_{\perp}(\AA^{-1})$')
        ax[1].set_title('Data plotted against $q_{\parallel}$ and $q_{\perp}$')
        ax[1].set_aspect('equal')  # Fixe un rapport d'aspect 1:1 en q-space

        # Ajout des cercles à q constant pour chaque reflection
        if self.drx:
            colors=['black','red','blue','green','white']
            theta = np.linspace(0, np.pi, 100)
            n=0
            text_angle=10
            for reflection in self.reflections:
                q=self.q_hkl(reflection)
                q_parr_half=q*np.cos(theta)
                q_perp_half=q*np.sin(theta)
                # Création du cercle à q constant
                ax[1].plot(q_parr_half, q_perp_half, color=colors[n], linestyle='dashed', linewidth=1)
                # ajout d'un label
                new_textangle=(text_angle+n*10)*np.pi/180
                ax[1].text(0.9*q*np.cos(new_textangle), 0.9*q*np.sin(new_textangle), f'{reflection}', 
               color='white', fontsize=10, ha='right', va='top', 
               bbox=dict(facecolor=colors[n], alpha=0.9, edgecolor=colors[n]))
                n+=1
        # Amélioration de l'affichage
        plt.tight_layout()
        plt.show()



    # Functions below have been introduced to extract azimuthal profiles from diffraction image, based on peak indexing

    def d_hkl(self,reflection):
        if self.drx:
            """Compute interplanar spacing d_hkl for given Miller indices."""
            h=reflection[0];k=reflection[1];l=reflection[2]
            # Convert angles from degrees to radians
            if self.alpha!=90 or self.beta_lattice!=90:
                raise Exception("Triclinic systems not implemented")
            
            else:
                alpha = np.radians(self.alpha)
                beta = np.radians(self.beta_lattice)
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
    
    def pixelindexes_constantq(self, reflection=None, qvalue=None):
        """Find detector pixels corresponding to a given Q value."""
        if self.drx:
            if reflection is None:
                raise ValueError("Reflection must be provided when cif file is specified.")
            q = self.q_hkl(reflection)
        else:
            if qvalue is None:
                raise ValueError("qvalue must be provided when no cif file is provided is False.")
            q = qvalue

        constantq_pixels_indexes = np.argwhere((np.abs(1e-10 * self.norm_Q[:, :] - q) / q) <= self.threshold)
        return constantq_pixels_indexes
        
        
    
    def compute_datavstheta_B(self,reflection=None,qvalue=None):
        if self.drx:
            poi=self.pixelindexes_constantq(reflection=reflection)
        else:
            poi=self.pixelindexes_constantq(qvalue=qvalue)
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
        #remove zeros (induced by detector gaps)
        data=[data for x in data if x!=0]
        return [theta_b,data]
    
    def compute_datavsbeta(self, reflection=None, qvalue=None):
        if self.drx:
            poi = self.pixelindexes_constantq(reflection=reflection)
        else:
            poi = self.pixelindexes_constantq(qvalue=qvalue)
        beta = []
        data = []
        for pixel in poi:
            i, j = pixel
            beta.append(self.beta[i, j])
            data.append(self.data[i, j])
        results = list(zip(beta, data))
        results = sorted(results)
        # Remove data (=0) and corresponding beta induced by detector gaps
        results = [(b, d) for b, d in results if d != 0]
        if results:
            beta, data = zip(*results)
            beta = np.array(beta)
            data = np.array(data)
        else:
            beta = np.array([])
            data = np.array([])
        # Save azimuthal profile data
        directory=os.path.dirname(self.filepath)+'/Azimuthal_Profiles/'
        os.makedirs(directory,exist_ok=True)
        outputfile=f"{directory}Azim_Profile_{self.samplename}_{self.file_number}.txt"
        #np.savetxt(outputfile, np.column_stack((beta, data)))

        return beta, data

    
    
    def pyFAI_extract_azimprofiles(self,qvalue):
        # set pyFAI detector instance
        detector = pyFAI.detectors.Detector(pixel1=self.pixel_size_x, pixel2=self.pixel_size_z)
        ai = AzimuthalIntegrator(dist=self.D, detector=detector)
        # extract azimuthal profile at given q value using integrate_radial method from AzimuthalIntegrator instance
        ai.setFit2D(self.D*1000,self.x_center,self.z_center,wavelength=self.wl*1e10)
        
        chi,I=ai.integrate_radial(self.data, 540,mask=self.maskdata, radial_range=(qvalue*(1-self.threshold), qvalue*(1+self.threshold)), radial_unit="q_A^-1",method=("no", "histogram", "cython"))
        return chi,I
        

    def compute_datavsphi(self,reflection=None,qvalue=None):
        if self.drx:
            poi=self.pixelindexes_constantq(reflection=reflection)
        else:
            poi=self.pixelindexes_constantq(qvalue=qvalue)   
        phi=[]
        data=[]
        for pixel in poi:
            i=pixel[0];j=pixel[1]
            phi.append(self.phi[i,j])
            #print('thetab',theta_b)
            data.append(self.data[i,j])
        #remove zeros (induced by detector gaps)
        data=[data for x in data if x!=0]
        return [phi,data]
        

 
    
    def plot_azim_profiles(self,plotphi=False):
        if self.drx:
            nb_plots=len(self.reflections)
            fig,ax=plt.subplots(nb_plots)
            i=0
            for reflection in self.reflections:
                if nb_plots==1:
                    subplt=ax
                else:
                    subplt=ax[i]
                profile=self.compute_datavsbeta(reflection=reflection)
                thetab=profile[0];data=profile[1]
                profile_phi=self.compute_datavsphi(reflection=reflection)
                phi=profile_phi[0];data_phi=profile_phi[1]
                subplt.plot(thetab,data,'.',label=f'Reflection: {reflection} '+'- vs $\\beta$')
                if plotphi:
                    subplt.plot(phi,data_phi,'.',label=f'Reflection: {reflection}- vs '+r'$\phi$')
                subplt.set_xlabel('Angle (°)',fontsize = 14)
                subplt.set_ylabel('Intensity',fontsize = 14)
                subplt.legend(fontsize = 14)
                i+=1
            plt.legend()    
            plt.tight_layout()
            plt.close()
        else:
            nb_plots=len(self.qvalues)
            fig,ax=plt.subplots(nb_plots)
            i=0
            for qvalue in self.qvalues:
                if nb_plots==1:
                    subplt=ax
                else:
                    subplt=ax[i]
                print('Plotting azimuthal profile for q=', qvalue)
                profile=self.compute_datavsbeta(qvalue=qvalue)
                thetab=profile[0];data=profile[1]
                profile_phi=self.compute_datavsphi(qvalue=qvalue)
                phi=profile_phi[0];data_phi=profile_phi[1]
                subplt.plot(thetab,data,'.',label=f'Q value {qvalue:.2f} '+'- vs $\\beta$')
                if plotphi:
                    subplt.plot(phi,data_phi,'.',label=f'Q value {qvalue:.2f}- vs '+r'$\phi$')
                subplt.set_xlabel('Angle (°)',fontsize = 14)
                subplt.set_ylabel('Intensity',fontsize = 14)
                subplt.legend(fontsize = 14)
                i+=1
            plt.title(f"{self.samplename}_{self.B}mT")
            plt.legend(fontsize = 14)    
            plt.tight_layout()
            plt.show()

       
    
        
    def pseudo_voigt(self,x,y0,I, x0, gamma,eta,slope):
        pi=np.pi
        ln2=np.log(2)
        a=(2/gamma)*(ln2/pi)**(1/2)
        b=(4*ln2/(gamma**2))
        return y0+slope*x+I*(eta*(a*np.exp(-b*(x-x0)**2))+(1-eta)*((1/pi)*((gamma/2)/((x-x0)**2+(gamma/2)**2))))

        
    def myfunc(self,x,I, x0, gamma,eta):
        # f(beta)sin(beta)
        return self.pv_nobckgd(x, I, x0, gamma,eta)*np.sin(x)
    
    def P2_nobckgd(self,x,I, x0, gamma,eta):
        return ((1/2)*(3*np.cos(x)*np.cos(x)-1)*self.pv_nobckgd(x, I, x0, gamma,eta)*np.sin(x))
    
    def pv_nobckgd(self,x,I, x0, gamma,eta):
        pi=np.pi
        ln2=np.log(2)
        a=(2/gamma)*(ln2/pi)**(1/2)
        b=(4*ln2/(gamma**2))
        return I*(eta*(a*np.exp(-b*(x-x0)**2))+(1-eta)*((1/pi)*((gamma/2)/((x-x0)**2+(gamma/2)**2))))
     
    def compute_S(self,x,I,x0,gamma,eta):
        # calculate degree of order
        norm_factor , error = quad(self.myfunc,0,np.pi,args=(I,x0*np.pi/180,gamma*np.pi/180,eta))
        S , error = quad(self.P2_nobckgd,0,np.pi,args=(I,x0*np.pi/180,gamma*np.pi/180,eta))
        S/=norm_factor
        return S
    
    def azim_profile_fit(self,plotflag=False,printResults=False,method=None,prefix='WAXS'):
        """
        performs fit of azimuthal profile for each reflection of a single image
        results are stored in a dictionnary {reflection:[y0,I,mean,sigma,aire]}
        """
        results={} 
        if self.drx: 
            for reflection in self.reflections:  
                profile=self.compute_datavsbeta(reflection=reflection)
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
                #y0,I, x0, gamma,eta,slope
                y0_guess=np.mean(y[amin:amin+5]) # flat horizonthal background
                x0_guess=x[index]
                I_guess=np.max(y)
                gamma_guess=5
                eta_guess=1
                slope_guess=0.001
                init_params=[y0_guess,I_guess,x0_guess,gamma_guess,eta_guess,slope_guess]
                
                # define bounds
                lb_G=[0,0,xmin,0,0,-np.inf] # sigma low bounds can be negative in the formula 
                ub_G=[np.inf,np.inf,xmax,np.inf,1,np.inf]
                bounds_G=(lb_G,ub_G) 
                
                # fit the parameters and extract sigmas

                try:
                    params, _ =curve_fit(self.pseudo_voigt,x2fit,y2fit,p0=init_params,bounds=bounds_G,method='trf')
                    y0=params[0]
                    I=params[1]
                    x0=params[2]
                    if x0<0 and x0 <-20:
                        x0_S = x0 + 180 # position must be between 0 and 180 (we only consider this range for S calculation)
                    else:
                        x0_S = x0
                    if x0_S >20:
                        x0_S -=90 # center position around 0 so that 0<S<1
                    gamma=params[3]
                    eta=params[4]
                    slope=params[5]
                    # Calculate R² (coefficient of determination)
                    residuals = y2fit - self.pseudo_voigt(x2fit, *params)
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((y2fit - np.mean(y2fit))**2)
                    r_squared = 1 - (ss_res / ss_tot)

                    # calculate surface areas with refined values
                    aire, erreur = quad(self.pv_nobckgd, a, b, args=(I, x0, gamma,eta))
                    fitflag=True

                    # calculate degree of order
                    """
                    norm_factor , error = quad(self.myfunc,0,np.pi,args=(I,x0*np.pi/180,gamma*np.pi/180,eta))
                    S , error = quad(self.P2_nobckgd,0,np.pi,args=(I,x0*np.pi/180,gamma*np.pi/180,eta))
                    S/=norm_factor
                    """
                    S = self.compute_S(x,I,x0_S,gamma,eta)
                except:
                    y0=np.nan
                    I=np.nan
                    mean=np.nan
                    sigma=np.nan
                    slope=np.nan
                    r_squared=np.nan
                    S=np.nan
                    fitflag=False
                results[str(reflection)]=[y0,I,x0,x0_S,gamma,eta,slope,S,r_squared] 
                
                if plotflag:
                    #save plots in an array
                    path=os.path.dirname(self.filepath)
                    outputdir=path+'/Azim_Profile_Fittings/'
                    os.makedirs(outputdir,exist_ok=True)
                    # create figure
                    fig=plt.figure()
                    ax=fig.add_subplot(111)
                    ax.plot(x2fit,y2fit,'.-k',label='Experimental data')
                    if fitflag:
                        xfine=np.linspace(np.min(x2fit),np.max(x2fit),200)
                        yfit=self.pseudo_voigt(xfine,y0,I,x0,gamma,eta,slope)
                        ax.plot(xfine,yfit,'--b',label=f'PV fit, R²={r_squared:.3f}')
                    fname=os.path.basename(self.filepath).split('/')[-1].split('.')[0]
                    title=f'{fname}-Reflection={reflection}-B={self.B}mT_fit'
                    ax.set_xlabel('Angle (°)',fontsize = 14)
                    ax.set_ylabel('Intensity',fontsize = 14)
                    ax.set_title(title)
                    ax.legend(fontsize = 14)
                    #plt.show() 
                    figname=f'{outputdir}/{title}_Img{self.file_number:05d}.png'   
                    print(f'plot saved: {figname}')             
                    plt.savefig(figname)
            #print(results) 
            if printResults:
                for reflection in self.reflections:
                    print(f'Reflection{reflection}: x0={results[str(reflection)][2]:.1f}, gamma={results[str(reflection)][3]:.1f},eta={results[str(reflection)][4]:.1f}, slope={results[str(reflection)][5]:.1f},S={results[str(qvalue)][6]:.2f},R²={results[str(reflection)][7]:.2f}')  
        else:# saxs case
            for qvalue in self.qvalues:  
                if method=='pyFAI':
                    profile=self.pyFAI_extract_azimprofiles(qvalue=qvalue)
                else:
                    profile=self.compute_datavsbeta(qvalue=qvalue)
                x=profile[0];y=savgol_filter(profile[1],5,1)
                # find position of max and define fitting range with "width" variable
                index=np.argmax(y[x<150]) #index of max Intensity
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
                x0_guess=x[index]
                I_guess=np.max(y)
                gamma_guess=5
                eta_guess=1
                slope_guess=0.001
                init_params=[y0_guess,I_guess,x0_guess,gamma_guess,eta_guess,slope_guess]
                
                # define bounds
                lb_G=[0,0,xmin,0,0,-np.inf] # sigma low bounds can be negative in the formula 
                ub_G=[np.inf,np.inf,xmax,np.inf,1,np.inf]
                bounds_G=(lb_G,ub_G) 
                
                # fit the parameters and extract sigmas

                #try:
                params, _ =curve_fit(self.pseudo_voigt,x2fit,y2fit,p0=init_params,bounds=bounds_G,method='trf')
                y0=params[0]
                I=params[1]
                x0=params[2] 
                if x0<0 and x0 <-20:
                    x0_S = x0 + 180 # position must be between 0 and 180 (we only consider this range for S calculation)
                else:
                    x0_S = x0
                if x0_S >20:
                    x0_S -=90 # center position around 0 so that 0<S<1 
                gamma=params[3]
                eta=params[4]
                slope=params[5]
                # Calculate R² (coefficient of determination)
                residuals = y2fit - self.pseudo_voigt(x2fit, *params)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y2fit - np.mean(y2fit))**2)
                r_squared = 1 - (ss_res / ss_tot)

                # calculate surface areas with refined values
                aire, erreur = quad(self.pv_nobckgd, a, b, args=(I, x0, gamma,eta))
                fitflag=True

                # calculate degree of order
                S = self.compute_S(x,I,x0_S,gamma,eta)
                """except:
                    y0=np.nan
                    I=np.nan
                    mean=np.nan
                    sigma=np.nan
                    aire=np.nan
                    S=np.nan
                    r_squared=np.nan
                    fitflag=False"""
                results[str(qvalue)]=[y0,I,x0,x0_S,gamma,eta,slope,S,r_squared] 
                
                if plotflag:
                    #save plots in an array
                    path=os.path.dirname(self.filepath)
                    outputdir=path+'/Azim_Profile_Fittings/'
                    os.makedirs(outputdir,exist_ok=True)
                    # create figure
                    fig=plt.figure()
                    ax=fig.add_subplot(111)
                    ax.plot(x2fit,y2fit,'.-k',label='Experimental data')
                    if fitflag:
                        xfine=np.linspace(np.min(x2fit),np.max(x2fit),200)
                        yfit=self.pseudo_voigt(xfine,y0,I,x0,gamma,eta,slope)
                        ax.plot(xfine,yfit,'--b',label=f'Pseudo Voigt fit, R²={r_squared:.3f}')
                    fname=os.path.basename(self.filepath).split('/')[-1].split('.')[0]
                    title=f'{self.samplename}_B={self.B}mT_q={qvalue}'
                    ax.set_xlabel('Angle (°)',fontsize = 14)
                    ax.set_ylabel('Intensity',fontsize = 14)
                    ax.set_title(title)
                    ax.legend(fontsize = 14)
                    #plt.show() 
                    figname=f'{outputdir}/{title}_Img{self.file_number:05d}_{prefix}.png'   
                   # print(f'plot saved: {figname}')             
                    plt.savefig(figname)
                    plt.close()
            #print(results) 
            if printResults:
                for qvalue in self.qvalues:
                    print(f'Q={qvalue}$(1/\AA)$: x0={results[str(qvalue)][2]:.1f}, sigma={results[str(qvalue)][3]:.1f},slope={results[str(qvalue)][4]:.1f},S={results[str(qvalue)][5]:.2f},R²={results[str(qvalue)][6]:.2f}')  
                
        return results
    
    
class BatchAzimProfileExtraction():
    def __init__(self, path, cif=None, reflections:np.ndarray=None,qvalues=None,instrument='ID02',file_filter='*_waxs*_raw.h5',threshold=0.0001,binning=2,plotflag=False,mask=None):
        """
        path: str path to the directory containing h5 files
        cif: cif file describing the sample crystalline structure (optional)
        reflections: list of reflections for which azimuth profiles are extracted
        qvalues: list of qvalues for which azimuthal profiles are extracted
        file_filter:str wildcard file filter (default=*_waxs*_raw.h5)
        threshold (float): Relative tolerance for q values.
        binning (int): Factor for downsampling the image data. 
        plotflag:bool Flag to plot azimuthal profile fits (optional)
        mask: path to maskfile (optional)
        """
        self.path=path
        self.cif=cif
        self.reflections=reflections
        self.instrument=instrument
        self.qvalues=qvalues
        self.h5_filelist=glob.glob(os.path.join(path,file_filter))
        self.h5_filelist=sorted(self.h5_filelist,key=self.extract_number)
        self.threshold=threshold
        self.binning=binning
        self.plotflag=plotflag
        self.mask=mask

        if self.reflections is not None:
            self.drx=True
        else:
            self.drx=False
        

    def extract_number(self,file_path,prefix='WAXS'):
        """
        Extract the number from the filename.
        
        Args:
            file_path (str): Path to the file.
        
        Returns:
            int: Extracted number from the filename.
        """
        # Split the filename to get the number
        
        filename = file_path.split('/')[-1]  
        extension=filename.split('.')[-1]
        #print('l960',extension)
        basename=filename.split('.')[0]
        if extension =='h5':
            number = filename.split('_')[2]
        if extension =='edf':
            #print('l963',basename.split('_')[-1])
            number= basename.split('_')[-1].split('-')[0]   
        return int(number)
    
    def extract_titles(self):
        titlelist=self.path+'/Sample_list.txt'
        line2write=''
        for file in self.h5_filelist:
            with h5py.File(file, "r") as f:
                group = list(f.keys())[0]  # Retrieve first group key
                title=str(f[group+'/instrument/id02-rayonixhs-waxs/header/Title'][()].decode('utf-8'))
            
            line2write+=f'{title}\n'
        with open(titlelist,'w') as f:
            f.write(line2write)
        print(line2write)

    def plot_save_azim_profiles_vs_B(self,prefix='WAXS',method=None):
        #let's store azim profiles in a dictionnary
        azimprofiles = {}
        B_array = []
        B=-1
        outputdir=self.path+'/Azimuthal_Profiles'
        os.makedirs(outputdir,exist_ok=True)
        # Loop through files and process data
        for file in self.h5_filelist:
            print(f"Extracting azimuthal profile for {file.split('/')[-1]}")
            map = Mapping(file, cif_file=self.cif, reflections=self.reflections,qvalues= self.qvalues, instrument=self.instrument,threshold=self.threshold, binning=self.binning,mask=self.mask)
            samplename=map.samplename
            filenumber=map.file_number

            B_array.append(map.B) # store B values
            # Process each reflection/qvalue
            if self.drx:
                for reflection in self.reflections:
                    reflection_key = tuple(reflection)

                    # Ensure the reflection key exists
                    if reflection_key not in azimprofiles:
                        azimprofiles[reflection_key] = {}

                    # Store the computed azimuthal profile
                    azimprofiles[reflection_key][map.B] = map.compute_datavsbeta(reflection=reflection)
                    azimprofile=azimprofiles[reflection_key][map.B]
                    filename=outputdir+f'/file_{filenumber:05d}_{samplename}_{map.B}mT_{reflection}_azim_profile.csv'
                    np.savetxt(filename,np.column_stack([azimprofile[0], azimprofile[1]]),delimiter=',')
                    #B=map.B # allows to keep only ascending B
            else:
                for qvalue in self.qvalues:
                    qvalue_key=str(qvalue)
                    # Ensure the reflection key exists
                    if qvalue_key not in azimprofiles:
                        azimprofiles[qvalue_key] = {}

                    # Compute and store azimuthal profile
                    if method=='pyFAI':
                        azimprofiles[qvalue_key][map.B] = map.pyFAI_extract_azimprofiles(qvalue=qvalue)
                        filename=outputdir+f'/file_{filenumber:05d}_{samplename}_{map.B}mT_q={qvalue}_azim_profile_pyFAI.csv'
                    else:
                        azimprofiles[qvalue_key][map.B] = map.compute_datavsbeta(qvalue=qvalue)
                        filename=outputdir+f'/file_{filenumber:05d}_{samplename}_{map.B}mT_q={qvalue}_azim_profile.csv'
                    azimprofile=azimprofiles[qvalue_key][map.B]
                    #filename=outputdir+f'/file_{filenumber:05d}_{samplename}_{map.B}mT_q={qvalue}_azim_profile.csv'
                    np.savetxt(filename,np.column_stack([azimprofile[0], azimprofile[1]]),delimiter=',')

        if self.drx:
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
                print('l1170, B_array',B_array)
                for B in B_array:
                    azimprofile = azimprofiles[hkl][B]
                    
                    if B>=Btemp:
                        # plot with '--' if B is ascending
                        ax[i].plot(azimprofile[0], azimprofile[1],'--',label=f'{B} mT')
                    else:
                        ax[i].plot(azimprofile[0], azimprofile[1],'-',label=f'{B} mT')
                    Btemp=B
                ax[i].set_xlabel('$\\theta_{B}$',fontsize = 14)
                ax[i].set_ylabel('Intensity',fontsize = 14)
                ax[i].set_title(samplename+ str(hkl))
            # Create a single legend outside the subplots
            handles, labels = ax[i].get_legend_handles_labels()
            fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5), title='Magnetic Field (mT)',fontsize = 14)

            plt.tight_layout()
            plt.subplots_adjust(right=0.9)  # Shrink plots to make space for the legend
            plt.savefig(outputdir + f'/{prefix}_AzimProfiles.png', bbox_inches='tight')
            plt.show()
        else:
            # make plots
            # Set up subplots — one per reflection
            fig, ax = plt.subplots(len(self.qvalues), figsize=(8, 4 * len(self.qvalues)))

            # Ensure ax is always iterable (handles 1 reflection case too)
            if len(self.qvalues) == 1:
                ax = [ax]

            # Plot each qvalue's data
            for i, qvalue in enumerate(self.qvalues):
                qvalue_key=str(qvalue)
                Btemp=-1
                for B in B_array:
                    azimprofile = azimprofiles[qvalue_key][B]
                    
                    if B>=Btemp:
                        # plot with '--' if B is ascending
                        ax[i].plot(azimprofile[0], azimprofile[1],'--',label=f'{B} mT')
                    else:
                        ax[i].plot(azimprofile[0], azimprofile[1],'-',label=f'{B} mT')
                    Btemp=B
                ax[i].set_xlabel('$\\beta$(°)',fontsize = 14)
                ax[i].set_ylabel('Intensity',fontsize = 14)
                ax[i].set_title(samplename+ '_q='+str(qvalue))
            # Create a single legend outside the subplots
            handles, labels = ax[i].get_legend_handles_labels()
            fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5), title='Magnetic Field (mT)',fontsize = 14)

            plt.tight_layout()
            plt.subplots_adjust(right=0.9)  # Shrink plots to make space for the legend
            plt.savefig(outputdir + f'/{prefix}_AzimProfiles.png', bbox_inches='tight')
            plt.show()
        

  
    def fit_azimprofiles(self,plot=True,r2_threshold=0.85,prefix='WAXS',method=None):
        logfile=self.path+'/BatchAzimProfileExtraction.log'
        line2write=''
        self.batch_results=[]
        for file in self.h5_filelist:
            #try:
            filename=os.path.basename(file)
            map=Mapping(file,cif_file=self.cif,reflections=self.reflections,qvalues=self.qvalues,instrument=self.instrument,threshold=self.threshold,binning=self.binning,mask=self.mask)
            samplename=map.samplename;Bstring=map.B
            if method=='pyFAI':
                results=map.azim_profile_fit(plotflag=plot,method='pyFAI',prefix=prefix)
            else:
                results=map.azim_profile_fit(plotflag=plot,prefix=prefix)
            
            #results are stored in a dictionnary {reflection:[y0,I,x0,x0_S,gamma,eta,slope,S,R²]}
            if self.drx:                
                for reflection in self.reflections:
                    background= results[str(reflection)][0]
                    I=results[str(reflection)][1]
                    position= results[str(reflection)][2]
                    x0_S=results[str(reflection)][3]
                    """
                    if position<0 and position <-20:
                        position+=180 # position must be between 0 and 180 (we only consider this range for S calculation)
                    if position >20:
                        position -=90 # center position around 0 so that 0<S<1
                        """
                    gamma =results[str(reflection)][4]
                    eta = results[str(reflection)][5]
                    slope = results[str(reflection)][6]
                    S= results[str(reflection)][7]
                    r_squared=results[str(reflection)][8]
                    self.batch_results.append([filename,samplename,Bstring,reflection,background,I,position,x0_S,gamma,eta,slope,S, r_squared])
            else:
                for qvalue in self.qvalues:
                    background= results[str(qvalue)][0]
                    I=results[str(qvalue)][1]
                    position= results[str(qvalue)][2]
                    x0_S=results[str(qvalue)][3]
                    """
                    if position<0 and position <-20:
                        position+=180 # position must be between 0 and 180 (we only consider this range for S calculation)
                    if position >20:
                        position -=90 # center position around 0 so that 0<S<1
                    """
                    gamma =results[str(qvalue)][4]
                    eta=results[str(qvalue)][5]
                    slope = results[str(qvalue)][6]
                    S= results[str(qvalue)][7]
                    r_squared=results[str(qvalue)][8]
                    self.batch_results.append([filename,samplename,Bstring,qvalue,background,I,position,x0_S,gamma,eta,slope,S, r_squared])

            #except:
                #print(f'Failed: {filename}\n')
                #line2write+=f'Failed: {filename}\n'
        with open(logfile,'w') as f:
            f.write(line2write)
        
        # Create a DataFrame from the list self.batch_results
        if self.drx:
            self.df = pd.DataFrame(self.batch_results, columns=['File Name', 'Sample_name', 'B (mT)', 'hkl','background','I','Peak position','Position_S','Gamma','eta','Background slope','Order Parameter S','R_squared'])
        else:
            self.df = pd.DataFrame(self.batch_results, columns=['File Name', 'Sample_name', 'B (mT)', 'qvalue','background','I','Peak position','Position_S','Gamma','eta','Background slope','Order Parameter S','R_squared'])
        
        self.df.to_csv(self.path+f'/{prefix}_azim_profiles_refinements.csv', index=False)
        
        # Plot area,sigma as a function of B 
        if plot:
            
            if self.drx:
            # Ensure 'hkl' is treated as tuples for reliable comparison
                self.df['hkl'] = self.df['hkl'].apply(lambda x: tuple(ast.literal_eval(str(x))))
                reflections = self.df['hkl'].unique()

                fig, ax = plt.subplots(3, figsize=(8, 12))
                markers = ['s', 'h', 'v', '^', '.', 'p', 'o', 'd']

                for i, hkl in enumerate(reflections):
                    # Filter by reflection and keep good quality fits 
                    # Combine both conditions within a single indexing statement to avoid reindexing warnings
                    subset = self.df[(self.df['hkl'] == hkl) & (self.df['R_squared'] > r2_threshold)]
                    ax[0].plot(subset['B (mT)'], subset['Position_S'], marker=markers[i], label=f'Reflection {hkl}')               
                    ax[1].plot(subset['B (mT)'], subset['Gamma'], marker=markers[i], label=f'Reflection {hkl}')
                    ax[2].plot(subset['B (mT)'], subset['Order Parameter S'], marker=markers[i], label=f'Reflection {hkl}')
                for a in ax:
                    a.set_xlabel('B (mT)',fontsize = 14)
                    a.grid=True              
                ax[0].set_ylabel('Position (°)',fontsize = 14)
                ax[1].set_ylabel('$\\gamma$',fontsize = 14)
                ax[2].set_ylabel('Nematic order parameter S',fontsize = 14)
                # Create a single legend outside the subplots
                handles, labels = ax[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5), title='Reflection',fontsize = 14)
                fig.suptitle(samplename)
                plt.tight_layout()
                plt.subplots_adjust(right=0.9)
                plt.savefig(self.path+f'/{prefix}_Fitting_results.png')
                #plt.show()
                plt.close()
            else:
                
                # --- Bloc de génération des figures pour self.drx == False ---
                qvalues = self.df['qvalue'].unique()

                fig, ax = plt.subplots(3, figsize=(8, 12))
                markers = ['s', 'h', 'v', '^', '.', 'p', 'o', 'd']

                for i, qvalue in enumerate(qvalues):
                    # Filtrage par qvalue et R²
                    subset = self.df[(self.df['qvalue'] == qvalue) & (self.df['R_squared'] > r2_threshold)]
                    if subset.empty:
                        continue
                    ax[0].plot(subset['B (mT)'], subset['Position_S'], marker=markers[i % len(markers)],
                            label=f'q = {qvalue}$\\AA^{{-1}}$')
                    ax[1].plot(subset['B (mT)'], subset['Gamma'], marker=markers[i % len(markers)],
                            label=f'q = {qvalue}$\\AA^{{-1}}$')
                    ax[2].plot(subset['B (mT)'], subset['Order Parameter S'], marker=markers[i % len(markers)],
                            label=f'q = {qvalue}$\\AA^{{-1}}$')

                # Configuration des axes
                for a in ax:
                    a.set_xlabel('B (mT)', fontsize=14)
                    a.grid(True)

                ax[0].set_ylabel('Position (°)', fontsize=14)
                ax[1].set_ylabel('$\\gamma$', fontsize=14)
                ax[2].set_ylabel('Nematic order parameter S', fontsize=14)

                # Récupération des courbes de tous les sous-graphiques pour la légende
                handles, labels = [], []
                for a in ax:
                    h, l = a.get_legend_handles_labels()
                    handles.extend(h)
                    labels.extend(l)

                if handles:
                    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5),
                            title='q values', fontsize=14)

                fig.suptitle(samplename)
                plt.tight_layout()
                plt.subplots_adjust(right=0.9)
                plt.savefig(self.path + f'/{prefix}_Fitting_results.png')
                plt.close()

        return self.df

       
    def plot_savedata(self,prefix='WAXS',vmin=None,vmax=None):
        print('Extracting and saving scattering data as png files')
            
        for file in self.h5_filelist:
            map=Mapping(file,cif_file=self.cif,reflections=self.reflections,qvalues=self.qvalues,instrument=self.instrument,threshold=self.threshold,binning=self.binning,mask=self.mask)
            samplename=map.samplename;Bstring=map.B;number=map.file_number
            if vmin is None:
                vmin=np.min(np.log10(map.data+1))
            if vmax is None:
                vmax=np.max(np.log10(map.data+1))

            # Apply log scale safely (avoid log(0))
            data_log = np.log10(map.data + 1)

            # Prepare output path
            figpath=self.path+'/png_images'
            os.makedirs(figpath,exist_ok=True)
            
            outputfile = os.path.join(figpath, f'{samplename}_{Bstring}mT_{prefix}_Img{int(number):05d}.png')
            print(f'{outputfile}')
            # Plot and save as PNG
            plt.clf()
            plt.figure()
            
            plt.imshow(data_log, cmap='jet',vmin=vmin,vmax=vmax)  
            plt.colorbar()
            plt.axis('off')
            plt.title(f'{samplename}_{Bstring}mT')
            #plt.legend()
            plt.savefig(outputfile, bbox_inches='tight', pad_inches=0)
            plt.close()
        


