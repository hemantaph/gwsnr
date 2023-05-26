# for delta_f =1/duration, duration = 16s  
# f_min =20Hz
# duration=16.0, sampling_frequency=4096,
# note: setting mtot_min and mtot_max is important.
# mtot_min=219. is accordance to minimum_frequency = 20
# __init__ paramters are important don't to change for a particular analysis
# they are detector and waveform dependent parameters
# at f_min==10Hz: mtot_max=439.6 
import numpy as np
import bilby
from pycbc.detector import Detector
from scipy.stats import norm
from scipy.interpolate import interp1d
import warnings

C = 299792458.
G = 6.67408*1e-11
Mo = 1.989*1e30

'''
------------------------------------------------
    class containing following methods
    1. to calculate fast SNR
    2. interpolation of with cubic spline
    with bilby SNR
    3. Pdet: probability of detection
------------------------------------------------
'''
class quintet():
    ####################################################
    #                                                  #
    #             Class initialization                 #
    #                                                  #
    ####################################################
    def __init__(self, mtot_min=2., mtot_max=439.6, nsamples=100, list_of_detectors=['L1', 'H1', 'V1'], sensitivity='O3', \
                duration=16., sampling_frequency=4096.,\
                waveform_arguments= dict(waveform_approximant = "TaylorF2", \
                                               reference_frequency = 20., minimum_frequency = 20.)):
        
        '''
        Initialized parameters and functions
        snr_half_scaled() : function for finding (f/PSD) integration in the limit [f_min,f_max]
        list_of_detectors :  list of detector initials, e.g. L1 for Livingston
        sensitivity : detectors sensitivity at various runs
        f_min : minimum frequency for the detector
        -----------------
        input parameters
        -----------------
        mtot_min           : minimum value of Mtotal=m1+m2, use in interpolation
        mtot_max           : maximum value of Mtotal=m1+m2, use in interpolation
        nsamples           : number of points you want to use for SNR interpolation (here it is half SNR not complete)
        list_of_detectors  : detector list. It can be single or multiple.
        sensitivity        : sensitivity related to the noise profile of 'O1', 'O2' or 'O3' run
        duration           : duration of the data in time domain. 
        sampling_frequency : sampling frequency of the data. e.g. 4096Hz,2048Hz,1024Hz
        waveform_arguments : contains which waveform model to use for interpolation. Extra paramters like reference_frequency\
                             minimum_frequency are also included. minimum_frequency will also relate to the mtot_max set inside\
                             the code. High mass blackholes tends to merge at lower frequency < f_min, and can have SNR=0
        
        '''
        self.mtot_min = mtot_min
        self.mtot_max = mtot_max
        self.nsamples = nsamples
        self.list_of_detectors = list_of_detectors
        
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.waveform_arguments = waveform_arguments
        self.list_of_detectors = list_of_detectors
        self.sensitivity = sensitivity
        self.obs_run =  {'O1':1126259462.4, 'O2':1187008672.43, 'O3':1246527224.169434 }
        self.f_min = waveform_arguments['minimum_frequency']
        
        # pre-initialized half scaled snr with search sort
        # self.halfSNR values are initialized
        self.__init_halfScaled() # you can also reinitialized this

    ####################################################
    #                                                  #
    #         fast snr with cubic spline               #
    #                                                  #
    ####################################################
    def snr(self, mass_1, mass_2, luminosity_distance=100., iota=0., \
            psi=0., phase=0., geocent_time=1246527224.169434, ra=0., dec=0.):
        '''
        -----------------
        Input parameters (GW parameters)
        -----------------
        mass_1               : Heavier compact object of the binary, unit: Mo (solar mass)
                               (flaot array or just float)
        mass_2               : Lighter compact object of the binary, unit: Mo (solar mass)
                               (flaot array or just float)
        luminosity_distance  : Distance between detector and binary, unit: Mpc 
        iota                 : Inclination angle of binary orbital plane wrt to the line of sight. unit: rad
        psi                  : Polarization angle. unit: rad
        phase                : Phase of GW at the the time of coalesence, unit: rad
        geocent_time         : GPS time of colescence of that GW, unit: sec
        ra                   : Right ascention of source position, unit: rad
        dec                  : Declination of source position, unit: rad
        -----------------
        Return values
        -----------------
        opt_snr              : dictionary containing net optimal snr and optimal snr of individual detectors
                               example of opt_snr_unscaled return values for len(m1)=3
                                {'opt_snr_net': array([156.53268655, 243.00092419, 292.10396943]),
                                 'L1': array([132.08275995, 205.04492349, 246.47822334]),
                                 'H1': array([ 84.00372897, 130.40716432, 156.75845871])}
        
        '''
        mass_1, mass_2 = np.array([mass_1]).reshape(-1), np.array([mass_2]).reshape(-1)
        num = len(mass_1)
        luminosity_distance, theta_jn, psi, phase, geocent_time, ra, dec = \
                                        np.array([luminosity_distance]).reshape(-1)*np.ones(num), \
                                        np.array([iota]).reshape(-1)*np.ones(num), \
                                        np.array([psi]).reshape(-1)*np.ones(num), \
                                        np.array([phase]).reshape(-1)*np.ones(num), \
                                        np.array([geocent_time]).reshape(-1)*np.ones(num), \
                                        np.array([ra]).reshape(-1)*np.ones(num), \
                                        np.array([dec]).reshape(-1)*np.ones(num)
        
        Mc = ( (mass_1*mass_2)**(3/5) )/( (mass_1+mass_2)**(1/5) )
        mtot = mass_1+mass_2
        Dl = luminosity_distance
        
        # dealing with mtot array
        # mtot > mtot_max will be have snr = 0.
        snr_half_scaled = np.zeros(num) # for mtot > mtot_max, set zero value will not change later
        idx2 = np.array(np.where(self.mtot_max>=mtot)).reshape(-1).tolist() # record index with mtot values less than mtot_max
        # getting simple snr_half_scaled values for interpolation
        halfSNR_interpolator = self.halfSNR
        
        A1 = Mc**(5./6.)
        ci_2 = np.cos(iota)**2
        ci_param = ((1+np.cos(iota)**2)/2)**2
        detectors = self.list_of_detectors
        
        opt_snr = {'opt_snr_net': 0}
        
        # loop wrt detectors
        for det in detectors:
            # calculation of snr_half_scaled for particular detector at the required mtot
            snr_half_scaled[idx2] = halfSNR_interpolator[det](mtot[idx2])
            
            Fp, Fc = Detector(det).antenna_pattern(ra, dec, psi, geocent_time)
            Deff1 = Dl/np.sqrt( Fp**2*ci_param + Fc**2*ci_2 )
            
            opt_snr[det] = (A1/Deff1)*snr_half_scaled
            opt_snr['opt_snr_net'] += opt_snr[det]**2

        opt_snr['opt_snr_net'] = np.sqrt(opt_snr['opt_snr_net'])

        return( opt_snr )
        
    ####################################################
    #                                                  #
    #   half_snr vs mtot table for interpolation       #
    #                                                  #
    ####################################################
    def __init_halfScaled(self):
        '''
        Function for finding (f/PSD) integration in the limit [f_min,f_max]
        f_min is already initialized
        f_max is taken as 'last stable orbit frequency' is a function of mtot
        __init_halfScaled(self) will initialize the interpolator (scipy cubic spline) as self.halfSNR
        -----------------
        Input parameters
        -----------------
        None
        -----------------
        Return values
        -----------------
        snrHalf_det  : cubic spline interpolator for halfScaledSNR --> (f/PSD) integration in the limit [f_min,f_max]
                       If there is 3 detectors, it will return 3 types of scipy cubic spline objects
        '''
        mtot_min = self.mtot_min
        mtot_max = self.mtot_max
        nsamples = self.nsamples
        detectors = self.list_of_detectors
        
        try:
            if  mtot_min<1.:
                raise ValueError
        except ValueError:
            print('Error: mass too low')
        
        C = 299792458.
        G = 6.67408*1e-11
        Mo = 1.989*1e30
        f_min = self.f_min
        # mtot_max_propose from f_min
        mtot_max_propose = (C**3)/( G*Mo*f_min*np.pi*6**(3/2) )
        
        
        if mtot_max_propose<mtot_max:
            warnings.warn\
                (f'\n Mtot_max={mtot_max} given here is smaller than Mtot_max set by \
                f_min={f_min}, \n new Mtot_max={mtot_max_propose}. \n If you want higher Mtot_max, set f_min lower \
                (e.g. f_min=10Hz, but not lesser than 10Hz)')
            mtot_max = mtot_max_propose
            self.mtot_max = mtot_max
        
        #mtot_table = np.sort(mtot_min+mtot_max-np.geomspace(mtot_min,  mtot_max, nsamples))
        #mtot_table = np.geomspace(mtot_min,  mtot_max, nsamples)
        mtot_table = np.linspace(mtot_min,  mtot_max, nsamples)
        
        mass_1_ = mtot_table/2
        mass_2_ = mass_1_
        mchirp = ( (mass_1_*mass_2_)**(3/5) )/( (mtot_table)**(1/5) )
        
        # observing run and chosen geocent time in that observing run
        # here i have assumed the sesitivity remains the same within each observing run
        obs_run_ = self.obs_run

        # geocent_time cannot be array here
        # this geocent_time is only to get halfScaledSNR
        geocent_time_ = obs_run_[self.sensitivity]
    
        iota_, ra_, dec_, psi_, phase_ = 0.,0.,0.,0.,0.
        luminosity_distance_ = 100.
        ######## calling bilby_snr ########
        opt_snr_unscaled = self.compute_bilby_snr_(mass_1=mass_1_, mass_2=mass_2_, luminosity_distance=luminosity_distance_, \
                                                theta_jn=iota_, psi=psi_, ra=ra_, dec=dec_)  
        '''
        example of opt_snr_unscaled return values
        {'opt_snr_net': array([156.53268655, 243.00092419, 292.10396943]),
         'L1': array([132.08275995, 205.04492349, 246.47822334]),
         'H1': array([ 84.00372897, 130.40716432, 156.75845871])}
        '''

        A2 = mchirp**(5./6.)
        ######## filling in interpolation table for different detectors ########
        snrHalf_det = {}
        for det in detectors:
            Fp, Fc = Detector(det).antenna_pattern(ra_, dec_, psi_, geocent_time_)
            Deff2 = luminosity_distance_/np.sqrt(Fp**2*((1+np.cos(iota_)**2)/2)**2+Fc**2*np.cos(iota_)**2 )
            snrHalf_det[det]= interp1d( mtot_table, (Deff2/A2)*opt_snr_unscaled[det], kind = 'cubic')
        
        # 2D array size: n_detectors X nsamples np.concatenate((a, b), axis=0)
        snrHalf_det['mtot'] = mtot_table
        #print(snrHalf_det)
        self.halfSNR = snrHalf_det
        
        # return value below is just for testing
        return(snrHalf_det)
    
    #######################################################################
    ##################### bilby snr with frequency limit ##################
    #######################################################################
    def compute_bilby_snr_(self, mass_1, mass_2, luminosity_distance=100., theta_jn=0., \
                            psi=0., phase=0., geocent_time=np.array([]), ra=0., dec=0.):
        '''
        SNR calculated using bilby python package
        Use for interpolation purpose
        -----------------
        Input parameters (GW parameters)
        -----------------
        mass_1               : Heavier compact object of the binary, unit: Mo (solar mass)
                               (flaot array or just float)
        mass_2               : Lighter compact object of the binary, unit: Mo (solar mass)
                               (flaot array or just float)
        luminosity_distance  : Distance between detector and binary, unit: Mpc 
        theta_jn             : Inclination angle of binary orbital plane wrt to the line of sight. unit: rad
        psi                  : Polarization angle. unit: rad
        phase                : Phase of GW at the the time of coalesence, unit: rad
        geocent_time         : GPS time of colescence of that GW, unit: sec
        ra                   : Right ascention of source position, unit: rad
        dec                  : Declination of source position, unit: rad
        -----------------
        Return values
        -----------------
        opt_snr              : dictionary containing net optimal snr and optimal snr of individual detectors
                               example of opt_snr_unscaled return values for len(m1)=3
                                {'opt_snr_net': array([156.53268655, 243.00092419, 292.10396943]),
                                 'L1': array([132.08275995, 205.04492349, 246.47822334]),
                                 'H1': array([ 84.00372897, 130.40716432, 156.75845871])}
        '''
        obs_run_ = self.obs_run
        geocent_time_ = obs_run_[self.sensitivity]
        
        # check whether there is input for geocent_time
        if not np.array(geocent_time).tolist():
            geocent_time = geocent_time_
        duration = self.duration
        
        sampling_frequency = self.sampling_frequency
        waveform_arguments = self.waveform_arguments
        detectors = self.list_of_detectors
        # reshape(-1) is so that either a float value is given or the input is an numpy array
        # np.ones is multipled to make sure everything is of same length
        mass_1, mass_2 = np.array([mass_1]).reshape(-1), np.array([mass_2]).reshape(-1)
        num = len(mass_1)
        luminosity_distance, theta_jn, psi, phase, ra, dec, geocent_time = \
                                        np.array([luminosity_distance]).reshape(-1)*np.ones(num), \
                                        np.array([theta_jn]).reshape(-1)*np.ones(num), \
                                        np.array([psi]).reshape(-1)*np.ones(num), \
                                        np.array([phase]).reshape(-1)*np.ones(num), \
                                        np.array([ra]).reshape(-1)*np.ones(num), \
                                        np.array([dec]).reshape(-1)*np.ones(num), \
                                        np.array([geocent_time]).reshape(-1)*np.ones(num)
        mtot = mass_1+mass_2
        bilby.core.utils.logger.disabled = True
        np.random.seed(88170235)

        # initialize interferometer object
        # it is ideal to initialized it outside the for loop
        waveform_generator = bilby.gw.WaveformGenerator(
                duration=duration,
                sampling_frequency=sampling_frequency,
                frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
                parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
                waveform_arguments=waveform_arguments,)

        det_ = {'L1':0,'H1':1,'V1':2}
        ifos = bilby.gw.detector.InterferometerList(detectors)
        ifos.set_strain_data_from_power_spectral_densities(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=geocent_time_-duration,
        )
        
        snr_det = np.zeros(num) # for mtot > mtot_max, set zero value will not change later
        idx2 = np.array(np.where(self.mtot_max>=mtot)).reshape(-1).tolist() # record index with mtot values less than mtot_max
        opt_snr = {'opt_snr_net': 0}
        f_min = waveform_arguments['minimum_frequency']
        for det in detectors:
            new_keys = {det: snr_det}
            opt_snr.update(new_keys)
            for i in idx2:
                ifos[det_[det]].minimum_frequency = f_min
                f_max = (C**3)/( G*(mtot[i])*Mo*np.pi*6**(3/2) )
                ifos[det_[det]].maximum_frequency = f_max

                injection_parameters = dict(mass_1=mass_1[i],mass_2=mass_2[i],a_1=0.,a_2=0.,tilt_1=0.,tilt_2=0.,phi_12=0.,\
                                            phi_jl=0.,luminosity_distance=luminosity_distance[i],theta_jn=theta_jn[i],psi=psi[i], \
                                            phase=phase[i],geocent_time=geocent_time[i],ra=ra[i],dec=dec[i])
                ifos.inject_signal(
                    waveform_generator=waveform_generator, parameters=injection_parameters, raise_error=False
                );

                opt_snr[det][i] = ifos[det_[det]].meta_data['optimal_SNR']

            opt_snr[det] = np.array(opt_snr[det])
            opt_snr['opt_snr_net'] += opt_snr[det]**2

        opt_snr['opt_snr_net'] = np.sqrt(opt_snr['opt_snr_net'])

        return(opt_snr)
    
    ####################################################
    #                                                  #
    #             Probaility of detection              #
    #                                                  #
    ####################################################
    def pdet(self, param, rho_th=8., rhoNet_th=8.):
        '''
        Probaility of detection of GW for the given sensitivity of the detectors
        -----------------
        Input parameters
        -----------------
        param      : dictionary of GW parameters (both extrinsic and intrinsic) 
                     e.g. param_ = {'m1':mass_1,'m2':mass_2,'Dl':luminosity_distance,'iota':theta_jn,\
                                   'psi':psi,'phase':phase,'ra':ra,'dec':dec,'geocent_time':}
        Each of the parameter can me a single float or numpy array of float
        
        -----------------
        Return values
        -----------------
        dict_pdet  : dictionary of {'pdet_net':pdet_net, 'pdet_L1':pdet_L1, 'pdet_H1':pdet_H1, 'pdet_V1':pdet_V1}                    
        '''
        m1 = param['m1']
        m2 = param['m2']
        Dl = param['Dl']
        i = param['iota']
        psi = param['psi']
        phase = param['phase']
        ra = param['ra']
        dec = param['dec']
        geocent_time = param['geocent_time']
        # rho, has snr of network, L1, H1, V1 by default
        rho = self.snr(mass_1=m1, mass_2=m2, luminosity_distance=Dl, iota=i, psi=psi, \
                       phase=phase, ra=ra, dec=dec, geocent_time=geocent_time)

        pdet_L1 = 1 - norm.cdf(rho_th - rho['L1'])
        pdet_H1 = 1 - norm.cdf(rho_th - rho['H1'])
        pdet_V1 = 1 - norm.cdf(rho_th - rho['V1'])
        pdet_net = 1 - norm.cdf(rhoNet_th - rho['opt_snr_net'])
        
        return( {'pdet_net':pdet_net, 'pdet_L1':pdet_L1, 'pdet_H1':pdet_H1, 'pdet_V1':pdet_V1} )