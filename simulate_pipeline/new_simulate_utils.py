import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.table import Table, vstack
from astropy import units as u
from astropy.cosmology import WMAP9 as cosmo
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import interp1d, interp2d
import george
from george import kernels
from functools import partial

### CUTS ###
first_detection_threshold = 10  #days
num_detections_threshold = 50 #observations
snr_threshold = 10
active_time_threshold = 10 #days
thresholds = [first_detection_threshold, num_detections_threshold, snr_threshold, active_time_threshold]


### PARAMS ###
length_scale = 20 #days
mjd_bins = 180
wave_bins = 32


### MAPS ###
band_to_wave = {
    0: 3670.69,
    1: 4826.85,
    2: 6223.24,
    3: 7545.98,
    4: 8590.90,
    5: 9710.28
}

filter_to_band_number = {
    "b'u '": 0,
    "b'g '": 1,
    "b'r '": 2,
    "b'i '": 3,
    "b'z '": 4,
    "b'Y '": 5
}

SN_TYPE_ID_MAP = {
    6: "Point source microlensing",
    15: "TDE",
    16: "EBE",
    42: "SNII",
    52: "SNIax",
    53: "Mira Variable",
    62: "SNIbc",
    64: "KN",
    65: "M-dwarf",
    67: "SNIa-91bg",
    88: "AGN",
    90: "SNIa",
    92: "RR Lyrae",
    95: "SLSN-1"
}


### HELPER FUNCTIONS ###
def create_err_func(rand, band):
    
    flux = rand['flux']
    flux_err = rand['flux_err']
    mjd = rand['mjd']
    
    err_func = interp2d(mjd, flux, flux_err)
    
    return(err_func)

def get_flux_err(input_flux, wavelength, date, band_err):
    
    single_day_band = band_err[band_err['mjd'] > date-0.5]
    single_day_band = single_day_band[single_day_band['mjd'] < date+0.5]
    
    if len(single_day_band) < 1:
        #print('bad sigma')
        print(len(single_day_band))
        sigma = np.mean(single_band['flux_err'])
    else:
        err_func = interp1d(single_day_band['flux'],single_day_band['flux_err'], fill_value='extrapolate')
        sigma = err_func(input_flux)

    return(sigma)

def set_new_mjd(sndata, timeshift):
    sndata['old_mjd'] = sndata['mjd']
    
    band3 = sndata[sndata['passband'] == 3]
    peak_flux_mjd = np.argmax(band3['flux'])
    
    sndata['mjd'] = sndata['mjd'] - sndata['mjd'][peak_flux_mjd] + timeshift
    
    return(sndata)

def simulate_z(type_meta):
    n_simz_bins = 500
    
    if np.max(type_meta['hostgal_photoz']) == 0:
        return(0)
    
    else:
        bins = np.linspace(0, np.max(type_meta['hostgal_photoz']), n_simz_bins)
        bins = np.append(bins, np.max(bins) + (bins[1]-bins[0]))
        counts, bins = np.histogram(type_meta['hostgal_photoz'], bins=bins, density=True)
    
        mask = np.isnan(counts)
        counts[mask] = 0
    
        z_array = np.linspace(0, np.max(type_meta['hostgal_photoz']), n_simz_bins)
    
        simulated_photoz = np.random.choice(z_array, size=1, p=counts/np.sum(counts))
    
        return(simulated_photoz)
    
def galactic_flux_multiplier(type_meta, model_snid):
    type_meta = type_meta[type_meta['peaku'] != 0]
    
    if np.max(type_meta['hostgal_photoz']) == 0:
        flux_multiplier = []
        for col in ['peaku', 'peakg', 'peakr', 'peaki', 'peakz', 'peaky']:
            bins = np.linspace(0, np.max(type_meta[col]), 50)
            bins = np.append(bins, np.max(bins) + (bins[1]-bins[0]))
            counts, bins = np.histogram(type_meta[col], bins=bins, density=True)

            mask = np.isnan(counts)
            counts[mask] = 0

            flux_array = np.linspace(0, np.max(type_meta[col]), 50)

            new_flux = np.random.choice(flux_array, size=1, p=counts/np.sum(counts))
            
            old_flux = float(type_meta[type_meta['object_id'] == model_snid][col])

            flux_multiplier.append(new_flux/old_flux)

        return(flux_multiplier)
    else:
        return([1,1,1,1,1,1])
    
def fit_gp(sndata):
    bands = [band_to_wave[elem] for elem in sndata['passband']]

    mjdall = sndata['mjd']
    fluxall = sndata['flux']
    flux_errall = sndata['flux_err']
    
    #Want to compute the scale factor that we will use...
    signal_to_noises = np.abs(fluxall) / np.sqrt(flux_errall ** 2 + (1e-2 * np.max(fluxall)) ** 2)
    scale = np.abs(fluxall[signal_to_noises == np.max(signal_to_noises)])
    if len(scale)<1:
        return None
    elif len(scale)>1:
        scale = scale.to_numpy()[0]
    #print(scale)
    kernel = (0.5 * int(scale)) ** 2 * kernels.Matern32Kernel([length_scale ** 2, 6000 ** 2], ndim=2)

    gp = george.GP(kernel)
    guess_parameters = gp.get_parameter_vector()

    x_data = np.vstack([mjdall, bands]).T
    gp.compute(x_data, flux_errall)

    def neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.log_likelihood(fluxall)

    def grad_neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(fluxall)

    bounds = [(0, np.log(1000 ** 2))]
    bounds = [(guess_parameters[0] - 10, guess_parameters[0] + 10)] + bounds + [(None, None)]
    # check if result with/without bounds are the same

    try:
        fit_result = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like, bounds=bounds)
        gp.set_parameter_vector(fit_result.x)
        gaussian_process = partial(gp.predict, fluxall)
    except ValueError:
        return None
    except np.linalg.LinAlgError:
        return None
    except TypeError:
        return None
    
    return gaussian_process

def redshift_model(unredshifted_gp, model_data, z, z_mod):
    
    wavelength = np.linspace(3000*(1+(0.9*z))/(1+z_mod), 10000*(1+(1.1*z))/(1+z_mod), 200) #start with 3000(1+z_min)/(1+z_mod) go to 10,000(1+z_max)/(1+z_mod), z_max = 1.1z aand z_min =0.9z
    dates = np.linspace(np.min(model_data['mjd']), np.max(model_data['mjd']), 200)
        
    time_wavelength_grid = np.transpose([np.tile(dates, len(wavelength)), np.repeat(wavelength, len(dates))])
    
    unredshifted_model, unredshifted_err = unredshifted_gp(time_wavelength_grid, return_var = True)
    unredshifted_model = np.array(unredshifted_model).reshape(len(wavelength), len(dates))
    
    #deredshift = unredshifted_model/(1+z_mod) #CHECK THIS
    if z_mod != 0:
        old_lum_distance_Mpc = cosmo.luminosity_distance(z_mod)
        new_lum_distance_Mpc = cosmo.luminosity_distance(z)
        #lum_distance_pc = lum_distance_Mpc.to(u.pc)
        #print(f"Redshift lum distance = {new_lum_distance_Mpc}")
        #print(f"OG Model lum distance = {old_lum_distance_Mpc}")
        print(f"flux factor: {(old_lum_distance_Mpc/new_lum_distance_Mpc)**2}")
        redshift = unredshifted_model * (old_lum_distance_Mpc/new_lum_distance_Mpc)**2
        
    #redshift = deredshift/(1+z) #CHECK THIS
    #if z != 0:
        #lum_distance_Mpc = cosmo.luminosity_distance(z)
        #lum_distance_pc = lum_distance_Mpc.to(u.pc)
        #redshift = redshift / (lum_distance_pc/(10 * u.pc))**2

    dates = dates*(1+z)/(1+z_mod)
    wavelength = wavelength*(1+z)/(1+z_mod)
    
    u_index = np.absolute(wavelength-3670.69).argmin()
    g_index = np.absolute(wavelength-4826.85).argmin()
    r_index = np.absolute(wavelength-6223.24).argmin()
    i_index = np.absolute(wavelength-7545.98).argmin()
    z_index = np.absolute(wavelength-8590.90).argmin()
    y_index = np.absolute(wavelength-9710.28).argmin()
    
    redshifted_band_data = pd.DataFrame(columns = ['mjd','passband','flux', 'flux_err'])
    
    udata = pd.DataFrame(data={'mjd': dates, 'passband': np.repeat([0],len(redshift[u_index])), 
                            'flux': redshift[u_index], 'flux_err': np.repeat([0],len(redshift[u_index]))},
                         columns=['mjd','passband','flux','flux_err'])
    gdata = pd.DataFrame(data={'mjd': dates, 'passband': np.repeat([1],len(redshift[g_index])), 
                            'flux': redshift[g_index], 'flux_err': np.repeat([0],len(redshift[g_index]))})
    rdata = pd.DataFrame(data={'mjd': dates, 'passband': np.repeat([2],len(redshift[r_index])), 
                            'flux': redshift[r_index], 'flux_err': np.repeat([0],len(redshift[r_index]))})
    idata = pd.DataFrame(data={'mjd': dates, 'passband': np.repeat([3],len(redshift[i_index])), 
                            'flux': redshift[i_index], 'flux_err': np.repeat([0],len(redshift[i_index]))})
    zdata = pd.DataFrame(data={'mjd': dates, 'passband': np.repeat([4],len(redshift[z_index])), 
                            'flux': redshift[z_index], 'flux_err': np.repeat([0],len(redshift[z_index]))})
    ydata = pd.DataFrame(data={'mjd': dates, 'passband': np.repeat([5],len(redshift[y_index])), 
                            'flux': redshift[y_index], 'flux_err': np.repeat([0],len(redshift[y_index]))})
    
    true_peakmjd = idata['mjd'][np.argmax(idata['flux'])]
    redshifted_band_data = pd.concat([udata, gdata, rdata, idata, zdata, ydata]).reset_index(drop=True)
    
    return(redshifted_band_data, true_peakmjd)

def simulate_band(gaussian_process, sn_meta, sn_data, band, flux_multiplier, band_err):
    wavelength = [band_to_wave[band]]
    
    dates = sn_data[sn_data['passband'] == band]['mjd'].reset_index(drop=True)
        
    time_wavelength_grid = np.transpose([np.tile(dates, len(wavelength)), np.repeat(wavelength, len(dates))])
    
    simulated, simulated_err = gaussian_process(time_wavelength_grid, return_var = True)
    simulated = np.array(simulated).reshape(len(wavelength), len(dates))
    
    #muliply by flux_muliplier to create some variation in brightness of galactic (z=0) events 
    #muliplier for extragalactic objects is set to 1
    #redshift = flux_multiplier*redshift

    noise = np.zeros((len(wavelength), len(dates)))
    sigma_grid = np.zeros((len(wavelength), len(dates)))
    for n in range(len(wavelength)):
        for m in range(len(dates)):
            sigma = get_flux_err(simulated[n,m], wavelength[n], dates[m]+np.min(sn_data['old_mjd']), band_err)
            sigma_grid[n,m] = sigma
            noise[n,m] = np.random.normal(0, np.abs(sigma))

    simulated += noise
    
    return(simulated[0], sigma_grid[0], dates)

def simulate_event(lcdata, metadata, model_snid, target_snid, band0_err, band1_err, band2_err, band3_err, band4_err, band5_err):
    target_snid = int(target_snid)
    model_snid = int(model_snid)
    
    sn_meta = metadata[metadata['object_id'] == target_snid]
    sn_data = lcdata[lcdata['object_id'] == target_snid].reset_index(drop=True)
    
    #get lc, event type, redshift data on model object
    model_data = lcdata[lcdata['object_id'] == model_snid].reset_index(drop=True)
    model_meta = metadata[metadata['object_id'] == model_snid].reset_index(drop=True)
    model_type = int(metadata[metadata['object_id'] == model_snid]['true_target'])
    type_meta = metadata[metadata['true_target'] == int(model_type)]
    z_mod = float(metadata[metadata['object_id'] == model_snid]['hostgal_specz'])
        
    #shift model in time, so peak in band 3 is centered at t = 0 + offset
    time_shift = np.random.uniform(np.min(lcdata['mjd'])-np.max(lcdata['mjd']), np.max(lcdata['mjd'])-np.min(lcdata['mjd']))
    
    while time_shift + float(model_meta['true_peakmjd']) > np.max(sn_data['mjd']) or time_shift + float(model_meta['true_peakmjd']) < np.min(sn_data['mjd']):
        time_shift = np.random.uniform(np.min(lcdata['mjd'])-np.max(lcdata['mjd']), np.max(lcdata['mjd'])-np.min(lcdata['mjd']))
    print(f"time_shift: {time_shift}")
    print(time_shift + float(model_meta['true_peakmjd']))
    print(f"target max mjd {np.max(sn_data['mjd'])}")
    print(f"target min mjd {np.min(sn_data['mjd'])}")
    
    model_data = set_new_mjd(model_data, time_shift)
    
    #choose random z from range of z's from other events of the same type
    #need to change this to be more representitive of PLAsTiCC dataset
    z = simulate_z(type_meta)
    delta_z = np.abs(float((z - z_mod)/z_mod))
    while delta_z > 0.2:
        z = simulate_z(type_meta)
        delta_z = np.abs(float((z - z_mod)/z_mod))
    print(f"delta_z: {delta_z}")
    
    #fit gp to zero centered dates, **unredshifted** model
    unredshifted = fit_gp(model_data)
    
    #redshift model
    redshifted_model_data, true_peakmjd = redshift_model(unredshifted, model_data, z, z_mod)
            
    gp = fit_gp(redshifted_model_data)
    
    if gp == None:
        print('took the out')
        return(None, None, None)
    
    #shift simulated dates to start at t = 0
    sn_data['old_mjd'] = sn_data['mjd']
    sn_data['mjd'] = sn_data['mjd'] - np.min(sn_data['mjd'])
    
    #choose random flux multiplicative factor from range of fluxes from other events of the same type
    flux_multiplier = np.array(galactic_flux_multiplier(type_meta, model_snid))
    
    peaku = flux_multiplier[0]*sn_meta['peaku']
    peakg = flux_multiplier[1]*sn_meta['peakg']
    peakr = flux_multiplier[2]*sn_meta['peakr']
    peaki = flux_multiplier[3]*sn_meta['peaki']
    peakz = flux_multiplier[4]*sn_meta['peakz']
    peaky = flux_multiplier[5]*sn_meta['peaky']
    
    #simulate data in each band
    uflux, uflux_err, udates = simulate_band(gp, sn_meta, sn_data, 0, flux_multiplier[0], band0_err)
    gflux, gflux_err, gdates = simulate_band(gp, sn_meta, sn_data, 1, flux_multiplier[1], band1_err)
    rflux, rflux_err, rdates = simulate_band(gp, sn_meta, sn_data, 2, flux_multiplier[2], band2_err)
    iflux, iflux_err, idates = simulate_band(gp, sn_meta, sn_data, 3, flux_multiplier[3], band3_err)
    zflux, zflux_err, zdates = simulate_band(gp, sn_meta, sn_data, 4, flux_multiplier[4], band4_err)
    Yflux, Yflux_err, Ydates = simulate_band(gp, sn_meta, sn_data, 5, flux_multiplier[5], band5_err)
    
    #compile single band data into Tables
    utable = Table([np.repeat(target_snid, len(udates)), udates, np.repeat(0, len(udates)), uflux, uflux_err],
                   names = ('object_id', 'mjd', 'passband', 'flux', 'flux_err'))
    gtable = Table([np.repeat(target_snid, len(gdates)), gdates, np.repeat(1, len(gdates)), gflux, gflux_err],
                   names = ('object_id', 'mjd', 'passband', 'flux', 'flux_err'))
    rtable = Table([np.repeat(target_snid, len(rdates)), rdates, np.repeat(2, len(rdates)), rflux, rflux_err],
                   names = ('object_id', 'mjd', 'passband', 'flux', 'flux_err'))
    itable = Table([np.repeat(target_snid, len(idates)), idates, np.repeat(3, len(idates)), iflux, iflux_err],
                   names = ('object_id', 'mjd', 'passband', 'flux', 'flux_err'))
    ztable = Table([np.repeat(target_snid, len(zdates)), zdates, np.repeat(4, len(zdates)), zflux, zflux_err],
                   names = ('object_id', 'mjd', 'passband', 'flux', 'flux_err'))
    Ytable = Table([np.repeat(target_snid, len(Ydates)), Ydates, np.repeat(5, len(Ydates)), Yflux, Yflux_err],
                   names = ('object_id', 'mjd', 'passband', 'flux', 'flux_err'))
    
    #compile single band data Tables into one lcdata Table
    simulated_data = vstack([utable, gtable, rtable, itable, ztable, Ytable])
    
    #shift times back to original mjds
    simulated_data['mjd'] = simulated_data['mjd'] + np.min(sn_data['old_mjd'])
    true_peakmjd = true_peakmjd + np.min(sn_data['old_mjd'])
    redshifted_model_data['mjd'] = redshifted_model_data['mjd'] + np.min(sn_data['old_mjd'])
    
    #create simulated metadata Table
    simulated_meta = Table(names=('model_id', 'true_target', 'object_id', 'sim_z', 'sim_timeshift', 'true_peakmjd', 'ra', 'decl', 'mwebv', 'ddf_bool', 'peaku', 'peakg', 'peakr', 'peaki', 'peakz', 'peaky'))
    simulated_meta.add_row([int(model_snid), int(model_type), int(target_snid), z, time_shift, true_peakmjd, sn_meta['ra'], sn_meta['decl'], sn_meta['mwebv'], 1, peaku, peakg, peakr, peaki, peakz, peaky])
    
    u_snr = np.sqrt((utable['flux']/utable['flux_err'])**2)
    g_snr = np.sqrt((gtable['flux']/gtable['flux_err'])**2)
    r_snr = np.sqrt((rtable['flux']/rtable['flux_err'])**2)
    i_snr = np.sqrt((itable['flux']/itable['flux_err'])**2)
    z_snr = np.sqrt((ztable['flux']/ztable['flux_err'])**2)
    y_snr = np.sqrt((Ytable['flux']/Ytable['flux_err'])**2)
    
    len_snr = np.array([len(u_snr[u_snr>5]), len(g_snr[g_snr>5]), len(r_snr[r_snr>5]), len(i_snr[i_snr>5]), len(z_snr[z_snr>5]), len(y_snr[y_snr>5])])

    if len(len_snr[len_snr>3]) > 2 and np.abs(delta_z) < 0.2 and true_peakmjd > np.min(simulated_data['mjd']) and true_peakmjd < np.max(simulated_data['mjd']):# or (len(len_snr[len_snr>3]) > 3 and delta_z < 0.5):
        return(simulated_data, simulated_meta, redshifted_model_data)
    else:
        return(None, None, None)

def plot_simulated(fig, ax, simulated_sndata, simulated_snmeta):
    ID = int(simulated_snmeta['object_id'])
    typ = SN_TYPE_ID_MAP[int(simulated_snmeta['true_target'])]
    
    udata =  simulated_sndata[simulated_sndata['passband'] == 0]
    gdata =  simulated_sndata[simulated_sndata['passband'] == 1]
    rdata =  simulated_sndata[simulated_sndata['passband'] == 2]
    idata =  simulated_sndata[simulated_sndata['passband'] == 3]
    zdata =  simulated_sndata[simulated_sndata['passband'] == 4]
    Ydata =  simulated_sndata[simulated_sndata['passband'] == 5]
    
    ax[0,0].plot(udata['mjd'], udata['flux'], '.', ms=8, c='b')
    ax[0,0].errorbar(udata['mjd'], udata['flux'], yerr=udata['flux_err'], fmt='none', ecolor='b')
    ax[0,0].set_title('u')
    ax[0,0].set_ylabel('flux [AU]')

    ax[0,1].plot(gdata['mjd'], gdata['flux'], '.', ms=8, c='c')
    ax[0,1].errorbar(gdata['mjd'], gdata['flux'], yerr=gdata['flux_err'], fmt='none', ecolor='c')
    ax[0,1].set_title('g')

    ax[0,2].plot(rdata['mjd'], rdata['flux'], '.', ms=8, c='g')
    ax[0,2].errorbar(rdata['mjd'], rdata['flux'], yerr=rdata['flux_err'], fmt='none', ecolor='g')
    ax[0,2].set_title('r')

    ax[1,0].plot(idata['mjd'], idata['flux'], '.', ms=8, c='gold')
    ax[1,0].errorbar(idata['mjd'], idata['flux'], yerr=idata['flux_err'], fmt='none', ecolor='gold')
    ax[1,0].set_title('i')
    ax[1,0].set_ylabel('flux [AU]')

    ax[1,1].plot(zdata['mjd'], zdata['flux'], '.', ms=8, c='r')
    ax[1,1].errorbar(zdata['mjd'], zdata['flux'], yerr=zdata['flux_err'], fmt='none', ecolor='r')
    ax[1,1].set_title('z')
    ax[1,1].set_xlabel('mjd [days]')

    ax[1,2].plot(Ydata['mjd'], Ydata['flux'], '.', ms=8, c='m')
    ax[1,2].errorbar(Ydata['mjd'], Ydata['flux'], yerr=Ydata['flux_err'], fmt='none', ecolor='m')
    ax[1,2].set_title('Y')

    plt.suptitle('SIMULATED ' + str(typ) + ', ID: ' + str(ID), fontsize=16)