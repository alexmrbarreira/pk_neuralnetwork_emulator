from commons import *

# Initialize camb
pars    = camb.CAMBparams()

# ==========================================================================
# Generate Latin Hypercube of parameters for training/testing
# ==========================================================================

lh_sampler = sp.stats.qmc.LatinHypercube(d = n_dim)
lh_train   = lh_sampler.random(n=int(n_train))
lh_valid   = lh_sampler.random(n=int(n_valid))
lh_test    = lh_sampler.random(n=int(n_test))

# ==========================================================================
# Compute nonlinear power spectrum for training/validation/testing
# ==========================================================================

def generate_data(lh, n_examples, string_name):
    params_save       = np.zeros([n_examples, n_dim])
    log10pk_save      = np.zeros([n_examples, nkpoints])
    for i in range(n_examples):
        if(np.mod(i,n_examples/10) == 0):
            print ('Dealt with', i, 'out of', n_examples)
        # Set parameters from LH
        H0_now      = lh[i,0]*(     H0_max -      H0_min) +      H0_min
        ombh2_now   = lh[i,1]*(  ombh2_max -   ombh2_min) +   ombh2_min
        omch2_now   = lh[i,2]*(  omch2_max -   omch2_min) +   omch2_min
        lnAse10_now = lh[i,3]*(lnAse10_max - lnAse10_min) + lnAse10_min
        ns_now      = lh[i,4]*(     ns_max -      ns_min) +      ns_min
        z_now       = lh[i,5]*(      z_max -       z_min) +       z_min
        pars.set_cosmology(H0=H0_now, ombh2=ombh2_now, omch2=omch2_now, mnu=mnu, omk=omk)
        pars.InitPower.set_params(As=np.exp(lnAse10_now)*1.0e-10, ns=ns_now)
        pars.set_matter_power(redshifts=[z_now], kmax=kmax)
        # Get the nonlinear power spectrum
        pars.NonLinear    = model.NonLinear_both
        results           = camb.get_results(pars)
        k, z, pk          = results.get_matter_power_spectrum(minkh=kmin, maxkh = kmax, npoints = nkpoints)
        params_save[i,:]       = np.array([H0_now, ombh2_now, omch2_now, lnAse10_now, ns_now, z_now])
        log10pk_save[i,:]      = np.log10(pk[0])
    np.save('data_store/data_params_'       + string_name+'.npy', params_save)       # parameters
    np.save('data_store/data_log10pk_'      + string_name+'.npy', log10pk_save)      # log10 spectra

print ('')
print ('Generating training data ... ')
generate_data(lh_train, n_train, 'train')

print ('')
print ('Generating validation data ... ')
generate_data(lh_valid, n_valid, 'valid')

print ('')
print ('Generating testing data ... ')
generate_data(lh_test, n_test, 'test')
