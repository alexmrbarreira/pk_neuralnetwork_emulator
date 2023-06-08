from commons import *
import tensorflow as tf

# ==========================================================================
# Get training/testing data and load the model
# ==========================================================================

# Load the data from numpy arrays; 
params_train  = np.load('data_store/data_params_train.npy')
params_valid  = np.load('data_store/data_params_valid.npy')
params_test   = np.load('data_store/data_params_test.npy')
log10pk_train = np.load('data_store/data_log10pk_train.npy')
log10pk_valid = np.load('data_store/data_log10pk_valid.npy')
log10pk_test  = np.load('data_store/data_log10pk_test.npy')

print ('')
print ('Shapes of train data')
print ('Parameters (features):', np.shape(params_train))
print ('Spectra (labels):'     , np.shape(log10pk_train))
print ('')
print ('Shapes of validation data')
print ('Parameters (features):', np.shape(params_valid))
print ('Spectra (labels):'     , np.shape(log10pk_valid))
print ('')
print ('Shapes of testing data')
print ('Parameters (features):', np.shape(params_test))
print ('Spectra (labels):'     , np.shape(log10pk_test))

model_nn = tf.keras.models.load_model('model_store/model_1.h5')

print ('')
print (model_nn.summary())
print ('')

# Model predictions 
pred_train = model_nn.predict(params_train)
pred_valid = model_nn.predict(params_valid)
pred_test  = model_nn.predict(params_test)

# ==========================================================================
# Time performance test: NN vs CAMB 
# ==========================================================================

# Choose set of parameters to make a prediction
params_values = [70., 0.022, 0.13, 3.0, 0.96, 1.1]

# Time the NN calculation
t0 = time.time()
model_nn.predict([params_values])
t1 = time.time()
print ('')
print ('For a single parameter evaluation:')
print ('The NN model took', t1-t0, 'seconds')

# Time the CAMB calculation
pars = camb.CAMBparams()
t0   = time.time()
pars.set_cosmology(H0=params_values[0], ombh2=params_values[1], omch2=params_values[2], mnu=mnu, omk=omk)
pars.InitPower.set_params(As=np.exp(params_values[3])*1.0e-10, ns=params_values[4])
pars.set_matter_power(redshifts=[params_values[5]], kmax=kmax)
# Get the nonlinear power spectrum
pars.NonLinear    = model.NonLinear_both
results           = camb.get_results(pars)
k, z, pk          = results.get_matter_power_spectrum(minkh=kmin, maxkh = kmax, npoints = nkpoints)
t1   = time.time()
print ('CAMB took', t1-t0, 'seconds')
print ('')

# ==========================================================================
# Plot parameters 
# ==========================================================================

labelsize   = 30
ticksize    = 30
ticklength_major  = 10.
ticklength_minor  = 5.
tickwidth   = 1.5
tickpad     = 6.
title_font  = 30
text_font   = 20
legend_font = 20

ind_check = 0 # index of specific training/validation/testing example to inspect

min_y = 0.0
max_y = 0.1
min_y2 = 1.0e-8
max_y2 = 1.0e-2

def annotate_params(params, ind):
    fonthere = 18
    plt.annotate(H0_symbol      + '=' + "%.2f" % params[ind][0], xy = (0.10, 0.40), xycoords = 'axes fraction', c = 'k', fontsize = fonthere)
    plt.annotate(ombh2_symbol   + '=' + "%.5f" % params[ind][1], xy = (0.10, 0.33), xycoords = 'axes fraction', c = 'k', fontsize = fonthere)
    plt.annotate(omch2_symbol   + '=' + "%.3f" % params[ind][2], xy = (0.10, 0.26), xycoords = 'axes fraction', c = 'k', fontsize = fonthere)
    plt.annotate(lnAse10_symbol + '=' + "%.2f" % params[ind][3], xy = (0.10, 0.19), xycoords = 'axes fraction', c = 'k', fontsize = fonthere)
    plt.annotate(ns_symbol      + '=' + "%.2f" % params[ind][4], xy = (0.10, 0.12), xycoords = 'axes fraction', c = 'k', fontsize = fonthere)
    plt.annotate(z_symbol       + '=' + "%.1f" % params[ind][5], xy = (0.10, 0.05), xycoords = 'axes fraction', c = 'k', fontsize = fonthere)

# ==========================================================================
# Inspection of specific case 
# ==========================================================================
fig0 = plt.figure(0, figsize=(17., 7.))
fig0.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.15, wspace = 0.35, hspace = 0.30)

# Training accuracy
panel = fig0.add_subplot(1,3,1)
plt.title('Training accuracy; example ' + str(ind_check), fontsize = title_font)
plt.plot(kk, log10pk_train[ind_check], c = 'k', linewidth = 2., linestyle = 'solid', label = 'True')
plt.plot(kk, pred_train[ind_check]   , c = 'b', linewidth = 2., linestyle = 'dashed', label = 'Prediction')
plt.xlim(kmin, kmax)
plt.ylim(-3., 4.)
plt.xscale('log')
plt.xlabel(r'$k\ \left[h/{\rm Mpc}\right]$' , fontsize = labelsize)
plt.ylabel(r'${\rm log}_{10}P(k)\ \left[{\rm Mpc}^3/h^3\right]$'          , fontsize = labelsize)
plt.tick_params(length=ticklength_major, width=tickwidth , bottom=True, top=True, left=True, right=True, direction = 'in', which = 'major', pad = tickpad, labelsize = ticksize)
params = {'legend.fontsize': legend_font}; plt.rcParams.update(params); plt.legend(loc = 'upper right', ncol = 1)
annotate_params(params_train, ind_check)

# Validation accuracy
panel = fig0.add_subplot(1,3,2)
plt.title('Validation accuracy; example ' + str(ind_check), fontsize = title_font)
plt.plot(kk, log10pk_valid[ind_check], c = 'k', linewidth = 2., linestyle = 'solid', label = 'True')
plt.plot(kk, pred_valid[ind_check]   , c = 'g', linewidth = 2., linestyle = 'dashed', label = 'Prediction')
plt.xlim(kmin, kmax)
plt.ylim(-3., 4.)
plt.xscale('log')
plt.xlabel(r'$k\ \left[h/{\rm Mpc}\right]$' , fontsize = labelsize)
plt.tick_params(length=ticklength_major, width=tickwidth , bottom=True, top=True, left=True, right=True, direction = 'in', which = 'major', pad = tickpad, labelsize = ticksize)
params = {'legend.fontsize': legend_font}; plt.rcParams.update(params); plt.legend(loc = 'upper right', ncol = 1)
annotate_params(params_valid, ind_check)

# Testing accuracy
panel = fig0.add_subplot(1,3,3)
plt.title('Testing accuracy; example ' + str(ind_check), fontsize = title_font)
plt.plot(kk, log10pk_test[ind_check], c = 'k', linewidth = 2., linestyle = 'solid', label = 'True')
plt.plot(kk, pred_test[ind_check]   , c = 'r', linewidth = 2., linestyle = 'dashed', label = 'Prediction')
plt.xlim(kmin, kmax)
plt.ylim(-3., 4.)
plt.xscale('log')
plt.xlabel(r'$k\ \left[h/{\rm Mpc}\right]$' , fontsize = labelsize)
plt.tick_params(length=ticklength_major, width=tickwidth , bottom=True, top=True, left=True, right=True, direction = 'in', which = 'major', pad = tickpad, labelsize = ticksize)
params = {'legend.fontsize': legend_font}; plt.rcParams.update(params); plt.legend(loc = 'upper right', ncol = 1)
annotate_params(params_test, ind_check)

# ==========================================================================
# Relative difference of the spectra 
# ==========================================================================
fig1 = plt.figure(1, figsize=(17., 7.))
fig1.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.15, wspace = 0.35, hspace = 0.30)

# Training
panel = fig1.add_subplot(1,3,1)
plt.title('Training', fontsize = title_font)
store_rel_diffs = np.zeros(n_train)
for i in range(n_train):
    plt.plot(kk, abs(10.**pred_train[i]/10.**log10pk_train[i]-1), c = 'grey', linewidth = 0.1, linestyle = 'solid')
    store_rel_diffs[i] = np.max(abs(10.**pred_train[i]/10.**log10pk_train[i]-1))
plt.plot(kk, abs(10.**pred_train[ind_check]/10.**log10pk_train[ind_check]-1), c = 'y', linewidth = 2.0, linestyle = 'solid', label = 'Selected single case')
plt.plot(kk, np.mean(abs(10.**pred_train/10.**log10pk_train-1), axis = 0), c = 'b', linewidth = 2., linestyle = 'solid', label = 'Mean')
plt.xlim(kmin, kmax)
plt.ylim(min_y, max_y)
plt.xscale('log')
plt.xlabel(r'$k\ \left[h/{\rm Mpc}\right]$'        , fontsize = labelsize)
plt.ylabel(r'$|P(k)^{\rm predicted} / P(k)^{\rm true}-1|$' , fontsize = labelsize)
plt.tick_params(length=ticklength_major, width=tickwidth , bottom=True, top=True, left=True, right=True, direction = 'in', which = 'major', pad = tickpad, labelsize = ticksize)
params = {'legend.fontsize': legend_font}; plt.rcParams.update(params); plt.legend(loc = 'upper left', ncol = 1)

# Validation
panel = fig1.add_subplot(1,3,2)
plt.title('Validation', fontsize = title_font)
store_rel_diffs = np.zeros(n_valid)
for i in range(n_valid):
    plt.plot(kk, abs(10.**pred_valid[i]/10.**log10pk_valid[i]-1), c = 'grey', linewidth = 0.1, linestyle = 'solid')
    store_rel_diffs[i] = np.max(abs(10.**pred_valid[i]/10.**log10pk_valid[i]-1))
plt.plot(kk, abs(10.**pred_valid[ind_check]/10.**log10pk_valid[ind_check]-1), c = 'y', linewidth = 2.0, linestyle = 'solid', label = 'Selected single case')
plt.plot(kk, np.mean(abs(10.**pred_valid/10.**log10pk_valid-1), axis = 0), c = 'g', linewidth = 2., linestyle = 'solid', label = 'Mean')
plt.xlim(kmin, kmax)
plt.ylim(min_y, max_y)
plt.xscale('log')
plt.xlabel(r'$k\ \left[h/{\rm Mpc}\right]$'        , fontsize = labelsize)
plt.tick_params(length=ticklength_major, width=tickwidth , bottom=True, top=True, left=True, right=True, direction = 'in', which = 'major', pad = tickpad, labelsize = ticksize)
params = {'legend.fontsize': legend_font}; plt.rcParams.update(params); plt.legend(loc = 'upper left', ncol = 1)

# Testing
panel = fig1.add_subplot(1,3,3)
plt.title('Testing', fontsize = title_font)
store_rel_diffs = np.zeros(n_test)
for i in range(n_test):
    plt.plot(kk, abs(10.**pred_test[i]/10.**log10pk_test[i]-1), c = 'grey', linewidth = 0.1, linestyle = 'solid')
    store_rel_diffs[i] = np.max(abs(10.**pred_test[i]/10.**log10pk_test[i]-1))
plt.plot(kk, abs(10.**pred_test[ind_check]/10.**log10pk_test[ind_check]-1), c = 'y', linewidth = 2.0, linestyle = 'solid', label = 'Selected single case')
plt.plot(kk, np.mean(abs(10.**pred_test/10.**log10pk_test-1), axis = 0), c = 'r', linewidth = 2., linestyle = 'solid', label = 'Mean')
plt.xlim(kmin, kmax)
plt.ylim(min_y, max_y)
plt.xscale('log')
plt.xlabel(r'$k\ \left[h/{\rm Mpc}\right]$'        , fontsize = labelsize)
plt.tick_params(length=ticklength_major, width=tickwidth , bottom=True, top=True, left=True, right=True, direction = 'in', which = 'major', pad = tickpad, labelsize = ticksize)
params = {'legend.fontsize': legend_font}; plt.rcParams.update(params); plt.legend(loc = 'upper left', ncol = 1)

# ==========================================================================
# Mean squared error of log10 spectra 
# ==========================================================================
fig2 = plt.figure(2, figsize=(17., 7.))
fig2.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.15, wspace = 0.35, hspace = 0.30)

# Training
panel = fig2.add_subplot(1,3,1)
plt.title('Training', fontsize = title_font)
store_mse = np.zeros(n_train)
for i in range(n_train):
    plt.plot(kk, (pred_train[i]-log10pk_train[i])**2., c = 'grey', linewidth = 0.1, linestyle = 'solid')
    store_mse[i] = np.mean((pred_train[i]-log10pk_train[i])**2.)
plt.axhline(np.mean(store_mse), c = 'b', linewidth = 2.0, linestyle = 'solid', label = 'Mean')
plt.plot(kk, (pred_train[ind_check] - log10pk_train[ind_check])**2., c = 'y', linewidth = 2.0, linestyle = 'solid', label = 'Selected single case')
plt.xlim(kmin, kmax)
plt.ylim(min_y2, max_y2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$k\ \left[h/{\rm Mpc}\right]$'        , fontsize = labelsize)
plt.ylabel(r'Mean sq. error: ${\rm log}_{10}P(k)\ \left[{\rm Mpc}^3/h^3\right]$' , fontsize = labelsize)
plt.tick_params(length=ticklength_major, width=tickwidth , bottom=True, top=True, left=True, right=True, direction = 'in', which = 'major', pad = tickpad, labelsize = ticksize)
params = {'legend.fontsize': legend_font}; plt.rcParams.update(params); plt.legend(loc = 'upper left', ncol = 1)

# Validation
panel = fig2.add_subplot(1,3,2)
plt.title('Validation', fontsize = title_font)
store_mse = np.zeros(n_valid)
for i in range(n_valid):
    plt.plot(kk, (pred_valid[i]-log10pk_valid[i])**2., c = 'grey', linewidth = 0.1, linestyle = 'solid')
    store_mse[i] = np.mean((pred_valid[i]-log10pk_valid[i])**2.)
plt.axhline(np.mean(store_mse), c = 'g', linewidth = 2.0, linestyle = 'solid', label = 'Mean')
plt.plot(kk, (pred_valid[ind_check] - log10pk_valid[ind_check])**2., c = 'y', linewidth = 2.0, linestyle = 'solid', label = 'Selected single case')
plt.xlim(kmin, kmax)
plt.ylim(min_y2, max_y2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$k\ \left[h/{\rm Mpc}\right]$'        , fontsize = labelsize)
plt.tick_params(length=ticklength_major, width=tickwidth , bottom=True, top=True, left=True, right=True, direction = 'in', which = 'major', pad = tickpad, labelsize = ticksize)
params = {'legend.fontsize': legend_font}; plt.rcParams.update(params); plt.legend(loc = 'upper left', ncol = 1)

# Testing
panel = fig2.add_subplot(1,3,3)
plt.title('Testing', fontsize = title_font)
store_mse = np.zeros(n_test)
for i in range(n_test):
    plt.plot(kk, (pred_test[i]-log10pk_test[i])**2., c = 'grey', linewidth = 0.1, linestyle = 'solid')
    store_mse[i] = np.mean((pred_test[i]-log10pk_test[i])**2.)
plt.axhline(np.mean(store_mse), c = 'r', linewidth = 2.0, linestyle = 'solid', label = 'Mean')
plt.plot(kk, (pred_test[ind_check] - log10pk_test[ind_check])**2., c = 'y', linewidth = 2.0, linestyle = 'solid', label = 'Selected single case')
plt.xlim(kmin, kmax)
plt.ylim(min_y2, max_y2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$k\ \left[h/{\rm Mpc}\right]$'        , fontsize = labelsize)
plt.tick_params(length=ticklength_major, width=tickwidth , bottom=True, top=True, left=True, right=True, direction = 'in', which = 'major', pad = tickpad, labelsize = ticksize)
params = {'legend.fontsize': legend_font}; plt.rcParams.update(params); plt.legend(loc = 'upper left', ncol = 1)

fig0.savefig('fig_store/fig_diagnostic_single_case.png')
fig1.savefig('fig_store/fig_diagnostic_spectra_relative_difference.png')
fig2.savefig('fig_store/fig_diagnostic_logspectra_mean_squared_error.png')

plt.show()


