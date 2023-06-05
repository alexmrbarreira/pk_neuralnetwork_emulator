from commons import *
import tensorflow as tf

# ==========================================================================
# Get training/validation data 
# ==========================================================================

# Load the data from numpy arrays; 
params_train  = np.load('data_store/data_params_train.npy')
params_valid  = np.load('data_store/data_params_valid.npy')
log10pk_train = np.load('data_store/data_log10pk_train.npy')
log10pk_valid = np.load('data_store/data_log10pk_valid.npy')

print ('')
print ('Shapes of train data')
print ('Parameters (features):', np.shape(params_train))
print ('Spectra (labels):'     , np.shape(log10pk_train))
print ('')
print ('Shapes of validation data')
print ('Parameters (features):', np.shape(params_valid))
print ('Spectra (labels):'     , np.shape(log10pk_valid))

# ==========================================================================
# Create, train model and save the model
# ==========================================================================

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='sigmoid', input_shape=(6,)),
    tf.keras.layers.Dense(64, activation='sigmoid'),
    tf.keras.layers.Dense(64, activation='sigmoid'),
    tf.keras.layers.Dense(nkpoints)
])

# Callback to reduce learning rate when loss stops decreasing
initial_lr = 0.007 #0.01
reduce_lr  = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.95, patience=75, min_lr=1.0e-7)

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = initial_lr), loss = 'mean_squared_error')

print ('')
print ('Model summary:')
print (model.summary())
print ('Model input shape:' , model.input_shape)
print ('Model output shape:', model.output_shape)

nepochs    = 10000
batch_size = 32
history = model.fit(params_train, log10pk_train, epochs = nepochs, shuffle=True, batch_size = batch_size, validation_data = (params_valid, log10pk_valid), callbacks = [reduce_lr])

model.save('model_store/model_1.h5')

# ==========================================================================
# Training history 
# ==========================================================================

ee = range(1, nepochs+1)

fig0 = plt.figure(0, figsize=(10., 7.))
fig0.subplots_adjust(left=0.14, right=0.97, top=0.92, bottom=0.12, wspace = 0.35, hspace = 0.30)
panel = fig0.add_subplot(1,1,1)
plt.title('Training history', fontsize = 30)
plt.plot(ee, history.history['loss']    , c = 'b', linewidth = 2. , label = 'Loss')
plt.plot(ee, history.history['val_loss'], c = 'g', linewidth = 2. ,label = 'Validation loss')
plt.plot(ee, history.history['lr']      , c = 'r', linewidth = 2. ,label = 'Learning rate')
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.5*min(ee), 1.20*max(ee))
plt.ylim(1.0e-6, 2.0e0)
plt.xlabel(r'Epochs+1' , fontsize = 28)
plt.ylabel(r'Loss and learning rate'          , fontsize = 28)
plt.tick_params(length=10., width=1.5 , bottom=True, top=True, left=True, right=True, direction = 'in', which = 'major', pad = 6, labelsize = 30)
params = {'legend.fontsize': 20}; plt.rcParams.update(params); plt.legend(loc = 'upper right', ncol = 1)

fig0.savefig('fig_store/fig_training_history.png')

plt.show()
