import numpy as np
import scipy.io as sio
import scipy.signal as signal
import soundfile as sf
import os
# this sript actualy generating natural wind using marcov model 
# aiming natural smooth transition of wind nature 
# low - low - low - mid - mid - mid - high - high - high
# with jump between states
class RWTHWindGenerator:
    def __init__(self, model_path='wind_noise_model'):
        self.model_path = model_path
        
        try:
            exc_data = sio.loadmat(os.path.join(model_path, 'exc_signals.mat')) # this file contain pulses ; snipt of real wind hit in mic
            self.exc_pulses = exc_data['exc_pulses'] 
            self.n_of_exc_pulses = self.exc_pulses.shape[0]
            
            tm_data = sio.loadmat(os.path.join(model_path, 'transition_prob_gusts.mat')) # transition probability matrics ; design how wind going to change from low to mid to high
            self.transition_matrix = tm_data['transitionMatrix']
        except Exception as e:
            print(f"Error loading .mat files: {e}")
            raise
            
        self.tm_cum = np.cumsum(self.transition_matrix, axis=1)
        
        self.lpc_coeff = np.array([2.4804, -2.0032, 0.5610, -0.0794, 0.0392])
        self.lpc_order = 5
        
        self.var_excitation_noise = 0.005874 # noise strength
        self.mean_state2 = 0.005 # mid wind
        self.mean_state3 = 0.25 # high wind
        self.alpha1 = 0.15 #  loudness low wind
        self.alpha2 = 0.5  # loudness high wind

    def generate(self, duration_sec, fs=16000):
        L = int(duration_sec * fs) # no of samples
        
        excite_sig_noise = np.random.randn(L) * np.sqrt(self.var_excitation_noise) # BG npise
        states_syn = np.zeros(L, dtype=int) # states od wind
        
        state_act = 0 # 0=low, 1=mid, 2=high (Python 0-indexed)
        for k in range(L):
            r = np.random.rand() # randm number
            p = self.tm_cum[state_act] # transition probality
            # decide the next state
            if r <= p[0]:
                state_act = 0
            elif r <= p[1]:
                state_act = 1
            else:
                state_act = 2
            states_syn[k] = state_act # store the states

        #   assign loudness based on states
        g_apl_raw = np.zeros(L)
        g_apl_raw[states_syn == 1] = self.mean_state2
        g_apl_raw[states_syn == 2] = self.mean_state3
        
        # smoothing the gain to make in natural
        win1 = signal.windows.hann(10000)
        win1 /= np.sum(win1)
        #  mode='same' to keep length L
        g_apl_lt = np.abs(signal.convolve(g_apl_raw, win1, mode='same'))
        
        g_apl_st_raw = np.random.randn(L)
        win2 = signal.windows.hann(int(fs * 50e-3)) # 50ms
        win2 /= np.sum(win2)
        g_apl_st = np.abs(signal.convolve(g_apl_st_raw, win2, mode='same'))
        
        # Final amplitude variation ; or Gain
        g_final = g_apl_lt * g_apl_st
        
        # 5. Signal Generation Loop
        n = np.zeros(L)
        exc_L = 0
        idx_exc = 0
        exc_pulse_cur = np.zeros(1)
        
        for k in range(self.lpc_order, L):
            if states_syn[k] > 0: # Note: If mid or high
                if idx_exc < exc_L - 1:
                    idx_exc += 1
                else: 
                    # New random pulse
                    r_pulse = np.random.randint(0, self.n_of_exc_pulses)
                    exc_L = int(self.exc_pulses[r_pulse, -1])
                    exc_pulse_cur = self.exc_pulses[r_pulse, 0:exc_L]
                    idx_exc = 0
            
            # Determine current excitation sample
            if states_syn[k] == 0: # no wind ; weak noise
                exc_sig = excite_sig_noise[k] / 2.0
            elif states_syn[k] == 1: # low wind -> mostly noise
                exc_sig = self.alpha1 * exc_pulse_cur[idx_exc] + (1 - self.alpha1) * excite_sig_noise[k]
            else: # mid/high-> mostly pulse
                exc_sig = self.alpha2 * exc_pulse_cur[idx_exc] + (1 - self.alpha2) * excite_sig_noise[k]
            
            # LPC  Filter - it design wind using past value; that make it natural
            prediction_vec = n[k-self.lpc_order:k][::-1]
            noise_term = g_apl_lt[k] * excite_sig_noise[k] * self.var_excitation_noise
                        
            vec_to_dot = prediction_vec + (g_final[k] * exc_sig) + noise_term
            n[k] = np.dot(vec_to_dot, self.lpc_coeff)

        # Scale to safe audio range
        n = n / (np.max(np.abs(n)) + 1e-8) * 0.95
        return n

def main():
    print("start generating synth wind")
    gen = RWTHWindGenerator(model_path='wind_noise_model')
    
    duration = 10.0 # 10 seconds
    print(f"Generating {duration}s of natural wind")
    audio = gen.generate(duration)
    
    output_file = "synthetic_wind/rwth_official_python.wav"
    sf.write(output_file, audio, 16000)
    print(f"Success! Saved as: {output_file}")

if __name__ == "__main__":
    main()
