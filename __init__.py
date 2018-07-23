from pyinstruments import CurveDB
import matplotlib.pylab as plt
import numpy as np
from scipy.optimize import leastsq
from scipy.constants import k, hbar, h



class OMIT(object):
    def __init__(self, curve_id, large_curve_id):
        self.curve_id = curve_id
        self.curve=CurveDB.objects.get(id=curve_id)
        self.large_curve_id = large_curve_id
        self.large_curve = CurveDB.objects.get(id=large_curve_id)
        self.portion_of_large_curve = self.large_curve.childs.first().childs.first()
        self.CAVITY_FREQUENCY = self.portion_of_large_curve.params["x0"]
        self.cavity_Qi = self.portion_of_large_curve.params["Q_i"]
        self.cavity_Qc = self.portion_of_large_curve.params["Q_c"]
        self.cavity_Q = self.portion_of_large_curve.params["Q"]
        self.ETA_C = self.cavity_Qi / (self.cavity_Qc + self.cavity_Qi)
        self.KAPPA_HZ = self.CAVITY_FREQUENCY / self.cavity_Q
        self.KAPPA_C_HZ = self.CAVITY_FREQUENCY / self.cavity_Qc
        self.normalization = np.mean(self.large_curve.data.get_values()[:10])
        self.PUMP_FREQUENCY = self.curve.params["pump_frequency"]
        self.DELTA_PUMP = self.PUMP_FREQUENCY - self.CAVITY_FREQUENCY

    def transmission(self, nu_probe, n, nu_pump, nu_cav, kappa_cav_hz,
                     K_coop, Q_m, nu_m, eta_c):
        coop = n * K_coop
        omega = 2. * np.pi * (nu_probe - nu_pump)
        delta = 2. * np.pi * (nu_pump - nu_cav)
        kappa = kappa_cav_hz * 2. * np.pi
        omega_m = nu_m * 2. * np.pi
        f = coop / (-(2 * Q_m * (1. - omega / omega_m) - 1j) * (
            1 + 2j * (delta - omega) / kappa))
        t = 1. - 0.5 * (1. + 1j * f) * eta_c * kappa / (
            1j * (delta + omega) + kappa / 2. + 2. * delta * f)
        return t

    def fit_func(self, nu_probe_s, n, K_coop, Q_m, nu_m, kappa_hz,
                 cavity_frequency):
        return self.transmission(nu_probe_s, self.PUMP_FREQUENCY, n,
                                 cavity_frequency,
                            kappa_hz, K_coop, Q_m, nu_m, self.ETA_C)


def to_minimize(params, nu_probe_s):
    n, K_coop, Q_m, nu_m, kappa_hz, cavity_frequency = params
    return np.abs(fit_func(nu_probe_s, n, K_coop, Q_m, nu_m, kappa_hz,
                           cavity_frequency) - datas)


def plot_re_im(x, z, **kwds):
    sub = plt.subplot(211)
    plt.plot(x, np.real(z), **kwds)
    plt.subplot(212, sharex=sub)
    plt.plot(x, np.imag(z), **kwds)


# step 3: plot data and guess on large scale
plt.close("all")
plot_re_im(large_curve.data.index,
           large_curve.data.get_values() / normalization, label='raw')
plot_re_im(large_curve.data.index,
           fit_func(np.array(large_curve.data.index), 0, 0, 1000, 1000,
                    KAPPA_HZ, CAVITY_FREQUENCY), label="fit 0 coop")
plt.legend()

coop_s = []
n_s = []
n_s_fit = []
power_s = []
nu_m_s = []
Q_m_s = []
kappa_hz_s = []
cavity_frequency_s = []

# step 4: plot data and guess OMIT
for child in parent.childs.all():
    if not "attenuation_pump" in child.params.keys():
        ATTENUATION_PUMP = child.params["attenuation"] + 40
        child.params["attenuation_pump"] = ATTENUATION_PUMP
        child.save()
    else:
        ATTENUATION_PUMP = child.params["attenuation_pump"]
    power = child.params["Sideband_power"]
    power_s.append(power)
    P0 = 1e-3  # 1dBm == 1mW
    a_squared = P0 * 10. ** (0.1 * (power + ATTENUATION_PUMP)) / (
    h * CAVITY_FREQUENCY)
    n = a_squared * 4 * KAPPA_C_HZ / (KAPPA_HZ ** 2 + 4 * DELTA_PUMP ** 2)
    n_s.append(n)
    datas = child.data.get_values() / normalization
    probe_frequencies = np.array(child.data.index)
    plt.figure()
    GUESS = [n, 0.05, 20e6, 664485, KAPPA_HZ,
             CAVITY_FREQUENCY]  # n, K_coop, Q_m, nu_m, kappa_hz, cavity_frequency
    plot_re_im(probe_frequencies, datas, label='raw')
    plot_re_im(probe_frequencies, fit_func(probe_frequencies, *GUESS),
               label='guess')
    plt.legend()

    # step 5: fit OMIT
    res, flag = leastsq(to_minimize, GUESS, probe_frequencies)
    n, K_coop, Q_m, nu_m, kappa_hz, cavity_frequency = res
    n_s_fit.append(n)
    coop_s.append(K_coop)
    Q_m_s.append(Q_m)
    nu_m_s.append(nu_m)
    kappa_hz_s.append(kappa_hz)
    cavity_frequency_s.append(cavity_frequency)
    print(res, flag)
    plot_re_im(probe_frequencies, fit_func(probe_frequencies, *res),
               label='fit')
    plt.legend()
