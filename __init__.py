from pyinstruments import CurveDB
import matplotlib.pylab as plt
import numpy as np
from scipy.optimize import leastsq
from scipy.constants import k, hbar, h



class OMIT(object):

    def __init__(self, curve_id, large_curve_id):
        self.curve_id = curve_id
        self.curve=CurveDB.objects.get(id=curve_id)
        self.n_curves = len(self.curve.childs.all())
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
        self.normalization = np.conjugate(0.5*(np.mean(
            self.large_curve.data.get_values()[
                                     :10])+np.mean(
            self.large_curve.data.get_values()[-10:])))
        self.PUMP_FREQUENCY = self.curve.params["pump_frequency"]
        self.DELTA_PUMP = self.PUMP_FREQUENCY - self.CAVITY_FREQUENCY
        self.MECHANICAL_FREQUENCY = self.curve.params['mechanical_frequency']
        self.datas = []

    def transmission(self, nu_probe, n, nu_pump, nu_cav,
                     kappa_cav_hz,
                     K_coop, Q_m, nu_m, eta_c):
        coop = n * np.abs(K_coop)
        omega = 2. * np.pi * (nu_probe - nu_pump)
        kappa = kappa_cav_hz * 2. * np.pi
        omega_m = nu_m * 2. * np.pi
        gamma_m = omega_m/np.abs(Q_m)
        delta = 2. * np.pi * (nu_pump -
                              nu_cav)+0.5*coop*kappa*gamma_m*omega_m/(
            omega_m**2+0.25*gamma_m**2)
        f = coop / ((2 * Q_m * (1. - omega / omega_m) - 1j) * (
            1 + 2j * (delta - omega) / kappa))
            # #coop / (-(2 * Q_m * (1. - omega / omega_m) - 1j) * (
            #1 + 2j * (delta - omega) / kappa))
        #results from Sam's paper
        t = 1. - 0.5 * (1. + 1j * f) * eta_c * kappa / (
            -1j * (delta + omega) + kappa / 2. + 2. * delta * f)
        #1. - 0.5 * (1. + 1j * f) * eta_c * kappa / (
            #1j * (delta + omega) + kappa / 2. + 2. * delta * f)
        #results from handmade computations
        t_bis = 1.-eta_c/(-2j*(
            delta+omega)/kappa+1-coop/(2j*(
            omega-omega_m)/gamma_m-1))
        return t_bis

    def fit_func(self, probes, n, params):
        K_coop, Q_m, *nu_m_s = params
        if len(nu_m_s)>1:
            nu_m = nu_m_s[self.ns.index(n)]
        else:
            nu_m = self.MECHANICAL_FREQUENCY
        return [self.transmission(nu_probe,
                                n,
                                self.PUMP_FREQUENCY,
                                self.CAVITY_FREQUENCY,
                                self.KAPPA_HZ,
                                K_coop,
                                Q_m,
                                nu_m,
                                self.ETA_C) for nu_probe in probes]

    def fit_global(self, guess):
        self.fitted_params, self.flag = leastsq(self.to_minimize,
                                                guess,
                                                (self.ns_all,self.probes_all,
                                                 self.datas_all))

    def fit(self, guess, n, datas):
        self.fitted_params, self.flag = leastsq(self.to_minimize,
                                                guess,
                                                ([n for i in datas],
                                                 self.probes,
                                                 datas))

    def to_minimize(self, params, ns, probes, datas):

        K_coop, Q_m, *nu_m_s = params
        nu_m_s_all = [nu_m for nu_m in nu_m_s for i in range(int(len(
            datas)/self.n_curves))]
        return np.array([np.abs(self.transmission(nu_probe, n,
                                        self.PUMP_FREQUENCY,
                                                self.CAVITY_FREQUENCY,
                                                self.KAPPA_HZ, K_coop, Q_m,
                                                nu_m,self.ETA_C)-data/self.normalization)
                         for
                       nu_probe, n, data, nu_m in zip(probes, ns,
                                                datas, nu_m_s_all)]).flatten()

    def plot_global_fit(self, one_plot=False, plot_guess=True):
        for ind, tuple in enumerate(zip(self.datas, self.ns)) :
            datas, n = tuple
            if (ind==0 and one_plot) or not one_plot:
                plt.figure()
            self.probes_fit = np.linspace(np.min(self.probes), np.max(
                self.probes), 10000)
            self.plot_re_im(self.probes_fit, self.fit_func(self.probes_fit, n,
                                                   self.fitted_params),
                            label='fit')
            if plot_guess:
                self.plot_re_im(self.probes_fit, self.fit_func(self.probes_fit, n,
                                                       self.guess()),
                            label='guess')
            self.plot_re_im(self.probes, datas/self.normalization, label='raw data')
            if not one_plot:
                plt.legend()


    def plot_fit(self, n, datas):
        self.plot_re_im(self.probes, self.fit_func(self.probes, n,
                                                   self.fitted_params),
                        label='fit')
        self.plot_re_im(self.probes, self.fit_func(self.probes, n,
                                                   self.guess()),
                        label='guess')
        self.plot_re_im(self.probes, datas / self.normalization,
                        label='raw data')
        plt.legend()


    def plot_re_im(self, x, z, **kwds):
        sub = plt.subplot(211)
        plt.plot(x, np.real(z), **kwds)
        plt.ylim([-1,1])
        plt.subplot(212, sharex=sub)
        plt.plot(x, np.imag(z), **kwds)
        plt.ylim([-1, 1])

    def get_number_of_photons(self, power, attenuation):
        P0=1e-3 # 1dBm == 1mW
        a_squared=((P0*10**(0.1*(power+attenuation)))/(h*self.CAVITY_FREQUENCY))
        return a_squared*4*self.KAPPA_C_HZ/(
            self.KAPPA_HZ**2+4.*self.DELTA_PUMP**2)

    def get_data_from_scans(self):
        ns_all = []
        ns = []
        probes = []
        datas_all = []
        datas = []
        for child in self.curve.childs.all():
            if not "attenuation_pump" in child.params.keys():
                ATTENUATION_PUMP = child.params["attenuation"] + 40
                child.params["attenuation_pump"] = ATTENUATION_PUMP
                child.save()
            else:
                ATTENUATION_PUMP = child.params["attenuation_pump"]
            power = child.params["Sideband_power"]
            data = np.conjugate(np.array(child.data.get_values()))
            probe = np.array(child.data.index)
            n = self.get_number_of_photons(power, ATTENUATION_PUMP)
            ns.append(n)
            datas.append(data)
            for t, f in zip(data, probe):
                datas_all.append(t)
                probes.append(f)
                ns_all.append(n)
        self.probes_all = probes
        self.ns_all = ns_all
        self.ns = ns
        self.datas_all = datas_all
        self.probes = self.probes_all[:int(len(self.probes_all) / len(
            self.curve.childs.all()))]
        self.datas = datas

    def plot_large_guess(self):
        plt.figure()
        self.plot_re_im(self.large_curve.data.index,
                   np.conjugate(self.large_curve.data.get_values()) /
                        self.normalization,
                        label='raw')
        params_large_guess = 0,1000,1000
        self.plot_re_im(self.large_curve.data.index,
                   self.fit_func(np.array(self.large_curve.data.index),
                                 0,
                                 params_large_guess)
                        , label="fit 0 coop")
        plt.legend()

    def guess(self):
        Q_m = 4e6
        ind = int(np.argmax(self.ns))
        grandchild = self.curve.childs.all()[ind]
        for fits in grandchild.childs.all():
            fits.delete()
        fit = grandchild.fit('lorentz_complex_sam', autosave=True)
        n = self.ns[ind]
        x0 = fit.params['x0']
        bandwidth = fit.params['bandwidth']
        nu_m = x0-self.PUMP_FREQUENCY
        natural_bandwidth = nu_m/Q_m
        coop = bandwidth/natural_bandwidth
        K_coop = coop/n
        self.MECHANICAL_FREQUENCY=nu_m
        return [K_coop, Q_m, *[nu_m for i in range(self.n_curves)]]



