[1mdiff --git a/bajes/obs/kn/approx/grossman_kbp/__init__.py b/bajes/obs/kn/approx/grossman_kbp/__init__.py[m
[1mindex ccab16f..8bbee41 100644[m
[1m--- a/bajes/obs/kn/approx/grossman_kbp/__init__.py[m
[1m+++ b/bajes/obs/kn/approx/grossman_kbp/__init__.py[m
[36m@@ -316,6 +316,40 @@[m [mclass korobkin_barnes_grossman_perego_et_al_two_nrfit_anisotropic_wrapper(Korobk[m
         # initialize flux factor interpolator[m
         self.ff_interp  = initialize_flux_factors(n_rays)[m
 [m
[32m+[m[32mclass korobkin_barnes_grossman_perego_et_al_two_joint_grb_isotropic_wrapper(KorobkinBarnesGrossmanPeregoEtAl):[m
[32m+[m
[32m+[m[32m    def __init__(self, times, lambdas, v_min=1.e-7, n_v=400, t_start=1., **kwargs):[m
[32m+[m
[32m+[m[32m        # initialize angular axis[m
[32m+[m[32m        # obs. the inclinations angle is divided in 12 slices[m
[32m+[m[32m        n_rays = 12[m
[32m+[m[32m        angles, omegas  = initialize_angular_axis(n_rays//2)[m
[32m+[m[32m        self.angles = angles[m
[32m+[m[32m        self.omegas = omegas[m
[32m+[m
[32m+[m[32m        # initialize nuclear heating rate model[m
[32m+[m[32m        heat    = Heating()[m
[32m+[m
[32m+[m[32m        # check time axis[m
[32m+[m[32m        if any(times < 0.):[m
[32m+[m[32m            times += t_start - times[0][m
[32m+[m[32m        self.times  = times[m
[32m+[m
[32m+[m[32m        # initialize shell components[m
[32m+[m[32m        self.ncomponents    = 2[m
[32m+[m[32m        self.components     = [Shell(name='dynamics',    geom='isotropic',   time=times,[m
[32m+[m[32m                                     angles=angles, omegas=omegas,      heat=heat,[m
[32m+[m[32m                                     v_min=v_min,   n_v=n_v),[m
[32m+[m[32m                               Shell(name='secular',   geom='isotropic',   time=times,[m
[32m+[m[32m                                     angles=angles, omegas=omegas,      heat=heat,[m
[32m+[m[32m                                     v_min=v_min,   n_v=n_v)][m
[32m+[m
[32m+[m[32m        # initialize quantities[m
[32m+[m[32m        self.lambdas    = lambdas[m
[32m+[m
[32m+[m[32m        # initialize flux factor interpolator[m
[32m+[m[32m        self.ff_interp  = initialize_flux_factors(n_rays)[m
[32m+[m
 #[m
 # three-component wrappers[m
 #[m
