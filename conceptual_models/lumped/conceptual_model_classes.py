#!/usr/bin/env python

# Programmer(s): Vincent Adombi
# This file is part of the 'conceptual_models.lumped' package.

import numpy
from hydroutils import OdeSolver


######################################################################

class ExphydroModel(object):
    """ An EXP-HYDRO bucket has its own climate inputs.

    It also has the properties of storage (both soil
    and snow), stream discharge (qsim), snowmelt (melt) and
    evapotranspiration (et)
    """

    def __init__(self, p, pet, t):

        """ This method is used to initialise, i.e., create an instance of the ExphydroModel class.

        Syntax: ExphydroModel(p, pet, t)

        Args:
            (1) p: Daily precipitation time-series (mm/day)
            (2) pet: Daily potential evapotranspiration time-series (mm/day)
            (3) t: Daily mean air temperature time-series (deg C)

        """

        # Below are the climate inputs
        self.P = p  # Daily precipitation (mm/day)
        self.PET = pet  # Daily PET (mm/day)
        self.T = t  # Daily mean air temperature (deg C)

        self.timespan = self.P.shape[0]  # Time length of the simulation period

        # Below are the state and flux variables of EXP-HYDRO
        #  All of them are initialised to zero
        self.storage = numpy.zeros(2)  # Storage of soil and snow buckets (mm)
        self.qsim = numpy.zeros(self.timespan)  # Simulated streamflow (mm/day)
        self.et = numpy.zeros(self.timespan)  # Simulated ET (mm/day)
        self.melt = numpy.zeros(self.timespan)  # Simulated snowmelt (mm/day)

    # ----------------------------------------------------------------

    def waterbalance(self, t, s, para):

        """ This method provides the right hand side of the dS/dt equations."""

        # EXP-HYDRO parameter values from object para
        f = para.f.value
        ddf = para.ddf.value
        smax = para.smax.value
        qmax = para.qmax.value
        mint = para.mint.value
        maxt = para.maxt.value

        # The line below ensures that the time step of input and output variables is always an integer.
        # ODE solvers can take fractional time steps, for which input data does not exist.
        tt = int(min(round(t), self.timespan - 1))

        # NOTE: The min condition in above line is very important and is needed when the ODE solver
        # jumps to a time-step that is beyond the time-series length.

        # Loading the input data for current time step
        p = self.P[tt]
        te = self.T[tt]
        pet = self.PET[tt]

        # Partitioning precipitation into rain and snow
        [ps, pr] = self.rainsnowpartition(p, te, mint)

        # Snow bucket
        m = self.snowbucket(s[0], te, ddf, maxt)

        # Soil bucket
        [et, qsub, qsurf] = self.soilbucket(s[1], pet, f, smax, qmax)

        # Water balance equations
        ds1 = ps - m
        ds2 = pr + m - et - qsub - qsurf

        ds = numpy.array([ds1, ds2])

        # Writing the flux calculations into output variables for the
        # current time step
        self.qsim[tt] = qsub + qsurf
        self.et[tt] = et
        self.melt[tt] = m

        return ds

    # ----------------------------------------------------------------

    @staticmethod
    def rainsnowpartition(p, t, mint):

        """ EXP-HYDRO equations to partition incoming precipitation
        into rain or snow."""

        if t < mint:
            psnow = p
            prain = 0
        else:
            psnow = 0
            prain = p

        return [psnow, prain]

    # ----------------------------------------------------------------

    @staticmethod
    def snowbucket(s, t, ddf, maxt):

        """ EXP-HYDRO equations for the snow bucket."""

        if t > maxt:
            if s > 0:
                melt = min(s, ddf * (t - maxt))
            else:
                melt = 0
        else:
            melt = 0

        return melt

    # ----------------------------------------------------------------

    @staticmethod
    def soilbucket(s, pet, f, smax, qmax):

        """ EXP-HYDRO equations for the soil bucket."""

        if s < 0:
            et = 0
            qsub = 0
            qsurf = 0
        elif s > smax:
            et = pet
            qsub = qmax
            qsurf = s - smax
        else:
            qsub = qmax * numpy.exp(-f * (smax - s))
            qsurf = 0
            et = pet * (s / smax)

        return [et, qsub, qsurf]

    # ----------------------------------------------------------------

    def simulate(self, para):

        """ This method performs the integration of dS/dt equations
        over the entire simulation time period
        """

        # Solving the ODE. To check which ODE solvers are available to use,
        # please check OdeSolver.py in hydroutils package
        OdeSolver.solve_rk4(self.waterbalance, self.storage, para, tlength=self.timespan)
        return self.qsim


######################################################################
class ABCDModel(object):
    """ An ABCD bucket has its own climate inputs.

    It also has the properties of storage (both soil
    and snow), stream discharge (qsim), snowmelt (melt) and
    evapotranspiration (et)
    """

    def __init__(self, p, pet, t):

        """ This method is used to initialise, i.e., create an instance of the ExphydroModel class.

        Syntax: ExphydroModel(p, pet, t)

        Args:
            (1) p: Daily precipitation time-series (mm/day)
            (2) pet: Daily potential evapotranspiration time-series (mm/day)
            (3) t: Daily mean air temperature time-series (deg C)

        """
        # Below are the climate inputs
        self.P = p  # Daily precipitation (mm/day)
        self.PET = pet  # Daily PET (mm/day)
        self.T = t  # Daily mean air temperature (deg C)

        self.timespan = self.P.shape[0]  # Time length of the simulation period

        # Below are the state and flux variables of EXP-HYDRO
        #  All of them are initialised to zero
        self.storage = numpy.zeros(3)  # Storage of snow, soil, and groundwater buckets (mm)
        self.melt = numpy.zeros(self.timespan)  # Simulated snowmelt (mm/day)
        self.ea = numpy.zeros(self.timespan)  # Simulated ET (mm/day)
        self.q_sim = numpy.zeros(self.timespan)  # Simulated streamflow (mm/day)
        self.rech = numpy.zeros(self.timespan)  # Simulated recharge (mm/day)

    # ----------------------------------------------------------------

    def waterbalance(self, t, s, para):

        """ This method provides the right hand side of the dS/dt equations."""

        # ABCD parameter values from object para
        ddf = para.ddf.value
        mint = para.mint.value  # min temperature threshold
        maxt = para.maxt.value  # max temperature threshold

        a = para.a.value
        b = para.b.value
        c = para.c.value
        d = para.d.value

        # The line below ensures that the time step of input and output variables is always an integer.
        # ODE solvers can take fractional time steps, for which input data does not exist.
        tt = int(min(round(t), self.timespan - 1))

        # NOTE: The min condition in above line is very important and is needed when the ODE solver
        # jumps to a time-step that is beyond the time-series length.

        # Loading the input data for current time step
        p = self.P[tt]
        te = self.T[tt]
        pet = self.PET[tt]

        # Partitioning precipitation into rain and snow
        [ps, pr] = self.rainsnowpartition(p, te, mint)

        # Snow bucket
        m = self.snowbucket(s[0], te, ddf, maxt)

        # Soil bucket
        [ea, q_sur, rech] = self.soilbucket(s[1], pr, m, pet, a, b, c)

        # Groundwater bucket
        q_sub = self.groundwater_bucket(s[2], d)

        # Water balance equations
        ds1 = ps - m
        ds2 = pr + m - ea - q_sur - rech
        ds3 = rech - q_sub

        ds = numpy.array([ds1, ds2, ds3])

        # Writing the flux calculations into output variables for the
        # current time step
        self.melt[tt] = m
        self.ea[tt] = ea
        self.q_sim[tt] = q_sur + q_sub
        self.rech = rech

        return ds

    # ----------------------------------------------------------------

    @staticmethod
    def rainsnowpartition(p, t, mint):

        """ EXP-HYDRO equations to partition incoming precipitation
        into rain or snow."""

        if t < mint:
            psnow = p
            prain = 0
        else:
            psnow = 0
            prain = p

        return [psnow, prain]

    # ----------------------------------------------------------------

    @staticmethod
    def snowbucket(s1, t, ddf, maxt):

        """ EXP-HYDRO equations for the snow bucket."""

        if t > maxt:
            if s1 > 0:
                melt = min(s1, ddf * (t - maxt))
            else:
                melt = 0
        else:
            melt = 0

        return melt

    # ----------------------------------------------------------------

    @staticmethod
    def soilbucket(s2, pr, m, pet, a, b, c):

        """ EXP-HYDRO equations for the soil bucket."""
        # compute process variables
        wa = pr + m + s2
        eo = (wa + b) / (2.0 * a) - numpy.sqrt(numpy.square((wa + b) / (2 * a)) - (wa * b) / a)

        ea = eo * (1.0 - numpy.exp(-pet / b))
        q_sur = (1 - c) * (wa - eo)
        rech = c * (wa - eo)
        return [ea, q_sur, rech]

    # ----------------------------------------------------------------

    @staticmethod
    def groundwater_bucket(s3, d):
        """
        Qb = d * S3
        :param s3: groundwater bucket storage
        :param d: subsurface discharge param
        :return: subsurface discharge
        """
        q_sub = d * s3
        return q_sub

    # ----------------------------------------------------------------

    def simulate(self, para):

        """ This method performs the integration of dS/dt equations
        over the entire simulation time period
        """

        # Solving the ODE. To check which ODE solvers are available to use,
        # please check OdeSolver.py in hydroutils package
        OdeSolver.solve_rk4(self.waterbalance, self.storage, para, tlength=self.timespan)
        return self.q_sim


######################################################################
class HBVModel(object):
    """ An HBV bucket has its own climate inputs.

    It also has the properties of storage (both soil
    and snow), stream discharge (qsim), snowmelt (melt) and
    evapotranspiration (et)
    """

    def __init__(self, p, pet, t):

        """ This method is used to initialise, i.e., create an instance of the ExphydroModel class.

        Syntax: ExphydroModel(p, pet, t)

        Args:
            (1) p: Daily precipitation time-series (mm/day)
            (2) pet: Daily potential evapotranspiration time-series (mm/day)
            (3) t: Daily mean air temperature time-series (deg C)

        """
        # Below are the climate inputs
        self.P = p  # Daily precipitation (mm/day)
        self.PET = pet  # Daily PET (mm/day)
        self.T = t  # Daily mean air temperature (deg C)

        self.timespan = self.P.shape[0]  # Time length of the simulation period

        # Below are the state and flux variables of EXP-HYDRO
        #  All of them are initialised to zero
        self.storage = numpy.zeros(4)  # Storage of snow, soil, quick and slow buckets (mm)
        self.melt = numpy.zeros(self.timespan)  # Simulated snowmelt (mm/day)
        self.ea = numpy.zeros(self.timespan)  # Simulated ET (mm/day)
        self.q_sim = numpy.zeros(self.timespan)  # Simulated streamflow (mm/day)
        self.rech = numpy.zeros(self.timespan)  # Simulated recharge (mm/day)

        # epsilon (numerical stability)
        self.epsilon = 1e-2

    # ----------------------------------------------------------------

    def waterbalance(self, t, s, para):

        """ This method provides the right hand side of the dS/dt equations."""

        # ABCD parameter values from object para
        ddf = para.ddf.value
        mint = para.mint.value  # min temperature threshold
        maxt = para.maxt.value  # max temperature threshold

        fc = para.fc.value
        bt = para.bt.value
        lp = para.lp.value
        max_perc = para.max_perc.value
        uzl = para.uzl.value
        k0 = para.k0.value
        k1 = para.k1.value
        k2 = para.k2.value

        # The line below ensures that the time step of input and output variables is always an integer.
        # ODE solvers can take fractional time steps, for which input data does not exist.
        tt = int(min(round(t), self.timespan - 1))

        # NOTE: The min condition in above line is very important and is needed when the ODE solver
        # jumps to a time-step that is beyond the time-series length.

        # Loading the input data for current time step
        p = self.P[tt]
        te = self.T[tt]
        pet = self.PET[tt]

        # Partitioning precipitation into rain and snow
        [ps, pr] = self.rainsnowpartition(p, te, mint)

        # Snow bucket
        m = self.snowbucket(s[0], te, ddf, maxt)

        # Soil bucket
        [q_in, qdr, rech, aet] = self.soilbucket(s[1], pr, m, pet, fc, bt, lp)

        # Groundwater bucket
        [perc, q0, q1, qbase] = self.quick_slow_bucket(s[2], s[3], max_perc, uzl, k0, k1, k2)

        # Water balance equations
        ds1 = ps - m
        ds2 = q_in - aet - qdr - rech
        ds3 = rech - perc - q0 - q1
        ds4 = perc - qbase

        ds = numpy.array([ds1, ds2, ds3, ds4])

        # Writing the flux calculations into output variables for the
        # current time step
        self.melt[tt] = m
        self.ea[tt] = aet
        self.q_sim[tt] = q0 + q1 + qbase
        self.rech = rech

        return ds

    # ----------------------------------------------------------------

    @staticmethod
    def rainsnowpartition(p, t, mint):

        """ EXP-HYDRO equations to partition incoming precipitation
        into rain or snow."""

        if t < mint:
            psnow = p
            prain = 0
        else:
            psnow = 0
            prain = p

        return [psnow, prain]

    # ----------------------------------------------------------------

    @staticmethod
    def snowbucket(s1, t, ddf, maxt):

        """ EXP-HYDRO equations for the snow bucket."""

        if t > maxt:
            if s1 > 0:
                melt = min(s1, ddf * (t - maxt))
            else:
                melt = 0
        else:
            melt = 0

        return melt

    # ----------------------------------------------------------------

    def soilbucket(self, s2, pr, m, pet, fc, bt, lp):

        """ EXP-HYDRO equations for the soil bucket."""
        # rain + melt
        q_in = pr + m
        # direct runoff
        qdr = numpy.maximum(s2 + q_in - fc, 0.0)
        # correct q_in
        q_in = q_in - qdr
        # compute recharge
        rech = q_in * (s2 / (fc + self.epsilon)) ** bt
        # compute actual evapotranspiration
        aet = pet * numpy.minimum(s2 / (lp * fc + self.epsilon), 1.0)
        return [q_in, qdr, rech, aet]

    # ----------------------------------------------------------------

    @staticmethod
    def quick_slow_bucket(s3, s4, max_perc, uzl, k0, k1, k2):
        # compute percolation
        perc = numpy.maximum(s3, max_perc)
        # rapid runoff components q0 and update su_curr
        q0 = k0 * numpy.maximum(s3 - uzl, 0.0)
        # rapid runoff components q1 and update su_curr
        q1 = k1 * s3
        # slow runoff component qbase
        qbase = k2 * s4
        return [perc, q0, q1, qbase]

    # ----------------------------------------------------------------

    def simulate(self, para):

        """ This method performs the integration of dS/dt equations
        over the entire simulation time period
        """

        # Solving the ODE. To check which ODE solvers are available to use,
        # please check OdeSolver.py in hydroutils package
        OdeSolver.solve_rk4(self.waterbalance, self.storage, para, tlength=self.timespan)
        return self.q_sim


######################################################################
class WBModel(object):
    """ An Water Balance Model (WBM) bucket has its own climate inputs.

    It also has the properties of storage (both soil
    and snow), stream discharge (qsim), snowmelt (melt) and
    evapotranspiration (et)
    """

    def __init__(self, p, pet, t):

        """ This method is used to initialise, i.e., create an instance of the ExphydroModel class.

        Syntax: ExphydroModel(p, pet, t)

        Args:
            (1) p: Daily precipitation time-series (mm/day)
            (2) pet: Daily potential evapotranspiration time-series (mm/day)
            (3) t: Daily mean air temperature time-series (deg C)

        """
        # Below are the climate inputs
        self.P = p  # Daily precipitation (mm/day)
        self.PET = pet  # Daily PET (mm/day)
        self.T = t  # Daily mean air temperature (deg C)

        self.timespan = self.P.shape[0]  # Time length of the simulation period

        # Below are the state and flux variables of EXP-HYDRO
        #  All of them are initialised to zero
        self.storage = numpy.zeros(2)  # Storage of snow, soil, quick and slow buckets (mm)
        self.melt = numpy.zeros(self.timespan)  # Simulated snowmelt (mm/day)
        self.ea = numpy.zeros(self.timespan)  # Simulated ET (mm/day)
        self.q_sim = numpy.zeros(self.timespan)  # Simulated streamflow (mm/day)

        # epsilon (numerical stability)
        self.epsilon = 1e-2

    # ----------------------------------------------------------------

    def waterbalance(self, t, s, para):

        """ This method provides the right hand side of the dS/dt equations."""

        # ABCD parameter values from object para
        ddf = para.ddf.value
        mint = para.mint.value  # min temperature threshold
        maxt = para.maxt.value  # max temperature threshold

        smax = para.smax.value
        ks = para.ks.value
        kg = para.kg.value

        # The line below ensures that the time step of input and output variables is always an integer.
        # ODE solvers can take fractional time steps, for which input data does not exist.
        tt = int(min(round(t), self.timespan - 1))

        # NOTE: The min condition in above line is very important and is needed when the ODE solver
        # jumps to a time-step that is beyond the time-series length.

        # Loading the input data for current time step
        p = self.P[tt]
        te = self.T[tt]
        pet = self.PET[tt]

        # Partitioning precipitation into rain and snow
        [ps, pr] = self.rainsnowpartition(p, te, mint)

        # Snow bucket
        m = self.snowbucket(s[0], te, ddf, maxt)

        # Soil bucket
        [et, qr, qb] = self.soilbucket(s[1], pr, m, pet, smax, ks, kg)

        # Water balance equations
        ds1 = ps - m
        ds2 = pr + m - et - qr - qb

        ds = numpy.array([ds1, ds2])

        # Writing the flux calculations into output variables for the
        # current time step
        self.melt[tt] = m
        self.ea[tt] = et
        self.q_sim[tt] = qr + qb

        return ds

    # ----------------------------------------------------------------

    @staticmethod
    def rainsnowpartition(p, t, mint):

        """ EXP-HYDRO equations to partition incoming precipitation
        into rain or snow."""

        if t < mint:
            psnow = p
            prain = 0
        else:
            psnow = 0
            prain = p

        return [psnow, prain]

    # ----------------------------------------------------------------

    @staticmethod
    def snowbucket(s1, t, ddf, maxt):

        """ EXP-HYDRO equations for the snow bucket."""

        if t > maxt:
            if s1 > 0:
                melt = min(s1, ddf * (t - maxt))
            else:
                melt = 0
        else:
            melt = 0

        return melt

    # ----------------------------------------------------------------

    def soilbucket(self, s2, pr, m, pet, smax, ks, kg):

        """ EXP-HYDRO equations for the soil bucket."""
        et = pet * s2 / (smax + self.epsilon)
        qr = ks * (pr + m) * s2 / (smax + self.epsilon)
        qb = kg * s2
        return [et, qr, qb]

    # ----------------------------------------------------------------

    def simulate(self, para):

        """ This method performs the integration of dS/dt equations
        over the entire simulation time period
        """

        # Solving the ODE. To check which ODE solvers are available to use,
        # please check OdeSolver.py in hydroutils package
        OdeSolver.solve_rk4(self.waterbalance, self.storage, para, tlength=self.timespan)
        return self.q_sim


######################################################################
class FLEXAModel(object):
    """ An FLEX_A bucket has its own climate inputs.

    It also has the properties of storage (both soil
    and snow), stream discharge (qsim), snowmelt (melt) and
    evapotranspiration (et)
    """

    def __init__(self, p, pet, t):

        """ This method is used to initialise, i.e., create an instance of the ExphydroModel class.

        Syntax: ExphydroModel(p, pet, t)

        Args:
            (1) p: Daily precipitation time-series (mm/day)
            (2) pet: Daily potential evapotranspiration time-series (mm/day)
            (3) t: Daily mean air temperature time-series (deg C)

        """
        # Below are the climate inputs
        self.P = p  # Daily precipitation (mm/day)
        self.PET = pet  # Daily PET (mm/day)
        self.T = t  # Daily mean air temperature (deg C)

        self.timespan = self.P.shape[0]  # Time length of the simulation period

        # Below are the state and flux variables of EXP-HYDRO
        #  All of them are initialised to zero
        self.storage = numpy.zeros(4)  # Storage of snow, soil, quick and slow buckets (mm)
        self.melt = numpy.zeros(self.timespan)  # Simulated snowmelt (mm/day)
        self.ea = numpy.zeros(self.timespan)  # Simulated ET (mm/day)
        self.q_sim = numpy.zeros(self.timespan)  # Simulated streamflow (mm/day)
        self.rech = numpy.zeros(self.timespan)

        # epsilon (numerical stability)
        self.epsilon = 1e-2

    # ----------------------------------------------------------------

    def waterbalance(self, t, s, para):
        """ This method provides the right hand side of the dS/dt equations.
        https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2006WR005563
        https://hess.copernicus.org/articles/10/139/2006/
        """

        # ABCD parameter values from object para
        ddf = para.ddf.value
        mint = para.mint.value  # min temperature threshold
        maxt = para.maxt.value  # max temperature threshold

        sfc = para.sfc.value
        bt = para.bt.value
        d = para.d.value
        pmax = para.pmax.value
        lp = para.lp.value
        kf = para.kf.value
        ks = para.ks.value

        # The line below ensures that the time step of input and output variables is always an integer.
        # ODE solvers can take fractional time steps, for which input data does not exist.
        tt = int(min(round(t), self.timespan - 1))

        # NOTE: The min condition in above line is very important and is needed when the ODE solver
        # jumps to a time-step that is beyond the time-series length.

        # Loading the input data for current time step
        p = self.P[tt]
        te = self.T[tt]
        pet = self.PET[tt]

        # Partitioning precipitation into rain and snow
        [ps, pr] = self.rainsnowpartition(p, te, mint)

        # Snow bucket
        m = self.snowbucket(s[0], te, ddf, maxt)

        # UR bucket
        [ea, re, ru, rf, ps, rs] = self.unsaturated_soil_reservoir(s[1], pr, m, pet, sfc, bt, d, pmax, lp)

        # FR bucket
        qf = self.fast_reservoir(s[2], kf)

        # SR bucket
        qs = self.fast_reservoir(s[3], ks)

        # Water balance equations
        ds1 = ps - m
        ds2 = ru - ea - ps # re = pr + m
        ds3 = rf - qf
        ds4 = ps + rs - qs

        ds = numpy.array([ds1, ds2, ds3, ds4])

        # Writing the flux calculations into output variables for the
        # current time step
        self.melt[tt] = m
        self.ea[tt] = ea
        self.q_sim[tt] = qf + qs
        self.rech[tt] = ps

        return ds

    # ----------------------------------------------------------------

    @staticmethod
    def rainsnowpartition(p, t, mint):

        """ EXP-HYDRO equations to partition incoming precipitation
        into rain or snow."""

        if t < mint:
            psnow = p
            prain = 0
        else:
            psnow = 0
            prain = p

        return [psnow, prain]

    # ----------------------------------------------------------------

    @staticmethod
    def snowbucket(s1, t, ddf, maxt):

        """ EXP-HYDRO equations for the snow bucket."""

        if t > maxt:
            if s1 > 0:
                melt = min(s1, ddf * (t - maxt))
            else:
                melt = 0
        else:
            melt = 0

        return melt

    # ----------------------------------------------------------------

    def unsaturated_soil_reservoir(self, s2, pr, m, pet, sfc, bt, d, pmax, lp):
        """ EXP-HYDRO equations for the soil bucket."""
        # re = pr + m
        re = pr + m
        # partition coefficient
        cr = 1.0 / (1.0 + numpy.exp((-s2 / (sfc + self.epsilon) + 0.5) / (bt + self.epsilon)))
        # ru (infiltration into soil)
        ru = (1.0 - cr) * re
        # preferential recharge
        rs = (re - ru) * d
        # runoff to FR (fast reservoir)
        rf = re - ru - rs
        # percolation
        ps = pmax * (s2 / (sfc + self.epsilon))
        # actual total evaporation
        ea = pet * numpy.minimum(1.0, s2 / (sfc * lp + self.epsilon))
        return [ea, re, ru, rf, ps, rs]

    # ----------------------------------------------------------------

    def fast_reservoir(self, s3, kf):
        qf = kf * s3
        return qf

    # ----------------------------------------------------------------

    def slow_reservoir(self, s4, ks):
        qs = ks * s4
        return qs

    # ----------------------------------------------------------------

    def simulate(self, para):

        """ This method performs the integration of dS/dt equations
        over the entire simulation time period
        """

        # Solving the ODE. To check which ODE solvers are available to use,
        # please check OdeSolver.py in hydroutils package
        OdeSolver.solve_rk4(self.waterbalance, self.storage, para, tlength=self.timespan)
        return self.q_sim
