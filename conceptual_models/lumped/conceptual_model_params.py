#!/usr/bin/env python

# Programmer(s): Vincent Adombi
# This file is part of the 'conceptual_models.lumped' package.

from hydroutils import Parameter


######################################################################

class ExphydroParameters(object):

    def __init__(self):
        """ Each parameter set contains a random realisation of all six
        EXP-HYDRO parameters as well as default values of Nash-Sutcliffe
        and Kling-Gupta efficiencies
        """

        self.f = Parameter(0, 0.1)
        self.smax = Parameter(100.0, 1500.0)
        self.qmax = Parameter(10.0, 50.0)
        self.ddf = Parameter(0.0, 5.0)
        self.mint = Parameter(-3.0, 0.0)
        self.maxt = Parameter(0.0, 3.0)

        self.objval = -9999  # This is the objective function value

    # ----------------------------------------------------------------

    def assignvalues(self, f, smax, qmax, ddf, mint, maxt):
        """ This method is used to manually assign parameter values,
        which are given by the user as input arguments.
        """

        self.f.value = f
        self.smax.value = smax
        self.qmax.value = qmax
        self.ddf.value = ddf
        self.mint.value = mint
        self.maxt.value = maxt

    # ----------------------------------------------------------------

    def updateparameters(self, param1, param2, w):
        """ This method is used for PSO algorithm.
            Each parameter in the model has to do the following
            two things:
            (1) Update its velocity
            (2) Update its value
        """

        # Update parameter velocities
        self.f.updatevelocity(param1.f, param2.f, w)
        self.ddf.updatevelocity(param1.ddf, param2.ddf, w)
        self.smax.updatevelocity(param1.smax, param2.smax, w)
        self.qmax.updatevelocity(param1.qmax, param2.qmax, w)
        self.mint.updatevelocity(param1.mint, param2.mint, w)
        self.maxt.updatevelocity(param1.maxt, param2.maxt, w)

        # Update parameter values
        self.f.updatevalue()
        self.ddf.updatevalue()
        self.smax.updatevalue()
        self.qmax.updatevalue()
        self.mint.updatevalue()
        self.maxt.updatevalue()


######################################################################

class ABCDParameters(object):

    def __init__(self):
        """ Each parameter set contains a random realisation of all
        ABCD parameters as well as default values of Nash-Sutcliffe
        and Kling-Gupta efficiencies
        """
        self.ddf = Parameter(0.0, 5.0)
        self.mint = Parameter(-3.0, 0.0)
        self.maxt = Parameter(0.0, 3.0)

        # ABCD params
        # https://doi.org/10.1029/2009WR008294
        self.a = Parameter(0.0, 1.0)
        self.b = Parameter(0.0, 4000.0)
        self.c = Parameter(0.0, 1.0)
        self.d = Parameter(0.0, 1.0)

        self.objval = -9999  # This is the objective function value

    # ----------------------------------------------------------------

    def assignvalues(self, ddf, mint, maxt, a, b, c, d):
        """ This method is used to manually assign parameter values,
        which are given by the user as input arguments.
        """
        self.ddf.value = ddf
        self.mint.value = mint
        self.maxt.value = maxt

        # ABCD params
        self.a.value = a
        self.b.value = b
        self.c.value = c
        self.d.value = d

    # ----------------------------------------------------------------

    def updateparameters(self, param1, param2, w):
        """ This method is used for PSO algorithm.
            Each parameter in the model has to do the following
            two things:
            (1) Update its velocity
            (2) Update its value
        """

        # Update parameter velocities
        # 1. Rainfall repartition and Snowmelt params
        self.ddf.updatevelocity(param1.ddf, param2.ddf, w)
        self.mint.updatevelocity(param1.mint, param2.mint, w)
        self.maxt.updatevelocity(param1.maxt, param2.maxt, w)
        # 2. ABCD params
        self.a.updatevelocity(param1.a, param2.a, w)
        self.b.updatevelocity(param1.b, param2.b, w)
        self.c.updatevelocity(param1.c, param2.c, w)
        self.d.updatevelocity(param1.d, param2.d, w)

        # Update parameter values
        # 1. Rainfall repartition and Snowmelt params
        self.ddf.updatevalue()
        self.mint.updatevalue()
        self.maxt.updatevalue()
        # 2. ABCD params
        self.a.updatevalue()
        self.b.updatevalue()
        self.c.updatevalue()
        self.d.updatevalue()


######################################################################

class HBVParameters(object):

    def __init__(self):
        """ Each parameter set contains a random realisation of all
        ABCD parameters as well as default values of Nash-Sutcliffe
        and Kling-Gupta efficiencies
        """
        self.ddf = Parameter(0.0, 5.0)
        self.mint = Parameter(-3.0, 0.0)
        self.maxt = Parameter(0.0, 3.0)

        # HBV params
        # https://doi.org/10.1002/2015WR018247
        self.fc = Parameter(50.0, 700.0)
        self.bt = Parameter(1.0, 6.0)
        self.lp = Parameter(0.3, 1.0)
        self.max_perc = Parameter(0.0, 6.0)
        self.uzl = Parameter(0.0, 100.0)
        self.k0 = Parameter(0.05, 0.99)
        self.k1 = Parameter(0.01, 0.8)
        self.k2 = Parameter(0.001, 0.15)

        self.objval = -9999  # This is the objective function value

    # ----------------------------------------------------------------

    def assignvalues(self, ddf, mint, maxt, fc, bt, lp, max_perc, uzl, k0, k1, k2):
        """ This method is used to manually assign parameter values,
        which are given by the user as input arguments.
        """
        self.ddf.value = ddf
        self.mint.value = mint
        self.maxt.value = maxt

        # HBV params
        self.fc.value = fc
        self.bt.value = bt
        self.lp.value = lp
        self.max_perc.value = max_perc
        self.uzl.value = uzl
        self.k0.value = k0
        self.k1.value = k1
        self.k2.value = k2

    # ----------------------------------------------------------------

    def updateparameters(self, param1, param2, w):
        """ This method is used for PSO algorithm.
            Each parameter in the model has to do the following
            two things:
            (1) Update its velocity
            (2) Update its value
        """

        # Update parameter velocities
        # 1. Rainfall repartition and Snowmelt params
        self.ddf.updatevelocity(param1.ddf, param2.ddf, w)
        self.mint.updatevelocity(param1.mint, param2.mint, w)
        self.maxt.updatevelocity(param1.maxt, param2.maxt, w)
        # 2. HBV params
        self.fc.updatevelocity(param1.fc, param2.fc, w)
        self.bt.updatevelocity(param1.bt, param2.bt, w)
        self.lp.updatevelocity(param1.lp, param2.lp, w)
        self.max_perc.updatevelocity(param1.max_perc, param2.max_perc, w)
        self.uzl.updatevelocity(param1.uzl, param2.uzl, w)
        self.k0.updatevelocity(param1.k0, param2.k0, w)
        self.k1.updatevelocity(param1.k1, param2.k1, w)
        self.k2.updatevelocity(param1.k2, param2.k2, w)

        # Update parameter values
        # 1. Rainfall repartition and Snowmelt params
        self.ddf.updatevalue()
        self.mint.updatevalue()
        self.maxt.updatevalue()
        # 2. HBV params
        self.fc.updatevalue()
        self.bt.updatevalue()
        self.lp.updatevalue()
        self.max_perc.updatevalue()
        self.uzl.updatevalue()
        self.k0.updatevalue()
        self.k1.updatevalue()
        self.k2.updatevalue()


######################################################################

class WBMParameters(object):

    def __init__(self):
        """ Each parameter set contains a random realisation of all
        WBM parameters as well as default values of Nash-Sutcliffe
        and Kling-Gupta efficiencies
        """
        self.ddf = Parameter(0.0, 5.0)
        self.mint = Parameter(-3.0, 0.0)
        self.maxt = Parameter(0.0, 3.0)

        # HBV params
        # https://doi.org/10.1016/j.quaint.2013.08.051
        self.smax = Parameter(10.0, 1000.0)  # inferred, no documentation specify this
        self.ks = Parameter(0.0, 1.0)
        self.kg = Parameter(0.001, 0.122)

        self.objval = -9999  # This is the objective function value

    # ----------------------------------------------------------------

    def assignvalues(self, ddf, mint, maxt, smax, ks, kg):
        """ This method is used to manually assign parameter values,
        which are given by the user as input arguments.
        """
        self.ddf.value = ddf
        self.mint.value = mint
        self.maxt.value = maxt

        # HBV params
        self.smax.value = smax
        self.ks.value = ks
        self.kg.value = kg

    # ----------------------------------------------------------------

    def updateparameters(self, param1, param2, w):
        """ This method is used for PSO algorithm.
            Each parameter in the model has to do the following
            two things:
            (1) Update its velocity
            (2) Update its value
        """

        # Update parameter velocities
        # 1. Rainfall repartition and Snowmelt params
        self.ddf.updatevelocity(param1.ddf, param2.ddf, w)
        self.mint.updatevelocity(param1.mint, param2.mint, w)
        self.maxt.updatevelocity(param1.maxt, param2.maxt, w)
        # 2. HBV params
        self.smax.updatevelocity(param1.smax, param2.smax, w)
        self.ks.updatevelocity(param1.ks, param2.ks, w)
        self.kg.updatevelocity(param1.kg, param2.kg, w)

        # Update parameter values
        # 1. Rainfall repartition and Snowmelt params
        self.ddf.updatevalue()
        self.mint.updatevalue()
        self.maxt.updatevalue()
        # 2. HBV params
        self.smax.updatevalue()
        self.ks.updatevalue()
        self.kg.updatevalue()


######################################################################

class FLEXParameters(object):

    def __init__(self):
        """ Each parameter set contains a random realisation of all
        WBM parameters as well as default values of Nash-Sutcliffe
        and Kling-Gupta efficiencies
        """
        self.ddf = Parameter(0.0, 5.0)
        self.mint = Parameter(-3.0, 0.0)
        self.maxt = Parameter(0.0, 3.0)

        # HBV params
        # https://hess.copernicus.org/articles/18/4839/2014/
        self.sfc = Parameter(0.0, 500.0)  # inferred, no documentation specify this
        self.bt = Parameter(0.0, 5.0)
        self.d = Parameter(0.0, 1.0)
        self.pmax = Parameter(0.0, 0.5)
        self.lp = Parameter(0.4, 0.6)
        self.kf = Parameter(0.0, 1.0)
        self.ks = Parameter(0.005, 0.05)

        self.objval = -9999  # This is the objective function value

    # ----------------------------------------------------------------

    def assignvalues(self, ddf, mint, maxt, sfc, bt, d, pmax, lp, kf, ks):
        """ This method is used to manually assign parameter values,
        which are given by the user as input arguments.
        """
        self.ddf.value = ddf
        self.mint.value = mint
        self.maxt.value = maxt

        # HBV params
        self.sfc.value = sfc
        self.bt.value = bt
        self.d.value = d
        self.pmax.value = pmax
        self.lp.value = lp
        self.kf.value = kf
        self.ks.value = ks

    # ----------------------------------------------------------------

    def updateparameters(self, param1, param2, w):
        """ This method is used for PSO algorithm.
            Each parameter in the model has to do the following
            two things:
            (1) Update its velocity
            (2) Update its value
        """

        # Update parameter velocities
        # 1. Rainfall repartition and Snowmelt params
        self.ddf.updatevelocity(param1.ddf, param2.ddf, w)
        self.mint.updatevelocity(param1.mint, param2.mint, w)
        self.maxt.updatevelocity(param1.maxt, param2.maxt, w)
        # 2. HBV params
        self.sfc.updatevelocity(param1.sfc, param2.sfc, w)
        self.bt.updatevelocity(param1.bt, param2.bt, w)
        self.d.updatevelocity(param1.d, param2.d, w)
        self.pmax.updatevelocity(param1.pmax, param2.pmax, w)
        self.lp.updatevelocity(param1.lp, param2.lp, w)
        self.kf.updatevelocity(param1.kf, param2.kf, w)
        self.ks.updatevelocity(param1.ks, param2.ks, w)

        # Update parameter values
        # 1. Rainfall repartition and Snowmelt params
        self.ddf.updatevalue()
        self.mint.updatevalue()
        self.maxt.updatevalue()
        # 2. HBV params
        self.sfc.updatevalue()
        self.bt.updatevalue()
        self.d.updatevalue()
        self.pmax.updatevalue()
        self.lp.updatevalue()
        self.kf.updatevalue()
        self.ks.updatevalue()
