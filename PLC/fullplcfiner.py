import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import builder
import cosmology
import params
from mpi4py import MPI
import os
from snapshot import Timeless_Snapshot
import ReadPinocchio as rp

# Bunch class for easy MPI handling
class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

# MPI comunicatior, rank and size of procs
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Check if the simplistic work balance will do
if params.nparticles%size:

    print("The data cannot be scattered on {} processes!".format(size))
    comm.Abort()

# Start loop on the snapshot files
for snapnum in range(params.numfiles):

   if rank == 0:

       print("Rank 0 is Working on the Geometry!")
       from geometry import geometry

       print("Rank 0 is reading the data!")
       if params.numfiles == 1:
          ts = Timeless_Snapshot(params.pintlessfile, -1, ready_to_bcast=True)
       else:
          ts = Timeless_Snapshot(params.pintlessfile+".{}".format(snapnum), -1, ready_to_bcast=True)
       print("All done! Let's go!")

       try:

           os.mkdir("Maps/")

       except FileExistsError:

           pass

   else:

      geometry = None

   geometry = comm.bcast(geometry)

   ###################################################

   if rank:
      # Defining ts to the other ranks
      ts   = Bunch(qPos = None, V1   = None, V2   = None, V31  = None, V32  = None, Zacc = None, Npart = None)

   npart = comm.bcast(ts.Npart)
   qPosslice = np.empty((npart//size,3), dtype = np.float32)
   V1slice   = np.empty((npart//size,3), dtype = np.float32)
   V2slice   = np.empty((npart//size,3), dtype = np.float32)
   V31slice  = np.empty((npart//size,3), dtype = np.float32)
   V32slice  = np.empty((npart//size,3), dtype = np.float32)
   Zaccslice = np.empty(npart//size, dtype = np.float32)
   aplcslice = np.empty(npart//size, dtype = np.float32)
   skycoordslice = np.empty((npart//size, 3), dtype = np.float32)

   comm.Scatterv(ts.qPos, qPosslice,root=0)
   comm.Scatterv(ts.V1  ,V1slice   ,root=0)
   comm.Scatterv(ts.V2  ,V2slice   ,root=0)
   comm.Scatterv(ts.V31 ,V31slice  ,root=0)
   comm.Scatterv(ts.V32 ,V32slice  ,root=0)
   comm.Scatterv(ts.Zacc ,Zaccslice,root=0)

   del ts

   if not rank:
      print("\n++++++++++++++++++++++\n")

   for i,(z1,z2) in enumerate( zip(cosmology.zlinf, cosmology.zlsup) ):

      zl = (z1 + z2)/2.0

      if not rank:

         # If working on this lens plane for the first time create an empy (zeros) delta map
         if not snapnum:
            deltai = np.zeros(params.npixels)
         # Else load it from disk
         else:
            print("Reopening density map:", 'Maps/delta_'+params.runflag+'_field_fullsky_{}.fits'.format(str(round(zl,4))))
            deltai = hp.read_map('Maps/delta_'+params.runflag+'_field_fullsky_{}.fits'.format(str(round(zl,4))))

      if not rank:
         print("Lens plane from z=[{0:.3f},{1:.3f}]".format(z1,z2))

      # Lens distances
      dlinf = cosmology.lcdm.comoving_distance(z1).to_value()
      dlsup = cosmology.lcdm.comoving_distance(z2).to_value()
      # Range in the scale factor compressed by dlinf and dlsup taking into account the buffer region
      amin  = 1.0/(1.0+cosmology.z_at_value(cosmology.lcdm.comoving_distance, dlsup*(1+params.beta_buffer)*cosmology.Mpc))
      if dlinf == 0.0:
          amax = 1.0
      else:
          amax  = 1.0/(1.0+cosmology.z_at_value(cosmology.lcdm.comoving_distance, dlinf*(1-params.beta_buffer)*cosmology.Mpc))

      # Fitting the cosmological functions inside the range
      # Select points inside the range
      auxcut  = (cosmology.a >= amin) & (cosmology.a <= amax)
      if not rank:
         print( "Number of points inside the redshift range is {} values!".format(np.sum(auxcut) ) )
      # If there is just one point inside the range use also the neighbours points
      if auxcut.sum() == 1:
          index = auxcut.argmax()
          if index < auxcut.size:
              auxcut[index + 1] = True
          if index > 0:
              auxcut[index - 1] = True
      # If there is no point inside the range use the neighbours points
      if auxcut.sum() == 0:
          index = ( (cosmology.a - (amax-amin)/2.0)**2 ).argmin()
          if index < auxcut.size:
              auxcut[index + 1] = True
          if index > 0:
              auxcut[index - 1] = True
      if not rank:
         print( "Interpolating the Cosmological functions using {} values!".format(np.sum(auxcut) ) )

      D       = cosmology.getWisePolyFit(cosmology.a[auxcut], cosmology.D[auxcut])
      D2      = cosmology.getWisePolyFit(cosmology.a[auxcut], cosmology.D2[auxcut])
      D31     = cosmology.getWisePolyFit(cosmology.a[auxcut], cosmology.D31[auxcut])
      D32     = cosmology.getWisePolyFit(cosmology.a[auxcut], cosmology.D32[auxcut])
      auxcut  = (cosmology.ainterp >= amin) & (cosmology.ainterp <= amax)
      DPLC    = cosmology.getWisePolyFit(cosmology.ainterp[auxcut], cosmology.Dinterp[auxcut]/params.boxsize)

      # Check which replications are compressed by the lens
      replicationsinside = geometry[ (geometry['nearestpoint'] < dlsup*(1+params.beta_buffer)) &
                                      (geometry['farthestpoint'] >= dlinf*(1-params.beta_buffer)) ]

      if not rank:
         print(" Replications inside:")
      # Loop on the replications
      for ii, repi in enumerate(replicationsinside):

         if not rank:
            print(" * Replication [{}/{}] of snap [{}/{}] {} {} {} "\
                       .format( str(ii + 1).zfill(int(np.log10(replicationsinside.size) + 1)), replicationsinside.size,
                                str(snapnum + 1).zfill(int(np.log10(params.numfiles) + 1)), params.numfiles,
                                repi['x'], repi['y'], repi['z']))

         # Position shift of the replication
         shift = np.array(repi[['x','y','z']].tolist()).dot(params.change_of_basis)
         # Get the scale parameter of the moment that the particle crossed the PLC
         builder.getCrossingScaleParameterNewtonRaphson (qPosslice + shift.astype(np.float32), V1slice, V2slice,\
                                                         V31slice, V32slice, aplcslice, npart//size, DPLC, D, D2,\
                                                         D31, D32, params.norder, amin, amax)


         # If the accretion redshift is hiher than the redshift crossing
         # ignore the particle
         aplcslice[ 1.0/aplcslice -1  < Zaccslice ] = -1.0
         builder.getSkyCoordinates(qPosslice, shift.astype(np.float32), V1slice, V2slice, V31slice, V32slice, aplcslice,\
                                                                 skycoordslice,npart//size, D, D2, D31, D32, params.norder)

         # Collect data from the other ranks
         for ranki in range(size):

            if (rank == ranki) and ranki > 0:

               comm.Send(skycoordslice, dest=0)

            if rank == 0:

               if ranki:
                  print(" Rank: 0 receving slice from Rank: {}".format(ranki))
                  comm.Recv(skycoordslice, source=ranki)
               else:
                  print(" Rank: 0 working on its own load".format(ranki))

               cut = skycoordslice[:,0] > 0
               theta, phi = skycoordslice[:,1][cut] + np.pi/2.0, skycoordslice[:,2][cut]
               pixels = hp.pixelfunc.ang2pix(hp.pixelfunc.npix2nside(params.npixels), theta, phi)
               # Rank 0 update the map
               deltai += np.histogram(pixels, bins=np.linspace(0,params.npixels,params.npixels+1).astype(int))[0]

            comm.Barrier()

      if rank == 0:
         # Rank 0 writes the collected map
         hp.fitsfunc.write_map('Maps/delta_'+params.runflag+'_field_fullsky_{}.fits'.format(str(round(zl,4))), deltai, overwrite=True)

         print("\n++++++++++++++++++++++\n")

      comm.Barrier()

# Everything done for the particles
# Constructs the density maps for halos
# and convergence maps for particles
if not rank:

   if params.fovindeg < 180.0:

       pixels = np.linspace(0,params.npixels,params.npixels+1).astype(int)
       mask   = hp.pix2ang( hp.pixelfunc.npix2nside(params.npixels), pixels)[:,0]
       mask   = mask > params.fovinradians

   else:

       mask   = np.linspace(0,params.npixels,params.npixels+1).astype(bool)

   kappa = np.zeros(params.npixels)

   for z1, z2 in zip(cosmology.zlinf, cosmology.zlsup):

      zl = 1.0/2.0*(z1+z2)

      deltahi = np.zeros(params.npixels)

      if params.numfiles > 1:

         for snapnum in range(params.numfiles):

            plc      = rp.plc(params.pinplcfile+".{}".format(snapnum))
            plc.Mass = (plc.Mass/plc.Mass.min()*params.minhalomass).astype(int)
            groupsinplane = (plc.redshift <= z2) & (plc.redshift > z1)
            pixels = hp.pixelfunc.ang2pix(hp.pixelfunc.npix2nside(params.npixels), \
                 plc.theta[groupsinplane]*np.pi/180.0+np.pi/2.0, plc.phi[groupsinplane]*np.pi/180.0)
            deltahi += np.histogram(pixels, bins=np.linspace(0,params.npixels,params.npixels+1).astype(int))[0]

      else:

         plc      = rp.plc(params.pinplcfile)
         plc.Mass = (plc.Mass/plc.Mass.min()*params.minhalomass).astype(int)
         groupsinplane = (plc.redshift <= z2) & (plc.redshift > z1)
         pixels = hp.pixelfunc.ang2pix(hp.pixelfunc.npix2nside(params.npixels), \
         plc.theta[groupsinplane]*np.pi/180.0+np.pi/2.0, plc.phi[groupsinplane]*np.pi/180.0)
         deltahi += np.histogram(pixels, bins=np.linspace(0,params.npixels,params.npixels+1).astype(int))[0]

      deltahi[mask]  = hp.UNSEEN
      deltahi[~mask] = deltahi[~mask]/deltahi[~mask].mean() - 1.0
      hp.fitsfunc.write_map('Maps/delta_'+params.runflag+'_halos_fullsky_{}.fits'.format(str(round(zl,4))), deltahi, overwrite=True)

      deltai = hp.fitsfunc.read_map('Maps/delta_'+params.runflag+'_field_fullsky_{}.fits'.format(str(round(zl,4))))

      deltai[mask]  = hp.UNSEEN
      deltai[~mask] = deltai[~mask]/deltai[~mask].mean() - 1.0
      hp.fitsfunc.write_map('Maps/delta_'+params.runflag+'_field_fullsky_{}.fits'.format(str(round(zl,4))), deltai, overwrite=True)

      kappai = (1.0+zl) * ( ( 1.0 - cosmology.lcdm.comoving_distance(zl)/cosmology.lcdm.comoving_distance(params.zsource) ) *\
                  cosmology.lcdm.comoving_distance(zl) *\
                ( cosmology.lcdm.comoving_distance(z2) - cosmology.lcdm.comoving_distance(z1) ) ).to_value() * deltai
      kappai *= (3.0 * cosmology.lcdm.Om0*cosmology.lcdm.H0**2/2.0/cosmology.cspeed**2).to_value()
      kappa += kappai
      hp.fitsfunc.write_map('Maps/kappa_'+params.runflag+'_field_fullsky_{}.fits'.format(str(round(zl,4))), kappai, overwrite=True)

   kappa[mask] = hp.UNSEEN
   hp.mollview(kappa)
   plt.show()

   cl = hp.anafast(kappa, lmax=512)
   ell = np.arange(len(cl))
   np.savetxt("Maps/Cls_kappa_z{}.txt".format(params.zsource), np.transpose([ell, cl, ell * (ell+1) * cl]))
