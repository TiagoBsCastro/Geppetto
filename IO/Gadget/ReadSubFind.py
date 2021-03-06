"""

Routines to read SubFind catalogs from GADGET

This library recognizes three formats for the FoF catalogs: old format, GADGET3 format and snapshot format.
The snapshot format should be read with the Snapshot.py library.
It checks the endianness of the files and automatically swaps the file if needed

Basic use:

  import ReadSubFind
  sub = ReadSubFind.catalog([base directory],[snapshot number],
                            long_IDs=[False(def),True],
                            SaveMassTab=[False,True(def)],
                            SOVelDisp=[False(def),True],
                            ConTermination=[False(def),True],
                            verbose=[0,1])

base directory: the directory that contains the output files of the simulation 
(or of the postprocessing if FoF has not been run on the fly)
snapshot number: can be an integer or a string
long_IDs (optional): particle IDS are uint64 (default: False)
SaveMassTab (optional): the field SubMassTab is present (default:True)
SOVelDisp (optional): the field relative to the SO velocity dispersion are present (default:False)
ConTermination (optional): the field HaloCont is present (default:False)
verbose (optional): 0 to keep it quiet

To read IDs for particles in halos:

  sub.read_IDs(verbose=[0,1])

To know what the object fof contains:

  sub.help()

Example:
  >>> sub=ReadSubFind.catalog("",100)
  >>> sub.read_IDs(True)
  >>> sub.help()


2016, written by Pierluigi Monaco (on the basis of older code)

"""

import numpy as np
import os
from . import Snapshot


def myswap(a,flag):
    if flag:
        return a.byteswap()
    else:
        return a


def guess_format(basedir,snapnum,verbose=0):

    if type(snapnum) is int:
        snapnum="%03d"%snapnum

    if (basedir!="" and basedir[-1]!="/"):
        basedir+="/"

    fname=basedir+"postproc_"+snapnum+"/sub_tab_"+snapnum+".0"
    if os.path.exists(fname):
        if verbose>1:
            print("FOUND",fname)

        fname=basedir+"postproc_"+snapnum+"/sub_tab_"+snapnum
        myformat=0
    else:
        if verbose>1:
            print(fname,"not found")

        fname=basedir+"groups_"+snapnum+"/subhalo_tab_"+snapnum+".0"
        if os.path.exists(fname):
            if verbose>1:
                print("FOUND",fname)

            fname=basedir+"groups_"+snapnum+"/subhalo_tab_"+snapnum
            myformat=1
        else:
            if verbose>1:
                print(fname,"not found")

            fname=basedir+"groups_"+snapnum+"/sub_"+snapnum+".0"
            if os.path.exists(fname):
                if verbose>1:
                    print("FOUND",fname)

                fname=basedir+"groups_"+snapnum+"/sub_"+snapnum
                myformat=2
            else:
                if verbose>1:
                    print(fname,"not found")

                print("ERROR: Subfind files not found")
                return None


    # checks how many files are found for the subfind catalog
    nfiles=-1
    exists=True
    while exists:
        nfiles+=1
        exists=os.path.exists(fname+".%d"%nfiles)

    if verbose>0:
        print("Subfind catalog found with format %d in %d files"%(myformat,nfiles))

    # reads the header from file N. 0
    if myformat==0:
        test=-1
        exists=True
        while exists:
            test+=1
            exists=os.path.exists(basedir+"groups_"+snapnum+"/group_tab_"+snapnum+".%d"%test)
        if verbose>1:
            print("FOF catalog was written in",test,"files")

        f=open(fname+".0","rb")
        f.seek(12,os.SEEK_SET)
        number=(np.fromfile(f,dtype=np.uint32,count=1))[0]

        # in the old format this should be =nfiles
        if number==test:
            if verbose>0:
                print("This is old format with native endianness")

            f.close()
            return (nfiles, 0, False)

        elif number.byteswap()==nfiles:
            if verbose>0:
                print("This is old format with inverted endianness")

            f.close()
            return (nfiles, 0, True)

        else:
            print("ERROR: I do not understand the format of file "+fname+".0")
            f.close()
            return None
        
    elif myformat==1:
        test=-1
        exists=True
        while exists:
            test+=1
            exists=os.path.exists(basedir+"groups_"+snapnum+"/group_tab_"+snapnum+".%d"%test)
        if verbose>1:
            print("FOF catalog was written in",test,"files")

        f=open(fname+".0","rb")
        f.seek(20,os.SEEK_SET)
        number=(np.fromfile(f,dtype=np.uint32,count=1))[0]

        # in the G3 format this should be =nfiles
        if number==nfiles:
            if verbose>0:
                print("This is G3 format with native endianness")

            f.close()
            return (nfiles, 1, False)

        elif number.byteswap()==nfiles:
            if verbose>0:
                print("This is G3 format with inverted endianness")

            f.close()
            return (nfiles, 1, True)

        else:
            print("ERROR: I do not understand the format of file "+fname+".0")
            f.close()
            return None

    else:
        snap=Snapshot.Init(basedir+"groups_"+snapnum+"/sub",snapnum)
        if snap.format==-99:
            print("ERROR: I do not understand the format of file "+fname+".0")
            return None
        else:
            return (nfiles, 2, snap.swap)


class catalog:

    """

    This class defines a SubFind catalog read from a simulation.
    For more details, use the help() method applied to the object
    
    Example:
    >>> sub=ReadSubFind.catalog("",100)
    >>> sub.read_IDs(True)
    >>> sub.help()

    """


    def __init__(self,basedir,snapnum,long_IDs=False,SaveMassTab=True,
                 SOVelDisp=False,ConTermination=False,verbose=0):

        # first it guesses which is the format
        if type(snapnum) is int:
            snapnum="%03d"%snapnum

        gform=guess_format(basedir,snapnum)

        # this can be improved by reading the snapshot
        if gform[1]==2:
            (nfiles,self.myformat,self.swap) = gform
            print("Please read this SubFind catalog as a snapshot")
            return None

        if gform==None:
            print("Error in reading SubFind catalogs")
            return None

        self.snapnum=snapnum
        self.basedir=basedir

        (nfiles,myformat,self.swap) = gform
        self.Nfiles=nfiles
        self.myformat=myformat
        self.long_IDs=long_IDs

        if self.long_IDs:
            self.id_format=np.uint64
        else: 
            self.id_format=np.uint32

        if myformat==0:
            fname=basedir+"postproc_"+snapnum+"/sub_tab_"+snapnum
        elif myformat==1:
            fname=basedir+"groups_"+snapnum+"/subhalo_tab_"+snapnum
        else:
            fname=basedir+"groups_"+snapnum+"/sub_"+snapnum

        #################  READ TAB FILES ################# 
        fnb,skipS,skipG=0,0,0
        Final=False
        if myformat==0:
            self.TotNids=long(0)
            self.TotNsubhalos=0
        while not(Final):
            f=open(fname+".%d"%fnb,'rb')

            if myformat==1:
                (Ngroups,TotNgroups,Nids)=myswap(np.fromfile(f,dtype=np.int32,count=3),self.swap)
                (TotNids)=myswap(np.fromfile(f,dtype=np.uint64,count=1),self.swap)[0]
                (Nfiles,Nsubhalos,TotNsubhalos)=myswap(np.fromfile(f,dtype=np.uint32,count=3),self.swap)
            else:
                (Ngroups,Nids,TotNgroups,Nfiles,Nsubhalos)=myswap(np.fromfile(f,dtype=np.int32,count=5),self.swap)

            if self.swap:
                Ngroups.byteswap()
                TotNgroups.byteswap()
                Nids.byteswap()
                Nfiles.byteswap()
                Nsubhalos.byteswap()
                if myformat==1:
                    TotNids.byteswap()    
                    TotNsubhalos.byteswap()

            if myformat==0:
                self.TotNids+=Nids
                self.TotNsubhalos+=Nsubhalos

            if fnb==0:
                self.TotNgroups=TotNgroups
                if myformat==1:
                    self.TotNids=TotNids
                    self.TotNsubhalos=TotNsubhalos

            if Nfiles != nfiles:
                print("WARNING: inconsistency, ",nfiles," files found but the header gives",Nfiles)

            if verbose>0:
                print()
                print("File N. ",fnb,":")
                print("Ngroups = ",Ngroups)
                print("TotNgroups = ",TotNgroups)
                print("Nids = ",Nids)
                print("TotNids = ",self.TotNids)
                print("Nfiles = ",Nfiles)
                print("Nsubhalos = ",Nsubhalos)
                print("TotNsubhalos = ",self.TotNsubhalos)


            # allocations
            if fnb == 0:
                self.NsubPerHalo      = np.empty(self.TotNgroups  ,dtype=np.int32)
                self.FirstSubOfHalo   = np.empty(self.TotNgroups  ,dtype=np.int32)
                self.SubLen           = np.empty(self.TotNsubhalos,dtype=np.int32)
                self.SubOffset        = np.empty(self.TotNsubhalos,dtype=np.int32)
                self.SubParentHalo    = np.empty(self.TotNsubhalos,dtype=np.int32)
                self.Halo_M_Mean200   = np.empty(self.TotNgroups  ,dtype=np.float32)
                self.Halo_R_Mean200   = np.empty(self.TotNgroups  ,dtype=np.float32)
                self.Halo_M_Crit200   = np.empty(self.TotNgroups  ,dtype=np.float32)
                self.Halo_R_Crit200   = np.empty(self.TotNgroups  ,dtype=np.float32)
                self.Halo_M_TopHat200 = np.empty(self.TotNgroups  ,dtype=np.float32)
                self.Halo_R_TopHat200 = np.empty(self.TotNgroups  ,dtype=np.float32)
                self.SubPos           = np.empty(self.TotNsubhalos,dtype=np.dtype((np.float32,3)))
                self.SubVel           = np.empty(self.TotNsubhalos,dtype=np.dtype((np.float32,3)))
                self.SubVelDisp       = np.empty(self.TotNsubhalos,dtype=np.float32)
                self.SubVmax          = np.empty(self.TotNsubhalos,dtype=np.float32)
                self.SubSpin          = np.empty(self.TotNsubhalos,dtype=np.dtype((np.float32,3)))
                self.SubMostBoundID   = np.empty(self.TotNsubhalos,dtype=self.id_format)
                self.SubHalfMass      = np.empty(self.TotNsubhalos,dtype=np.float32)
                if SaveMassTab:
                    self.SubMassTab   = np.empty(self.TotNsubhalos,dtype=np.dtype((np.float32,6)))
                if ConTermination:
                    self.HaloCont     = np.empty(self.TotNgroups  ,dtype=np.float32)
                if myformat==1:
                    self.HaloLen      = np.empty(self.TotNgroups  ,dtype=np.int32)
                    self.HaloMemberID = np.empty(self.TotNgroups  ,dtype=np.int32)
                    self.HaloMass     = np.empty(self.TotNgroups  ,dtype=np.float32)
                    self.HaloPos      = np.empty(self.TotNgroups  ,dtype=np.dtype((np.float32,3)))
                    if SOVelDisp:
                        self.VelDisp_Mean200   = np.empty(self.TotNgroups  ,dtype=np.float32)
                        self.VelDisp_Crit200   = np.empty(self.TotNgroups  ,dtype=np.float32)
                        self.VelDisp_TopHat200 = np.empty(self.TotNgroups  ,dtype=np.float32)
                    self.HaloContCount  = np.empty(self.TotNgroups  ,dtype=np.float32)
                    self.SubTMass       = np.empty(self.TotNsubhalos,dtype=np.float32)
                    self.SubCM          = np.empty(self.TotNsubhalos,dtype=np.dtype((np.float32,3)))
                    self.SubVmaxRad     = np.empty(self.TotNsubhalos,dtype=np.float32)
                    self.SubGroupNumber = np.empty(self.TotNsubhalos,dtype=np.int32)

            if Ngroups>0:
                locG=slice(skipG,skipG+Ngroups)
                locS=slice(skipS,skipS+Nsubhalos)

                if myformat==0:
                    self.NsubPerHalo[locG]      = myswap(np.fromfile(f, dtype=np.int32,                count=self.TotNgroups  ),self.swap)
                    self.FirstSubOfHalo[locG]   = myswap(np.fromfile(f, dtype=np.int32,                count=self.TotNgroups  ),self.swap)
                    self.SubLen[locS]           = myswap(np.fromfile(f, dtype=np.int32,                count=self.TotNsubhalos),self.swap)
                    self.SubOffset[locS]        = myswap(np.fromfile(f, dtype=np.int32,                count=self.TotNsubhalos),self.swap)
                    self.SubParentHalo[locS]    = myswap(np.fromfile(f, dtype=np.int32,                count=self.TotNsubhalos),self.swap)
                    self.Halo_M_Mean200[locG]   = myswap(np.fromfile(f, dtype=np.float32,              count=self.TotNgroups  ),self.swap)
                    self.Halo_R_Mean200[locG]   = myswap(np.fromfile(f, dtype=np.float32,              count=self.TotNgroups  ),self.swap)
                    self.Halo_M_Crit200[locG]   = myswap(np.fromfile(f, dtype=np.float32,              count=self.TotNgroups  ),self.swap)
                    self.Halo_R_Crit200[locG]   = myswap(np.fromfile(f, dtype=np.float32,              count=self.TotNgroups  ),self.swap)
                    self.Halo_M_TopHat200[locG] = myswap(np.fromfile(f, dtype=np.float32,              count=self.TotNgroups  ),self.swap)
                    self.Halo_R_TopHat200[locG] = myswap(np.fromfile(f, dtype=np.float32,              count=self.TotNgroups  ),self.swap)
                    self.SubPos[locS]           = myswap(np.fromfile(f, dtype=np.dtype((np.float32,3)),count=self.TotNsubhalos),self.swap)
                    self.SubVel[locS]           = myswap(np.fromfile(f, dtype=np.dtype((np.float32,3)),count=self.TotNsubhalos),self.swap)
                    self.SubVelDisp[locS]       = myswap(np.fromfile(f, dtype=np.float32,              count=self.TotNsubhalos),self.swap)
                    self.SubVmax[locS]          = myswap(np.fromfile(f, dtype=np.float32,              count=self.TotNsubhalos),self.swap)
                    self.SubSpin[locS]          = myswap(np.fromfile(f, dtype=np.dtype((np.float32,3)),count=self.TotNsubhalos),self.swap)
                    self.SubMostBoundID[locS]   = myswap(np.fromfile(f, dtype=self.id_format,          count=self.TotNsubhalos),self.swap)
                    self.SubHalfMass[locS]      = myswap(np.fromfile(f, dtype=np.float32,              count=self.TotNsubhalos),self.swap)
                    if SaveMassTab:
                        self.SubMassTab[locS]   = myswap(np.fromfile(f, dtype=np.dtype((np.float32,6)),count=self.TotNsubhalos),self.swap)
                    if ConTermination:
                        self.HaloCont[locG]     = myswap(np.fromfile(f, dtype=np.float32,              count=self.TotNgroups  ),self.swap)

                else:
                    self.HaloLen[locG]          = myswap(np.fromfile(f, dtype=np.int32,                count=self.TotNgroups  ),self.swap)
                    self.HaloMemberID[locG]     = myswap(np.fromfile(f, dtype=np.int32,                count=self.TotNgroups  ),self.swap)
                    self.HaloMass[locG]         = myswap(np.fromfile(f, dtype=np.float32,              count=self.TotNgroups  ),self.swap)
                    self.HaloPos[locG]          = myswap(np.fromfile(f, dtype=np.dtype((np.float32,3)),count=self.TotNgroups  ),self.swap)
                    self.Halo_M_Mean200[locG]   = myswap(np.fromfile(f, dtype=np.float32,              count=self.TotNgroups  ),self.swap)
                    self.Halo_R_Mean200[locG]   = myswap(np.fromfile(f, dtype=np.float32,              count=self.TotNgroups  ),self.swap)
                    self.Halo_M_Crit200[locG]   = myswap(np.fromfile(f, dtype=np.float32,              count=self.TotNgroups  ),self.swap)
                    self.Halo_R_Crit200[locG]   = myswap(np.fromfile(f, dtype=np.float32,              count=self.TotNgroups  ),self.swap)
                    self.Halo_M_TopHat200[locG] = myswap(np.fromfile(f, dtype=np.float32,              count=self.TotNgroups  ),self.swap)
                    self.Halo_R_TopHat200[locG] = myswap(np.fromfile(f, dtype=np.float32,              count=self.TotNgroups  ),self.swap)
                    if SOVelDisp:
                        self.VelDisp_Mean200[locG] = myswap(np.fromfile(f, dtype=np.float32,           count=self.TotNgroups  ),self.swap)
                        self.VelDisp_Crit200[locG] = myswap(np.fromfile(f, dtype=np.float32,           count=self.TotNgroups  ),self.swap)
                        self.VelDisp_TopHat200[locG] = myswap(np.fromfile(f, dtype=np.float32,         count=self.TotNgroups  ),self.swap)
                    self.HaloContCount[locG]    = myswap(np.fromfile(f, dtype=np.float32,              count=self.TotNgroups  ),self.swap)
                    if ConTermination:
                        self.HaloCont[locG]     = myswap(np.fromfile(f, dtype=np.float32,              count=self.TotNgroups  ),self.swap)
                    self.NsubPerHalo[locG]      = myswap(np.fromfile(f, dtype=np.int32,                count=self.TotNgroups  ),self.swap)
                    self.FirstSubOfHalo[locG]   = myswap(np.fromfile(f, dtype=np.int32,                count=self.TotNgroups  ),self.swap)
                    self.SubLen[locS]           = myswap(np.fromfile(f, dtype=np.int32,                count=self.TotNsubhalos),self.swap)
                    self.SubOffset[locS]        = myswap(np.fromfile(f, dtype=np.int32,                count=self.TotNsubhalos),self.swap)
                    self.SubParentHalo[locS]    = myswap(np.fromfile(f, dtype=np.int32,                count=self.TotNsubhalos),self.swap)
                    self.SubTMass[locS]         = myswap(np.fromfile(f, dtype=np.float32,              count=self.TotNsubhalos),self.swap)
                    self.SubPos[locS]           = myswap(np.fromfile(f, dtype=np.dtype((np.float32,3)),count=self.TotNsubhalos),self.swap)
                    self.SubVel[locS]           = myswap(np.fromfile(f, dtype=np.dtype((np.float32,3)),count=self.TotNsubhalos),self.swap)
                    self.SubCM[locS]            = myswap(np.fromfile(f, dtype=np.dtype((np.float32,3)),count=self.TotNsubhalos),self.swap)
                    self.SubSpin[locS]          = myswap(np.fromfile(f, dtype=np.dtype((np.float32,3)),count=self.TotNsubhalos),self.swap)
                    self.SubVelDisp[locS]       = myswap(np.fromfile(f, dtype=np.float32,              count=self.TotNsubhalos),self.swap)
                    self.SubVmax[locS]          = myswap(np.fromfile(f, dtype=np.float32,              count=self.TotNsubhalos),self.swap)
                    self.SubVmaxRad[locS]       = myswap(np.fromfile(f, dtype=np.float32,              count=self.TotNsubhalos),self.swap)
                    self.SubHalfMass[locS]      = myswap(np.fromfile(f, dtype=np.float32,              count=self.TotNsubhalos),self.swap)
                    self.SubMostBoundID[locS]   = myswap(np.fromfile(f, dtype=self.id_format,          count=self.TotNsubhalos),self.swap)
                    self.SubGroupNumber[locS]   = myswap(np.fromfile(f, dtype=np.int32,                count=self.TotNsubhalos),self.swap)
                    if SaveMassTab:
                        self.SubMassTab[locS]   = myswap(np.fromfile(f, dtype=np.dtype((np.float32,6)),count=self.TotNsubhalos),self.swap)


            skipG+=Ngroups
            skipS+=Nsubhalos

            curpos = f.tell()
            f.seek(0,os.SEEK_END)
            if curpos != f.tell():
                print("Warning: the file is not finished",fnb)
            f.close()
            fnb+=1
            if fnb==self.Nfiles: Final=True




    def read_IDs(self,verbose=0):


        #################  READ IDS FILES ################# 
        if self.myformat==0:
            fname=self.basedir+"postproc_"+self.snapnum+"/sub_ids_"+self.snapnum+"."
        elif self.myformat==1:
            fname=self.basedir+"groups_"+self.snapnum+"/subhalo_ids_"+self.snapnum+"."
        fnb,skip=0,0
        Final=False
        while not(Final):
            f=open(fname+str(fnb),'rb')

            if self.myformat==1:
                (Ngroups,TotNgroups,Nids)=myswap(np.fromfile(f,dtype=np.int32,count=3))
                (TotNids)=myswap(np.fromfile(f,dtype=np.uint64,count=1))
                (Nfiles,Nsubhalos,TotNsubhalos)=myswap(np.fromfile(f,dtype=np.uint32,count=3))
            else:
                (Ngroups,Nids,TotNgroups,Nfiles)=myswap(np.fromfile(f,dtype=np.int32,count=4))
                TotNids=self.TotNids
                TotNsubhalos=self.TotNsubhalos

            if self.swap:
                Ngroups.byteswap()
                TotNgroups.byteswap()
                Nids.byteswap()
                Nfiles.byteswap()
                Nsubhalos.byteswap()
                if myformat==1:
                    TotNids.byteswap()    
                    TotNsubhalos.byteswap()


            if TotNgroups != self.TotNgroups:
                print("ERROR: inconsistency in TotNgroups, ",TotNgroups,self.TotNgroups)
                return None

            if TotNids != self.TotNids:
                print("ERROR: inconsistency in TotNgroups, ",TotNgroups,self.TotNgroups)
                return None

            if verbose>0:
                print()
                print("File N. ",fnb,":")
                print("Ngroups = ",Ngroups)
                print("TotNgroups = ",TotNgroups)
                print("Nids = ",Nids)
                print("TotNids = ",self.TotNids)
                print("Nfiles = ",Nfiles)

            if fnb==0:
                self.GroupIDs=np.zeros(dtype=self.id_format,shape=self.TotNids)

            if Ngroups>0:
                if self.long_IDs:
                    IDs=myswap(np.fromfile(f,dtype=np.uint64,count=Nids))
                else:
                    IDs=myswap(np.fromfile(f,dtype=np.uint32,count=Nids))
                if self.swap:
                    IDs=IDs.byteswap(True)

                self.GroupIDs[skip:skip+Nids]=IDs[:]
                skip+=Nids
                del IDs

            curpos = f.tell()
            f.seek(0,os.SEEK_END)
            if curpos != f.tell():
                print("Warning: finished reading before EOF for IDs file",fnb)
            f.close()
            fnb+=1
            if fnb==self.Nfiles:
                Final=True



    def help(self):
        
        if self.myformat==2:
            print("Please treat this SubFind catalog as a snapshot")
            return None

        print("Quantities contained in the ReadSubFind catalog structure:")
        print("Control parameters")
        print("  snapnum:      ",self.snapnum)
        print("  basedir:      ",self.basedir)
        print("  swap:         ",self.swap)
        print("  myformat:     ",self.myformat)
        print("  Nfiles:       ",self.Nfiles)
        print("  id_format:    ",self.id_format)
        print("From the header")
        print("  TotNgroups:   ",self.TotNgroups)
        print("  TotNsubhalos: ",self.TotNsubhalos)
        print("  TotNids:      ",self.TotNids)
        print("Vectors (length TotNgroups)")


        if self.myformat==1:
            print("  HaloLen            (dtype=np.int32)")
            print("  HaloMemberID       (dtype=np.int32)")
            print("  HaloMass           (dtype=np.float32)")
            print("  HaloPos            (dtype=np.dtype((np.float32,3)))")
            print("  HaloContCount      (dtype=np.float32)")
            if hasattr(self,"VelDisp_Mean200"):
                print("  VelDisp_Mean200   (dtype=np.float32)")
                print("  VelDisp_Crit200   (dtype=np.float32)")
                print("  VelDisp_TopHat200 (dtype=np.float32)")

        print("  NsubPerHalo       (dtype=np.int32)")
        print("  FirstSubOfHalo    (dtype=np.int32)")
        print("  Halo_M_Mean200    (dtype=np.float32)")
        print("  Halo_R_Mean200    (dtype=np.float32)")
        print("  Halo_M_Crit200    (dtype=np.float32)")
        print("  Halo_R_Crit200    (dtype=np.float32)")
        print("  Halo_M_TopHat200  (dtype=np.float32)")
        print("  Halo_R_TopHat200  (dtype=np.float32)")

        if hasattr(self,"HaloCont"):
            print("  HaloCont           (dtype=np.float32)")

        print("Vectors (length TotNsubhalos)")
        print("  SubLen            (dtype=np.int32)")
        print("  SubOffset         (dtype=np.int32)")
        print("  SubParentHalo     (dtype=np.int32)")
        print("  SubPos            (dtype=np.dtype((np.float32,3)))")
        print("  SubVel            (dtype=np.dtype((np.float32,3)))")
        print("  SubVelDisp        (dtype=np.float32)")
        print("  SubVmax           (dtype=np.float32)")
        print("  SubSpin           (dtype=np.dtype((np.float32,3)))")
        print("  SubMostBoundID    (dtype=",self.id_format,")")
        print("  SubHalfMass       (dtype=np.float32)")
        if hasattr(self,"SaveMassTab"):
            print("  SubMassTab        (dtype=np.dtype((np.float32,6)))")

        if self.myformat==1:
            print("  SubTMass          (dtype=np.float32)")
            print("  SubCM             (dtype=np.dtype((np.float32,3)))")
            print("  SubVmaxRad        (dtype=np.float32)")
            print("  SubGroupNumber    (dtype=np.int32)")

        if hasattr(self,"GroupIDs"):
            print("Particle IDs of FoF groups (length TotNids)")
            print("  GroupIDs          (dtype=",self.id_format,")")
        else:
            print("Particle IDs have not been read")
            print("To read them: catalog.readIDs(long_IDs=[True,False(default)],verbose=[0,1])")


