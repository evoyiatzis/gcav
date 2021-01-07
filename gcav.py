#!/usr/local/bin/python

import os
import sys 
import datetime
import numpy 
 
import scipy.weave as weave
from scipy.spatial import Delaunay

path = os.environ["LAMMPS_PYTHON_TOOLS"] # link with the pizza toolkit
sys.path.append(path)
from dump import dump
from data import data

num_confs = 1 # parameter defining how many configurations there are in the dump file
mc_attempts = 200000 # number of iterations in the MC free volume calculation
clustering_radius = 1.28 # radius of inscribed circle which is necessary in order to connect two tetrahedra
unoccupied_volume_info = True # 0 write the unoccupied volume of each tetrahedron and 1 avoids writing
AtomsPerCavityInfo = True # 0 writes the cavity index for each atom of the system in a separate file 

# open both files
f = data(sys.argv[1]) # data file

g = dump(sys.argv[2], 0)
g.map(1,"id",2,"x",3,"y",4,"z")

# read the number of atoms in the data file 
n_atoms = f.headers["atoms"]

Mass = numpy.zeros(n_atoms) 
AtTypes = numpy.zeros(n_atoms, dtype = numpy.int)

TempMass = f.get("Masses")
TempSigma = f.get("Pair Coeffs")
TempAtoms = f.get("Atoms")

# assign the mass & the LJ sigma of each atom type
TableMass = TempMass[numpy.int(TempMass[:][0]) - 1][1]
TableSigma = 0.5*pow(2.0,1.0/6.0)*TempSigma[TempSigma[:][0].astype(int) - 1][2]

# read the atom types & assign the atomic masses 
AtTypes[TempAtoms[:][0].astype(int) - 1] = TempAtoms[:][2].astype(int) - 1
Mass[TempAtoms[:][0].astype(int) - 1] = TableMass[TempAtoms[:][2].astype(int) - 1]

# start the iterative procedure for all configurations 
for iconf in range(0, num_confs):

 # read one configuration and get the coordinates
 time = g.next()
 if time == -1:  
   sys.exit("the dump file contains less than NumConfs configurations")

 # the qhull library is not working for periodic boundary conditions
 points = numpy.zeros((n_atoms,3))
 non_pbc_points = numpy.zeros((27*n_atoms,3))

 box_x = g.snaps[iconf].xhi - g.snaps[iconf].xlo
 box_y = g.snaps[iconf].yhi - g.snaps[iconf].ylo
 box_z = g.snaps[iconf].zhi - g.snaps[iconf].zlo
 
 box = numpy.asarray([box_x, box_y, box_z])
  
 g.unscale()
 pizza_id, pizza_x, pizza_y, pizza_z = g.vecs(time,"id","x","y","z")

 points[:, 0] = numpy.asarray(pizza_x[numpy.asarray(pizza_id, dtype=numpy.int) - 1])
 points[:, 1] = numpy.asarray(pizza_y[numpy.asarray(pizza_id, dtype=numpy.int) - 1])
 points[:, 2] = numpy.asarray(pizza_z[numpy.asarray(pizza_id, dtype=numpy.int) - 1])

 # fold the coordinates inside the primary simulation box 
 points[:, 0] = points[:, 0] - box_x*numpy.round(points[:, 0]/box_x)
 points[:, 1] = points[:, 1] - box_y*numpy.round(points[:, 1]/box_y)
 points[:, 2] = points[:, 2] - box_z*numpy.round(points[:, 2]/box_z)
 
 # multiply the atoms in the primary simulation box to all three directions 
 non_pbc_points[0:n_atoms, 0] = points[:, 0]
 non_pbc_points[0:n_atoms, 1] = points[:, 1] 
 non_pbc_points[0:n_atoms, 2] = points[:, 2]
 
 indx = -1
 for ix in range(-1, 2):
     for iy in range(-1, 2):
         for iz in range(-1, 2):
             if numpy.abs(ix) + numpy.abs(iy) + numpy.abs(iz) == 0: 
                 continue
             indx += 1
             non_pbc_points[n_atoms+indx:27*n_atoms:26, 0] = points[:, 0] + ix*box[0]
             non_pbc_points[n_atoms+indx:27*n_atoms:26, 1] = points[:, 1] + iy*box[1]
             non_pbc_points[n_atoms+indx:27*n_atoms:26, 2] = points[:, 2] + iz*box[2]
 
 first_timing = datetime.datetime.now()

 tri = Delaunay(non_pbc_points)

 # delete all tetrahedra that lay completely out of the primary simulation box
 n_tetrahedra = len(tri.vertices)
 CorrespondingTetrahedron = numpy.zeros(n_tetrahedra, dtype = numpy.int)

 aa = (tri.vertices[:, 0] > n_atoms-1)
 bb = (tri.vertices[:, 1] > n_atoms-1)
 cc = (tri.vertices[:, 2] > n_atoms-1)
 dd = (tri.vertices[:, 3] > n_atoms-1)
 ee =  aa + bb + cc + dd
 CorrespondingTetrahedron[ee > 2] = -10 
 SuspiciousTriangles = numpy.where(ee == 2)[0]
  
 # change the index of the atoms that lay outside the box and are minimum images of atoms within the box
 for icount, itetrahedron in enumerate(numpy.where(CorrespondingTetrahedron > -10)[0]): 
     CorrespondingTetrahedron[itetrahedron] = icount 
     
 for idim in range(0, 4):
  aa = (tri.vertices[:, idim] > n_atoms-1)
  tri.vertices[aa, idim] = tri.vertices[aa, idim] % n_atoms

 # sort the atom indices 
 for CurrentTriangleID in SuspiciousTriangles: 
  sorted_atom_array = numpy.sort(tri.vertices[CurrentTriangleID])
  tri.vertices[CurrentTriangleID][:] = sorted_atom_array

 # remove periodic images of the same tetrahedron with two atoms as images
 n_suspicious_triangles = len(SuspiciousTriangles)
 iduplicate = 0
 while iduplicate < n_suspicious_triangles:
  itetrahedron = SuspiciousTriangles[iduplicate]
  if CorrespondingTetrahedron[itetrahedron] > -10:
   IduplAtoms = tri.vertices[itetrahedron]
   break_happened = 0
   for jtetrahedron in SuspiciousTriangles[iduplicate+1:n_suspicious_triangles]:
    if CorrespondingTetrahedron[jtetrahedron] > -10:
     if (IduplAtoms == tri.vertices[jtetrahedron]).all(): 
         CorrespondingTetrahedron[itetrahedron] = -10
         iduplicate += 1
         break_happened = 1
         break
   if break_happened == 0:
    iduplicate += 1
  else:
   iduplicate += 1

 second_timing = datetime.datetime.now()
 print('delaunay time: {}'.format((second_timing - first_timing).seconds))
 
 delaunay_volume = numpy.zeros(n_tetrahedra) 
 TetrahedronCOM = numpy.zeros([n_tetrahedra,3])

 # divide the space to smaller sections and create the neighborhood list
 MaxLenSubboxes = max(TableSigma) + clustering_radius
 NumSubboxes = numpy.int(box/MaxLenSubboxes) + 1

 ListOfNeighbors = []

 ibin_x = numpy.int((points[:, 0] + 0.5*box_x) / MaxLenSubboxes)
 ibin_y = numpy.int((points[:, 1] + 0.5*box_y) / MaxLenSubboxes)
 ibin_z = numpy.int((points[:, 2] + 0.5*box_z) / MaxLenSubboxes)
 indx = ibin_x + NumSubboxes[0]*ibin_y + NumSubboxes[0]*NumSubboxes[1]*ibin_z
 for ibox in range(0, NumSubboxes[0]*NumSubboxes[1]*NumSubboxes[2]):
     ListOfNeighbors.append(numpy.where(indx == ibox)[0])

 # compute the free volume of each tetrahedron by means of Monte Carlo integration  
 for iat in numpy.where(CorrespondingTetrahedron != -10)[0] : 

  a = tri.vertices[iat, 0:4]  # get the index of the neighboors

  PointA = points[a[0], :]  
  PointB = points[a[1], :] - box*numpy.round((points[a[1], :] - PointA)/box) 
  PointC = points[a[2], :] - box*numpy.round((points[a[2], :] - PointA)/box) 
  PointD = points[a[3], :] - box*numpy.round((points[a[3], :] - PointA)/box) 

  SqSigmaA = TableSigma[AtTypes[a[0] % n_atoms]] + clustering_radius
  SqSigmaB = TableSigma[AtTypes[a[1] % n_atoms]] + clustering_radius
  SqSigmaC = TableSigma[AtTypes[a[2] % n_atoms]] + clustering_radius
  SqSigmaD = TableSigma[AtTypes[a[3] % n_atoms]] + clustering_radius

  # here I compute the centre-of-mass of the tetrahedron
  normalize_mass = Mass[a[0] % n_atoms] + Mass[a[1] % n_atoms] + Mass[a[2] % n_atoms] + Mass[a[3] % n_atoms] 
  TetrahedronCOM[iat,:] = (Mass[a[0] % n_atoms]*PointA + Mass[a[1] % n_atoms]*PointB + Mass[a[2] % n_atoms]*PointC + Mass[a[3] % n_atoms]*PointD)/normalize_mass

  # find the atoms that should be additionally checked for overlaps
  ibin_xa = int((PointA[0] + 0.5*box_x) / MaxLenSubboxes)
  if PointA[0] + 0.5*box_x < 0 : 
   ibin_xa -= 1
  ibin_ya = int((PointA[1] + 0.5*box_y) / MaxLenSubboxes)
  if PointA[1] + 0.5*box_y < 0 : 
   ibin_ya -= 1
  ibin_za = int((PointA[2] + 0.5*box_z) / MaxLenSubboxes)
  if PointA[2] + 0.5*box_z < 0 : 
   ibin_za -= 1

  ibin_xb = int((PointB[0] + 0.5*box_x) / MaxLenSubboxes)
  if PointB[0] + 0.5*box_x < 0 : 
   ibin_xb -= 1
  ibin_yb = int((PointB[1] + 0.5*box_y) / MaxLenSubboxes)
  if PointB[1] + 0.5*box_y < 0 : 
   ibin_yb -= 1
  ibin_zb = int((PointB[2] + 0.5*box_z) / MaxLenSubboxes)
  if PointB[2] + 0.5*box_z < 0 : 
   ibin_zb -= 1

  ibin_xc = int((PointC[0] + 0.5*box_x) / MaxLenSubboxes)
  if PointC[0] + 0.5*box_x < 0 : 
   ibin_xc -= 1
  ibin_yc = int((PointC[1] + 0.5*box_y) / MaxLenSubboxes)
  if PointC[1] + 0.5*box_y < 0 : 
   ibin_yc -= 1
  ibin_zc = int((PointC[2] + 0.5*box_z) / MaxLenSubboxes)
  if PointC[2] + 0.5*box_z < 0 : 
   ibin_zc -= 1

  ibin_xd = int((PointD[0] + 0.5*box_x) / MaxLenSubboxes)
  if PointD[0] + 0.5*box_x < 0 : 
   ibin_xd -= 1
  ibin_yd = int((PointD[1] + 0.5*box_y) / MaxLenSubboxes)
  if PointD[1] + 0.5*box_y < 0 : 
   ibin_yd -= 1
  ibin_zd = int((PointD[2] + 0.5*box_z) / MaxLenSubboxes)
  if PointD[2] + 0.5*box_z < 0 : 
   ibin_zd -= 1

  istart_x = min(ibin_xa,ibin_xb,ibin_xc,ibin_xd) - 1
  istart_y = min(ibin_ya,ibin_yb,ibin_yc,ibin_yd) - 1
  istart_z = min(ibin_za,ibin_zb,ibin_zc,ibin_zd) - 1

  ifinish_x = max(ibin_xa,ibin_xb,ibin_xc,ibin_xd) + 1
  ifinish_y = max(ibin_ya,ibin_yb,ibin_yc,ibin_yd) + 1
  ifinish_z = max(ibin_za,ibin_zb,ibin_zc,ibin_zd) + 1

  AdditionalList = [ ]

  ixx = numpy.linspace(istart_x, ifinish_x, ifinish_x+1)
  ixx[ixx < 0] += NumSubboxes[0]
  ixx[ixx > NumSubboxes[0] - 1] -= NumSubboxes[0]

  iyy = numpy.linspace(istart_y, ifinish_y, ifinish_y+1)
  iyy[iyy < 0] += NumSubboxes[1]
  iyy[iyy > NumSubboxes[1] - 1] -= NumSubboxes[1]

  izz = numpy.linspace(istart_z, ifinish_z, ifinish_z+1)
  izz[izz < 0] += NumSubboxes[2]
  izz[izz > NumSubboxes[2] - 1] -= NumSubboxes[2]

  for ix in ixx:
      for iy in iyy: 
          for iz in izz: 
              indx = ixx + NumSubboxes[0]*iyy + NumSubboxes[0]*NumSubboxes[1]*izz
              AdditionalList += ListOfNeighbors[indx]

  # check if any of the points in the neighboring boxes does not interact with the current tetrahedron
  AdditionalList.remove(a[0])
  AdditionalList.remove(a[1])
  AdditionalList.remove(a[2])
  AdditionalList.remove(a[3])
  
  AdditionalAtoms = len(AdditionalList)
  PassCoordinatesToWeave = numpy.zeros([4*AdditionalAtoms],dtype=numpy.float64)
  for indx, iatom in enumerate(AdditionalList):
   dx = points[iatom, :] - box*numpy.round((points[iatom, :] - TetrahedronCOM[iat, :])/box)
   PassCoordinatesToWeave[4*indx:4*indx+3] = dx[:] 
   PassCoordinatesToWeave[4*indx+3] = TableSigma[AtTypes[iatom % n_atoms]] + clustering_radius
  
  weave_options = \
  {
   'headers': ['<stdlib.h>','<time.h>','<math.h>'],
   'extra_compile_args': ['-O2']
  }

  # inline C++ code for the Monte Carlo 
  code = """ double InlineA[3]; \
             double InlineB[3]; \
             double VectorE[3]; \
\
             InlineA[0] = PointB[0] - PointD[0]; \
             InlineA[1] = PointB[1] - PointD[1]; \
             InlineA[2] = PointB[2] - PointD[2]; \
\
             InlineB[0] = PointC[0] - PointD[0]; \
             InlineB[1] = PointC[1] - PointD[1]; \
             InlineB[2] = PointC[2] - PointD[2]; \
\
             VectorE[0] = InlineA[1]*InlineB[2] - InlineA[2]*InlineB[1]; \
             VectorE[1] = InlineA[2]*InlineB[0] - InlineA[0]*InlineB[2]; \
             VectorE[2] = InlineA[0]*InlineB[1] - InlineA[1]*InlineB[0]; \
\
             double FullVolume =  fabs((PointA[0]-PointD[0])*VectorE[0] + (PointA[1]-PointD[1])*VectorE[1] + (PointA[2]-PointD[2])*VectorE[2])/6.0; \
\
             int FinalAdditionalAtoms = 0; \
\
             int tri_in_tetra[4][3]; \
             tri_in_tetra[0][0] = 0; tri_in_tetra[0][1] = 1; tri_in_tetra[0][2] = 2; \
             tri_in_tetra[1][0] = 0; tri_in_tetra[1][1] = 2; tri_in_tetra[1][2] = 3; \
             tri_in_tetra[2][0] = 0; tri_in_tetra[2][1] = 1; tri_in_tetra[2][2] = 3; \
             tri_in_tetra[3][0] = 1; tri_in_tetra[3][1] = 2; tri_in_tetra[3][2] = 3; \
\
             double tri[5][3]; \
             for(int coord=0; coord < 3; coord++) { \
              tri[0][coord]=PointA[coord]; \
              tri[1][coord]=PointB[coord]; \
              tri[2][coord]=PointC[coord]; \
              tri[3][coord]=PointD[coord]; \
             } \
\
             double FinalComparisonList[4*AdditionalAtoms]; \
             int noover; \
             int overlap; \
             for (int itetra = 0; itetra < AdditionalAtoms; itetra++) { \
\ 
              double tri_origin[3][3]; \
\
              double rr = PassCoordinatesToWeave[4*itetra+3]; \
              noover = 0; \
              overlap = 0; \
\
              tri[4][0] = PassCoordinatesToWeave[4*itetra]; \
              tri[4][1] = PassCoordinatesToWeave[4*itetra+1]; \
              tri[4][2] = PassCoordinatesToWeave[4*itetra+2]; \
\
             for (int itri = 0; itri < 4; itri++) { \
\
              for (int coord = 0; coord < 3; coord++) {  \
               tri_origin[0][coord]=tri[tri_in_tetra[itri][0]][coord]-tri[4][coord]; \
               tri_origin[1][coord]=tri[tri_in_tetra[itri][1]][coord]-tri[4][coord]; \
               tri_origin[2][coord]=tri[tri_in_tetra[itri][2]][coord]-tri[4][coord]; \
              } \
\
              double vec_V[3], vec_u[3], vec_v[3] ; \
\
              for(int coord = 0; coord < 3; coord++) { \
               vec_u[coord] = tri_origin[1][coord] - tri_origin[0][coord]; \
               vec_v[coord] = tri_origin[2][coord] - tri_origin[0][coord]; \
              } \
\
              vec_V[0] = vec_u[1]*vec_v[2] - vec_u[2]*vec_v[1]; \
              vec_V[1] = vec_u[2]*vec_v[0] - vec_u[0]*vec_v[2]; \
              vec_V[2] = vec_u[0]*vec_v[1] - vec_u[1]*vec_v[0]; \
\
              double d = tri_origin[0][0]*vec_V[0] + tri_origin[0][1]*vec_V[1] + tri_origin[0][2]*vec_V[2]; \
              double e = vec_V[0]*vec_V[0]+vec_V[1]*vec_V[1]+vec_V[2]*vec_V[2]; \
\
              double aa = tri_origin[0][0]*tri_origin[0][0] + tri_origin[0][1]*tri_origin[0][1] + tri_origin[0][2]*tri_origin[0][2]; \
              double ab = tri_origin[0][0]*tri_origin[1][0] + tri_origin[0][1]*tri_origin[1][1] + tri_origin[0][2]*tri_origin[1][2]; \
              double ac = tri_origin[0][0]*tri_origin[2][0] + tri_origin[0][1]*tri_origin[2][1] + tri_origin[0][2]*tri_origin[2][2]; \
              double bb = tri_origin[1][0]*tri_origin[1][0] + tri_origin[1][1]*tri_origin[1][1] + tri_origin[1][2]*tri_origin[1][2]; \
              double bc = tri_origin[1][0]*tri_origin[2][0] + tri_origin[1][1]*tri_origin[2][1] + tri_origin[1][2]*tri_origin[2][2]; \
              double cc = tri_origin[2][0]*tri_origin[2][0] + tri_origin[2][1]*tri_origin[2][1] + tri_origin[2][2]*tri_origin[2][2]; \
\
              double vec_AB[3], vec_BC[3], vec_CA[3]; \
\
              for (int coord = 0; coord < 3; coord++) {  \
               vec_AB[coord] = tri_origin[1][coord] - tri_origin[0][coord] ; \
               vec_BC[coord] = tri_origin[2][coord] - tri_origin[1][coord] ; \
               vec_CA[coord] = tri_origin[0][coord] - tri_origin[2][coord] ; \
              } \
\
              double d1 = ab - aa; \
              double d2 = bc - bb; \
              double d3 = ac - cc; \
\
              double e1 = vec_AB[0]*vec_AB[0]+vec_AB[1]*vec_AB[1]+vec_AB[2]*vec_AB[2]; \
              double e2 = vec_BC[0]*vec_BC[0]+vec_BC[1]*vec_BC[1]+vec_BC[2]*vec_BC[2]; \
              double e3 = vec_CA[0]*vec_CA[0]+vec_CA[1]*vec_CA[1]+vec_CA[2]*vec_CA[2]; \
\
              double vec_Q1[3], vec_Q2[3], vec_Q3[3], vec_QC[3], vec_QA[3], vec_QB[3]; \
\
              for (int coord = 0; coord < 3; coord++) { \
               vec_Q1[coord]=(tri_origin[0][coord]*e1)-(vec_AB[coord]*d1); \
               vec_Q2[coord]=(tri_origin[1][coord]*e2)-(vec_BC[coord]*d2); \
               vec_Q3[coord]=(tri_origin[2][coord]*e3)-(vec_CA[coord]*d3); \
               vec_QC[coord]=(tri_origin[2][coord]*e1)-(vec_Q1[coord]); \
               vec_QA[coord]=(tri_origin[0][coord]*e2)-(vec_Q2[coord]); \
               vec_QB[coord]=(tri_origin[1][coord]*e3)-(vec_Q3[coord]); \
              } \
\
              int sep1 = 1, sep2 = 1, sep3 = 1, sep4 = 1, sep5 = 1, sep6 = 1, sep7 = 1; \
\
              double Q1_Q1 = vec_Q1[0]*vec_Q1[0]+vec_Q1[1]*vec_Q1[1]+vec_Q1[2]*vec_Q1[2]; \
              double Q1_QC = vec_Q1[0]*vec_QC[0]+vec_Q1[1]*vec_QC[1]+vec_Q1[2]*vec_QC[2]; \
              double Q2_Q2 = vec_Q2[0]*vec_Q2[0]+vec_Q2[1]*vec_Q2[1]+vec_Q2[2]*vec_Q2[2]; \
              double Q2_QA = vec_Q2[0]*vec_QA[0]+vec_Q2[1]*vec_QA[1]+vec_Q2[2]*vec_QA[2]; \
              double Q3_Q3 = vec_Q3[0]*vec_Q3[0]+vec_Q3[1]*vec_Q3[1]+vec_Q3[2]*vec_Q3[2]; \
              double Q3_QB = vec_Q3[0]*vec_QB[0]+vec_Q3[1]*vec_QB[1]+vec_Q3[2]*vec_QB[2]; \
\
              if (d*d>rr*e) { \
               sep1 = 0; \
              } \
              if ((aa>rr) && (ab>aa) && (ac>aa)) { \
               sep2 = 0; \
              } \
              if ((bb>rr) && (ab>bb) && (bc>bb)) { \
               sep3 = 0; \
              } \
              if ((cc>rr) && (ac>cc) && (bc>cc)) { \
               sep4 = 0; \
              } \
              if ((Q1_Q1>rr*e1*e1) && (Q1_QC>0)) { \
               sep5 = 0; \
              } \
              if ((Q2_Q2>rr*e2*e2) && (Q2_QA>0)) { \
               sep6 = 0; \
              } \
              if ((Q3_Q3>rr*e3*e3) && (Q3_QB>0)) { \
               sep7 = 0; \
              } \
\
              if (sep1==0) { \
               noover++ ; \
              } else if ((sep2==0) || (sep3==0) || (sep4==0))  { \
               noover++ ; \              
              } else if ((sep5==0) || (sep6==0) || (sep7==0))  { \
               noover++ ; \
              } \
             if (noover<4) ; { \
              overlap++ ; \
             } \
\
             } \
             if (overlap>0) { \
              FinalComparisonList[4*FinalAdditionalAtoms] =   PassCoordinatesToWeave[4*itetra]; \
              FinalComparisonList[4*FinalAdditionalAtoms+1] = PassCoordinatesToWeave[4*itetra+1]; \
              FinalComparisonList[4*FinalAdditionalAtoms+2] = PassCoordinatesToWeave[4*itetra+2]; \
              FinalComparisonList[4*FinalAdditionalAtoms+3] = PassCoordinatesToWeave[4*itetra+3]; \
              FinalAdditionalAtoms++; \
             } \
            } \
\
             int AccVolume = 0; \
             srand (time(NULL));\
             int iMC ; \
             double lambda1, lambda2, lambda3, lambda4, lambda_sum, TrialPointX, TrialPointY, TrialPointZ, dx, dy, dz, dist; \
\
             for (iMC = 0; iMC < mc_attempts; iMC++) { \
              lambda1 = ((double) rand()) / (RAND_MAX) ; \
              lambda2 = ((double) rand()) / (RAND_MAX) ; \
              lambda3 = ((double) rand()) / (RAND_MAX) ; \
              lambda4 = ((double) rand()) / (RAND_MAX) ; \
              lambda_sum = lambda1 + lambda2 + lambda3 + lambda4; \
              lambda1 = lambda1 / lambda_sum; \
              lambda2 = lambda2 / lambda_sum; \
              lambda3 = lambda3 / lambda_sum; \
              lambda4 = lambda4 / lambda_sum; \
\
              TrialPointX = lambda1*PointA[0] + lambda2*PointB[0] + lambda3*PointC[0] + lambda4*PointD[0]; \
              TrialPointY = lambda1*PointA[1] + lambda2*PointB[1] + lambda3*PointC[1] + lambda4*PointD[1]; \
              TrialPointZ = lambda1*PointA[2] + lambda2*PointB[2] + lambda3*PointC[2] + lambda4*PointD[2]; \
\
              dx = PointA[0] - TrialPointX; \
              dy = PointA[1] - TrialPointY; \
              dz = PointA[2] - TrialPointZ; \
              dist = sqrt(dx*dx + dy*dy + dz*dz); \
              if (dist < SqSigmaA) continue; \
\
              dx = PointB[0] - TrialPointX; \
              dy = PointB[1] - TrialPointY; \
              dz = PointB[2] - TrialPointZ; \
              dist = sqrt(dx*dx + dy*dy + dz*dz); \
              if (dist < SqSigmaB) continue; \
\
              dx = PointC[0] - TrialPointX; \
              dy = PointC[1] - TrialPointY; \
              dz = PointC[2] - TrialPointZ; \
              dist = sqrt(dx*dx + dy*dy + dz*dz); \
              if (dist < SqSigmaC) continue; \
\
              dx = PointD[0] - TrialPointX; \
              dy = PointD[1] - TrialPointY; \
              dz = PointD[2] - TrialPointZ; \
              dist = sqrt(dx*dx + dy*dy + dz*dz); \
              if (dist < SqSigmaD) continue; \
\
              int count = 0; 
              for (int iat = 0; iat < FinalAdditionalAtoms; iat++) { \
               dx = FinalComparisonList[4*iat] - TrialPointX; \
               dy = FinalComparisonList[4*iat+1] - TrialPointY; \
               dz = FinalComparisonList[4*iat+2] - TrialPointZ; \
               dist = sqrt(dx*dx + dy*dy + dz*dz); \
               if (dist < FinalComparisonList[4*iat+3]) { \
                count++; \
               }; \
              }
              if (count == 0) { \
               AccVolume++ ; \
              }\
             } \
             return_val = FullVolume * ((double) AccVolume) / ((double) mc_attempts) ;"""

  delaunay_volume[iat] = weave.inline(code,['PointA','PointB','PointC','PointD','SqSigmaA','SqSigmaB','SqSigmaC','SqSigmaD','PassCoordinatesToWeave','AdditionalAtoms','mc_attempts'],**weave_options)

 # write the information for the unoccupied volume of each tetrahedron to a file
 if unoccupied_volume_info: 
  kstring = 'UnoccupiedVolume_'+str(iconf)+'.txt'
  with open(kstring,'w') as UnVlFile: # output file each line corresponds to one tetrahdron 
   numpy.savetxt(UnVlFile, delaunay_volume[CorrespondingTetrahedron[iat] != -10], delimiter=' ')

 third_timing = datetime.datetime.now()
 print('Monte Carlo computing time: {}'.format((third_timing - second_timing).seconds))

 # clustering algorithm of the delaunay tetrahedra
 Ncavities = -1
 CavityIndex = numpy.full(n_tetrahedra, -1)

 ZeroElements = n_tetrahedra
 
 CavityIndex[CorrespondingTetrahedron == -10] = -10

 while ZeroElements > 0: # are there tetrahedra which do not belong to a specific cavity?

  iat = numpy.where(CavityIndex == -1)[0][0] # pick up one tetrahedron which is not assigned to a cavity

  Ncavities += 1 # assign this tetrahedron to a new cavity
  CavityIndex[iat] = Ncavities

  MergeTetrahedron = [iat]

  while len(MergeTetrahedron) > 0:

   NewMergeTetrahedron = [ ]

   for item_tetrahedron in MergeTetrahedron: 

    IndexFirstNeighbors = tri.neighbors[item_tetrahedron] # get the neighboring tetrahedra 
    CentralAtoms = tri.vertices[item_tetrahedron] # get the index of the atoms forming the current tetrahedron

    for jtetrahedron in IndexFirstNeighbors[IndexFirstNeighbors < 0]: # for each of the neighboring triangles
     if CavityIndex[jtetrahedron] == Ncavities:
         continue # the two tetrahedra are already clustered 
     CommonAtoms = numpy.intersect1d(CentralAtoms, tri.vertices[jtetrahedron]) # find the common atoms with the surrounding tetrahedra
     if numpy.size(CommonAtoms) == 3: # if the two tetrahedra form a surface then  
      A = points[CommonAtoms[0], :] 
      B = points[CommonAtoms[1], :]
      C = points[CommonAtoms[2], :]

      SigmaA = TableSigma[AtTypes[CommonAtoms[0] % n_atoms]] 
      SigmaB = TableSigma[AtTypes[CommonAtoms[1] % n_atoms]] 
      SigmaC = TableSigma[AtTypes[CommonAtoms[2] % n_atoms]] 

      ClusteringCode = """ \
\
        double DecisionVariable = 2000; \
\
        double dx = B[0] - A[0]; \
        double dy = B[1] - A[1]; \
        double dz = B[2] - A[2]; \
\
        dx = dx - box_x*round(dx/box_x); \
        dy = dy - box_y*round(dy/box_y); \
        dz = dz - box_z*round(dz/box_z); \
\
        B[0] = dx + A[0]; \
        B[1] = dy + A[1]; \
        B[2] = dz + A[2]; \
\
        double distanceAB = sqrt(dx*dx + dy*dy + dz*dz); \
\
        dx = C[0] - A[0]; \
        dy = C[1] - A[1]; \
        dz = C[2] - A[2]; \
\
        dx = dx - box_x*round(dx/box_x); \
        dy = dy - box_y*round(dy/box_y); \
        dz = dz - box_z*round(dz/box_z); \
\
        C[0] = dx + A[0]; \
        C[1] = dy + A[1]; \
        C[2] = dz + A[2]; \
\
        double distanceAC = sqrt(dx*dx + dy*dy + dz*dz); \
\
        dx = B[0] - C[0]; \
        dy = B[1] - C[1]; \
        dz = B[2] - C[2]; \
\
        double distanceBC = sqrt(dx*dx + dy*dy + dz*dz); \
\
        if (distanceAB < SigmaA + clustering_radius) { \
         if (distanceAC < SigmaA + clustering_radius) { \
          DecisionVariable = -1; \
         } \
        } \
\
        if (distanceAB < SigmaB + clustering_radius) { \
         if (distanceBC < SigmaB + clustering_radius) { \
          DecisionVariable = -1; \
         } \
        } \
\
        if (distanceAC < SigmaC + clustering_radius) { \
         if (distanceBC < SigmaC + clustering_radius) { \
          DecisionVariable = -1; \
         } \
        } \
\
        if (DecisionVariable > 0) { \
        if (distanceAC < SigmaC + SigmaA + 2*clustering_radius) { \
         if (distanceAB < SigmaB + SigmaA + 2*clustering_radius) { \
          if (distanceBC < SigmaC + SigmaB + 2*clustering_radius) { \
\
           double d = distanceAB; \
\
           double ExVector[3]; \
           ExVector[0] = (B[0] - A[0]) / distanceAB; \
           ExVector[1] = (B[1] - A[1]) / distanceAB; \
           ExVector[2] = (B[2] - A[2]) / distanceAB; \
\
           double i = ExVector[0]*(C[0]-B[0]) + ExVector[1]*(C[1]-B[1]) + ExVector[2]*(C[2]-B[2]); \
\
           double EyVector[3]; \
           dx = C[0] - A[0] - i*ExVector[0]; \
           dy = C[1] - A[1] - i*ExVector[1]; \
           dz = C[2] - A[2] - i*ExVector[2]; \
           double norm = sqrt(dx*dx+dy*dy+dz*dz); \
           EyVector[0] = dx / norm; \
           EyVector[1] = dy / norm; \
           EyVector[2] = dz / norm; \
\
           double j = EyVector[0]*(C[0]-A[0]) + EyVector[1]*(C[1]-A[1]) + EyVector[2]*(C[2]-A[2]); \
\
           double r1 = SigmaA + clustering_radius; \
           double r2 = SigmaB + clustering_radius; \
           double r3 = SigmaC + clustering_radius; \
\
           double xcoordinate = (r1*r1 + d*d - r2*r2) / (2*d); \
           double ycoordinate = (r1*r1-r3*r3+i*i+j*j)/(2*j) - (i/j)*xcoordinate; \
           double sq_zcoordinate = r1*r1 - xcoordinate*xcoordinate - ycoordinate*ycoordinate; \
           if (sq_zcoordinate < 0) { \
            DecisionVariable = 1; \
           } else { \
            DecisionVariable = -2; \
           } \
          } \
         } \
        } \
\
        if (DecisionVariable == 2000) { \
\
        if (distanceAC < SigmaC + SigmaA + 2*clustering_radius) { \
         if (distanceAB < SigmaB + SigmaA + 2*clustering_radius) { \
          double radiusA = SigmaA + clustering_radius; \
\
          double dxCA = C[0] - A[0]; \
          double dyCA = C[1] - A[1]; \
          double dzCA = C[2] - A[2]; \
\
          double dxBC = B[0] - C[0]; \
          double dyBC = B[1] - C[1]; \
          double dzBC = B[2] - C[2]; \
\
          double alpha = dxBC*dxBC + dyBC*dyBC + dzBC*dzBC; \
          double beta  = -2*(dxCA*dxBC + dyCA*dyBC + dzCA*dzBC); \
          double gamma = dxCA*dxCA + dyCA*dyCA + dzCA*dzCA - radiusA*radiusA; \
\
          double diakrinousa = beta*beta - 4*alpha*gamma; \
          if (diakrinousa > 0) {
           double tau1 = (-beta + sqrt(diakrinousa)) / (2*alpha); \
           double tau2 = (-beta - sqrt(diakrinousa)) / (2*alpha); \
           if (tau1 > tau2) { \
            double temp = tau2; \
            tau2 = tau1; \
            tau1 = temp; \
           } \
\
           if (tau1 > 1 || tau2 < 0) { \
            DecisionVariable = 1; \
           } \
\
           double dist1 = -1;\
           if (tau1 > 0) { \
            double dx1 = dxBC*tau1; \
            double dy1 = dyBC*tau1; \
            double dz1 = dzBC*tau1; \
            dist1 = sqrt(dx1*dx1 + dy1*dy1 + dz1*dz1) - SigmaC - clustering_radius ; \
           } \
\
           double dist2 = -1;\
           if (tau2 > 0 && tau2 < 1) { \
            double dx2 = dxCA + dxBC*tau2 - (B[0] - A[0]); \
            double dy2 = dyCA + dyBC*tau2 - (B[1] - A[1]); \
            double dz2 = dzCA + dzBC*tau2 - (B[2] - A[2]); \
            double dist2 = sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2) - SigmaB - clustering_radius; \
           } \
\
           if (dist1 < 0 && dist2 < 0) { \
            DecisionVariable = 3; \
           } else { \
            DecisionVariable = 1; \
           }
          } else { \
           DecisionVariable = 1; \
          } \
         } \
        } \
\
        if (distanceAC < SigmaC + SigmaA + 2*clustering_radius) { \
         if (distanceBC < SigmaC + SigmaB + 2*clustering_radius) { \
          double radiusC = SigmaC + clustering_radius; \
\
          double dxAC = A[0] - C[0]; \
          double dyAC = A[1] - C[1]; \
          double dzAC = A[2] - C[2]; \
\
          double dxBA = B[0] - A[0]; \
          double dyBA = B[1] - A[1]; \
          double dzBA = B[2] - A[2]; \
\
          double alpha = dxBA*dxBA + dyBA*dyBA + dzBA*dzBA; \
          double beta  = -2*(dxAC*dxBA + dyAC*dyBA + dzAC*dzBA); \
          double gamma = dxAC*dxAC + dyAC*dyAC + dzAC*dzAC - radiusC*radiusC; \
\
          double diakrinousa = beta*beta - 4*alpha*gamma; \
          if (diakrinousa > 0) {
           double tau1 = (-beta + sqrt(diakrinousa)) / (2*alpha); \
           double tau2 = (-beta - sqrt(diakrinousa)) / (2*alpha); \
           if (tau1 > tau2) { \
            double temp = tau2; \
            tau2 = tau1; \
            tau1 = temp; \
           } \
\
           if (tau1 > 1 || tau2 < 0) { \
            DecisionVariable = 1; \
           } \
\
           double dist1 = -1;\
           if (tau1 > 0) { \
            double dx1 = dxBA*tau1; \
            double dy1 = dyBA*tau1; \
            double dz1 = dzBA*tau1; \
            dist1 = sqrt(dx1*dx1 + dy1*dy1 + dz1*dz1) - SigmaA - clustering_radius ; \
           } \
\
           double dist2 = -1;\
           if (tau2 > 0 && tau2 < 1) { \
            double dx2 = dxAC + dxBA*tau2 - (B[0] - C[0]); \
            double dy2 = dyAC + dyBA*tau2 - (B[1] - C[1]); \
            double dz2 = dzAC + dzBA*tau2 - (B[2] - C[2]); \
            double dist2 = sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2) - SigmaB - clustering_radius; \
           } \
\
           if (dist1 < 0 && dist2 < 0) { \
            DecisionVariable = 3; \
           } else { \
            DecisionVariable = 1; \
           }
          } else { \
           DecisionVariable = 1; \
          } \

         } \
        } \
\
        if (distanceAB < SigmaB + SigmaA + 2*clustering_radius) { \
         if (distanceBC < SigmaC + SigmaB + 2*clustering_radius) { \
          double radiusB = SigmaB + clustering_radius; \
\
          double dxCB = C[0] - B[0]; \
          double dyCB = C[1] - B[1]; \
          double dzCB = C[2] - B[2]; \
\
          double dxAC = A[0] - C[0]; \
          double dyAC = A[1] - C[1]; \
          double dzAC = A[2] - C[2]; \
\
          double alpha = dxAC*dxAC + dyAC*dyAC + dzAC*dzAC; \
          double beta  = -2*(dxCB*dxAC + dyCB*dyAC + dzCB*dzAC); \
          double gamma = dxCB*dxCB + dyCB*dyCB + dzCB*dzCB - radiusB*radiusB; \
\
          double diakrinousa = beta*beta - 4*alpha*gamma; \
          if (diakrinousa > 0) {
           double tau1 = (-beta + sqrt(diakrinousa)) / (2*alpha); \
           double tau2 = (-beta - sqrt(diakrinousa)) / (2*alpha); \
           if (tau1 > tau2) { \
            double temp = tau2; \
            tau2 = tau1; \
            tau1 = temp; \
           } \
\
           if (tau1 > 1 || tau2 < 0) { \
            DecisionVariable = 1; \
           } \
\
           double dist1 = -1;\
           if (tau1 > 0) { \
            double dx1 = dxAC*tau1; \
            double dy1 = dyAC*tau1; \
            double dz1 = dzAC*tau1; \
            dist1 = sqrt(dx1*dx1 + dy1*dy1 + dz1*dz1) - SigmaC - clustering_radius ; \
           } \
\
           double dist2 = -1;\
           if (tau2 > 0 && tau2 < 1) { \
            double dx2 = dxAC + dxAC*tau2 - (A[0] - B[0]); \
            double dy2 = dyAC + dyAC*tau2 - (A[1] - B[1]); \
            double dz2 = dzAC + dzAC*tau2 - (A[2] - B[2]); \
            double dist2 = sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2) - SigmaB - clustering_radius; \
           } \
\
           if (dist1 < 0 && dist2 < 0) { \
            DecisionVariable = 3; \
           } else { \
            DecisionVariable = 1; \
           }
          } else { \
           DecisionVariable = 1; \
          } \
         } \
        } \
\
        } \
\
        } \
\
        return_val = DecisionVariable ; """
 
      ClusterDecision = weave.inline(ClusteringCode,['A','B','C','box_x','box_y','box_z','SigmaA','SigmaB','SigmaC','clustering_radius'],**weave_options)

      if ClusterDecision > 1: 
       CavityIndex[jtetrahedron] = Ncavities
       NewMergeTetrahedron.append(jtetrahedron)

   MergeTetrahedron = NewMergeTetrahedron

  ZeroElements = CavityIndex.count(-1) # count how many tetrahedra do not belong to a specific cavity

 # compute the COM of each cavity 
 CavityMinImage = numpy.zeros([3, (Ncavities+1)])
 cavity_com = numpy.zeros([3, (Ncavities+1)]) 
 cavity_volume = numpy.zeros(Ncavities+1)
 
 for iat, indx in zip(numpy.where(CavityIndex > -10)[0], CavityIndex[CavityIndex > -10]):
   cavity_volume[indx] += delaunay_volume[iat]

   if CavityMinImage[0,indx] == 0: 
    CavityMinImage[:,indx] = TetrahedronCOM[iat,:]

   TetrahedronCOM[iat,:] = TetrahedronCOM[iat,:] - box*numpy.round((TetrahedronCOM[iat,:] - CavityMinImage[:,indx])/box)
   cavity_com[:,indx] += delaunay_volume[iat]*TetrahedronCOM[iat,:]

 TrueCavities = numpy.where(cavity_volume > 0)[0]

 # export the information which atoms form a specific cavity to a file
 if AtomsPerCavityInfo:
  AtomsPerCavityString = 'AtomsPerCavityFile_'+str(iconf)+'.txt'
  with open(AtomsPerCavityString,'w') as AtomsPerCavityFile:
     
   for icavity in TrueCavities:
       atoms_in_cavity = numpy.zeros(0, dtype=numpy.int)
       for idelaunay in numpy.where(CavityIndex == icavity)[0]:
           atoms_in_cavity = numpy.append(atoms_in_cavity, tri.vertices[idelaunay, :])
           
       AtomsPerCavityFile.write("{} ".format(cavity_volume[icavity])) 
       numpy.savetxt(AtomsPerCavityFile, (atoms_in_cavity + 1))
       AtomsPerCavityFile.write("\n")

 # compute the volume-weighted centre of each cavity  
 cavity_com[:, TrueCavities] /= cavity_volume[TrueCavities]

 # compute the gyration tensor of each cavity
 CavityGyrationTensor = numpy.zeros([3,3,(Ncavities+1)]) 
 
 for icavity in TrueCavities: 
     for iat in numpy.where(CavityIndex == icavity)[0]:
         dx = TetrahedronCOM[iat, :] - cavity_com[:, icavity]
         CavityGyrationTensor[:,:,icavity] += (delaunay_volume[iat]*numpy.outer(dx,dx) /cavity_volume[icavity])

 # output the location and the characteristics of the cavities
 hstring = 'cavity2_'+str(iconf)+'.txt'
 with open(hstring,'w') as h: # output file each line corresponds to one cavity entry 

  h.write("{} \n".format((cavity_volume > 0).sum()))

  # compute the asphericity and eigenvalues of each gyration tensor
  for icount, icavity in enumerate(TrueCavities):
    [u, v] =  numpy.linalg.eigh(CavityGyrationTensor[:, :, icavity])
    u = numpy.sort(u) 
    SqRg = u[0] + u[1] + u[2]
    Acylindricity = u[1] - u[0]
    if SqRg > 0: 
     Asphericity = (u[2] - 0.5*(u[0] + u[1])) 
     Anisotropy  = 1.5*(u[0]*u[0] + u[1]*u[1] + u[2]*u[2])/pow(u[0]+u[1]+u[2],2) - 0.5   
    else: 
     Asphericity = 1.0
     Anisotropy  = 1.0

    h.write("{} {} {} {} {} {} {} {} {} {} {} {} \n".format(icount, cavity_com[0,icavity], cavity_com[1,icavity], cavity_com[2,icavity], cavity_volume[icavity], u[2], u[1], u[0], SqRg, Asphericity, Anisotropy, Acylindricity))

  fourth_timing = datetime.datetime.now()
  print("It took {} seconds for clustering with triangle method".format((fourth_timing-third_timing).seconds))