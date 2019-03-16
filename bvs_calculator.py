import numpy as np
import itertools as itr
import ase.io
import ase
import time
import re
from numpy import unravel_index as unrav, ravel_multi_index as rav
import mendeleev as chem



# functions
def return_value(start_string,main_text):
    return re.search(start_string+"""(.*?)\n""",main_text).group(1).strip(" '")

def ChemFormula(cif):
    with open(cif) as f:
        fileCotents = f.read()
        chemicalFormula_aliasList = [
            '_chemical_formula_sum',
            '_chemical_formula_structural',
            '_pd_phase_name']
        chemical_formula = None
        for alias in chemicalFormula_aliasList:
            if alias in fileCotents:
                chemical_formula = return_value(alias,fileCotents)
                break
        assert chemical_formula != None, 'chemical formula alias list not comprehenisve'
    print(chemical_formula)
    return chemical_formula

def max_electronegativity(chemical_formula):
    'returns the element symbol with the max electronegatvity'
    element_list = re.sub("[\.\(\)0-9]",'',chemical_formula).split()
    electronegativities = [chem.element(i).electronegativity() for i in element_list]
    return element_list[electronegativities.index(max(electronegativities))]

halogens = 'F Cl Br I At'.split()
def find(element_name,chemical_formula):
    'checks if an element is in a chemical formula'
    el = re.compile('{eln}[0-9A-Z]|{eln}[0-9A-Z]*$'.format(eln=element_name))
    return bool(el.search(chemical_formula))
def guess_anions(chemform):
    anions = [x for x in halogens if find(x,chemform)]
    maxelec = max_electronegativity(chemform)
    if maxelec not in anions: 
        anions.append(maxelec)
    return anions
    

def diffMap(cif, Ro = 1.1745, b = 0.514, co = 6., dx=0.4, dy = 0.4, dz=0.4, mesh = None, anion = None ):
    
    a = ase.io.read(cif)
    formula = ChemFormula(cif)
    counter_ion_label = max_electronegativity(formula)
    counter_ions = a[a.numbers==chem.element(counter_ion_label).atomic_number]
    counter_ion_positions = counter_ions.positions

    lat = np.linalg.norm(a.cell,axis=1)
    shape = tuple( int(round(i)) for i in lat/np.array([dx,dy,dz]) )
    V = np.ones(shape)

    counter_ion_scaled_positions = counter_ions.get_scaled_positions().copy()
    connectivity = [np.array(i) for i in itr.product([-1,0,1],repeat=len(V.shape))]
    counter_ion_scaled_positions = np.concatenate([counter_ion_scaled_positions + con for con in connectivity])

    ctIons = np.einsum('ij,kj->ik', a.cell.T,counter_ion_scaled_positions).T
    
    counter_ion_positions = counter_ions.positions
    
    for i in range(len(V.flatten())):
        ri = np.dot(a.cell.T,(2*np.array(unrav(i,V.shape))+1.)/(2*np.array([V.shape]).astype(float))[0])
        cti = ctIons[np.all(((ctIons < ri + co) & (ctIons > ri - co)),axis=1)] # counter ions
        # put in & instaed of == ...
        Ri = np.linalg.norm(cti-ri,axis=1)
        V[unrav(i,V.shape)] = np.abs( np.sum( np.exp((Ro-Ri[Ri<co])/b) ) - 1 )

    output_filename = cif.split('.')[0] + '.grd'
    with open(output_filename,"w") as savefile:
        savefile.write("Bond Valence Sum Difference\r") # Title
        from ase.geometry import cell_to_cellpar
        cellParams = cell_to_cellpar(a.cell) # get ABC alpha, beta, gamma
        savefile.write(" ".join([str(k) for k in cellParams])+"\r" )
        savefile.write(" ".join([str(k) for k in V.shape])+"\r" )
        for i in np.nditer(V.flatten()):
            savefile.write("%.6f  "%(i)) # Write each valence difference value
            
    return V


def diffMap_0(input_filename,Ro,b,co=6., grid_size=None,dx=0.4,dy=0.4,dz=0.4,anion='O',anion_number=8):

    output_filename = input_filename.split('.')[0]

    # Read in the CIF file
    a = ase.io.read("{}".format(input_filename))
    a = a[a.numbers==8]

    lat = np.linalg.norm(a.cell,axis=1)

    if grid_size != None:
        dX,dY,dZ = np.around( np.ones(3)*grid_size / lat, 2)
    else:
        dX,dY,dZ = np.around( np.array([dx,dy,dz]) / lat, 2)
    # Make a mesh over the resolution defined by dx,dy,dz
    x,y,z = np.mgrid[0:1+dX:dX, 0:1+dY:dY,0:1+dZ:dZ]
    r_scaled = np.stack([x,y,z]) # Cartesian coordinates of each voxel

    # Transform the unit cell to access each voxel by real lengths
    r = np.dot(r_scaled.reshape((3,np.array(r_scaled.shape[1:]).prod())).T,a.cell).T.reshape(r_scaled.shape)

    # Make an empty array to calculate the Valence over
    V = np.ones(x.shape) 

    # Define oxygen positions
    O = a[a.numbers==8].get_positions()
    # Add coordinates of all anions in adjacent cells 
    # (important if the interaction length is larger than the unit cell)
    connectivity = [i for i in itr.product([-1,0,1],repeat=len(V.shape)) if i != (0,0,0)]
    for i in connectivity:
        if i != (0,0,0):
            shift = a.cell[0]*i[0]+a.cell[1]*i[1]+a.cell[2]*i[2]
            a.translate(shift)
            O = np.concatenate((O,a.get_positions()),axis=0)
            a.translate(-shift)


    # optionally, time this section because it is the time critical step
    start_time=time.time()

    # Iterate through each volume element in the unit cell
    lens = [range(r.shape[1]),range(r.shape[2]),range(r.shape[3])]
    for i in itr.product(*lens):

        # define upper and lower bound for anion coordinates as deffined by the cuttoff radii (co)
        top = r[:,i[0],i[1],i[2]]+co 
        bottom = r[:,i[0],i[1],i[2]]-co

        # Store all oxygen within a box around the sphere defined by the cuttoff radii
        O2 = O[np.all(((O < top) == (O > bottom)),axis=1)]

        # Calculate the distance to each anion in the box
        Ri = np.array([ np.sqrt( (r[0,i[0],i[1],i[2]]-O2[k,0])**2 + # This way worked faster
                                (r[1,i[0],i[1],i[2]]-O2[k,1])**2 +  # than with linalg.norm()
                                (r[2,i[0],i[1],i[2]]-O2[k,2])**2)
                       for k in range(len(O2[:,0])) ])

        # Calculate the valence sum and apply the cutoff
        V[i[0],i[1],i[2]] = np.abs( np.sum( np.exp((Ro-Ri[Ri<co])/b) ) - 1 )


    # Save the information
    with open('{0}.grd'.format(output_filename),"w") as savefile:
    
        savefile.write("Bond Valence Sum Difference\r") # Title
        from ase.geometry import cell_to_cellpar
        cellParams = cell_to_cellpar(a.cell) # get ABC alpha, beta, gamma
        savefile.write(" ".join([str(k) for k in cellParams])+"\r" )
        savefile.write(" ".join([str(k) for k in V.shape])+"\r" )
        for i in np.nditer(V.flatten()):
            savefile.write("%.6f  "%(i)) # Write each valence difference value

    ##print "Total time taken = %.4f s" % (time.time()-start_time)
    return V


def diffMap2(input_filename,Ro,b,co,dx,dy,dz,output_filename,anion='O',anion_number=8):

    # Read in the CIF file
    a = ase.io.read("{}".format(input_filename))

    # Take only the anion positions and unit cell vectors
    a = ase.Atoms([anion for i in range(len(a[a.numbers==anion_number]))],
              cell = a.cell,
             positions = a.positions[a.numbers==anion_number],
             pbc = True)

    # Make a mesh over the resolution defined by dx,dy,dz
    x,y,z = np.mgrid[0:1+dx:dx, 0:1+dy:dy,0:1+dz:dz]
    r_scaled = np.stack([x,y,z]) # Cartesian coordinates of each voxel

    # Transform the unit cell to access each voxel by real lengths
    r = np.dot(r_scaled.reshape((3,r_scaled.shape[1]**3)).T,a.cell).T.reshape(r_scaled.shape)

    # Make an empty array to calculate the Valence over
    V = np.ones(x.shape) 

    # Define oxygen positions
    O = a.get_positions()
    # Add coordinates of all anions in adjacent cells 
    # (important if the interaction length is larger than the unit cell)
    permutations = [[-1,0,1] for i in range(3)] 
    for i in itr.product(*permutations):
        if i != (0,0,0):
            shift = a.cell[0]*i[0]+a.cell[1]*i[1]+a.cell[2]*i[2]
            a.translate(shift)
            O = np.concatenate((O,a.get_positions()),axis=0)
            a.translate(-shift)


    # optionally, time this section because it is the time critical step
    start_time=time.time()

    # Iterate through each volume element in the unit cell
    lens = [range(r.shape[1]),range(r.shape[2]),range(r.shape[3])]
    for i in itr.product(*lens):

        # define upper and lower bound for anion coordinates as deffined by the cuttoff radii (co)
        top = r[:,i[0],i[1],i[2]]+co 
        bottom = r[:,i[0],i[1],i[2]]-co

        # Store all oxygen within a box around the sphere defined by the cuttoff radii
        O2 = O[np.all(((O < top) == (O > bottom)),axis=1)]

        # Calculate the distance to each anion in the box
        Ri = np.array([ np.sqrt( (r[0,i[0],i[1],i[2]]-O2[k,0])**2 + # This way worked faster
                                (r[1,i[0],i[1],i[2]]-O2[k,1])**2 +  # than with linalg.norm()
                                (r[2,i[0],i[1],i[2]]-O2[k,2])**2)
                       for k in range(len(O2[:,0])) ])

        # Calculate the valence sum and apply the cutoff
        V[i[0],i[1],i[2]] = np.abs( np.sum( np.exp((Ro-Ri[Ri<co])/b) ) - 1 )


    # Save the information
    with open('{0}.grd'.format(output_filename),"w") as savefile:
    
        savefile.write("Bond Valence Sum Difference\r") # Title
        from ase.geometry import cell_to_cellpar
        cellParams = cell_to_cellpar(a.cell) # get ABC alpha, beta, gamma
        savefile.write(" ".join([str(k) for k in cellParams])+"\r" )
        savefile.write(" ".join([str(k) for k in V.shape])+"\r" )
        for i in np.nditer(V.flatten()):
            savefile.write("%.6f  "%(i)) # Write each valence difference value

    #print "Total time taken = %.4f s" % (time.time()-start_time)
    return V

def valence_map(input_filename,anion,anion_number,Ro,b,co,dx,dy,dz,output_filename):

    # Read in CIF file
    a = ase.io.read("{}.cif".format(input_filename)) 

    # 
    a = ase.Atoms([anion for i in range(len(a[a.numbers==anion_number]))],
              cell = a.cell,
             positions = a.positions[a.numbers==anion_number],
             pbc = True)
    x,y,z = np.mgrid[0:1+dx:dx, 0:1+dy:dy,0:1+dz:dz] # Make mesh
    # Here we will take the mesh and transform it to fit the unit cell in real space
    r = np.stack([x,y,z]) # Gives the cartesian coordinates of each voxel
    lens = [range(r.shape[2]),range(r.shape[3])] # List to iterate over all y,z columns
    for j in itr.product(*lens):
        r[:,:,j[0],j[1]] = np.dot(r[:,:,j[0],j[1]].T,a.cell).T # Transform to coordinates of unit cell 
    # make an empty array to hold values of the Valence difference    
    V = np.ones(x.shape) 

    O = a.get_positions()[a.numbers==8]
    #print O.shape # coordinates of all anions in the unit cell
    # Add coordinates of all anions in adjacent cells
    permutations = [[-1,0,1] for i in range(3)]
    for i in itr.product(*permutations):
        if i != (0,0,0):
            shift = a.cell[0]*i[0]+a.cell[1]*i[1]+a.cell[2]*i[2]
            a.translate(shift)
            O = np.concatenate((O,a.get_positions()),axis=0)
            a.translate(-shift)

    start_time=time.time() # We're going to time this section because it's the long part
    # Iterate through each volume element (the index of which is contained in lens)
    lens = [range(r.shape[1]),range(r.shape[2]),range(r.shape[3])]
    for i in itr.product(*lens): # iterate through each voxel in the unit cell
        top = r[:,i[0],i[1],i[2]]+co # define upper and lower bound for anion coordinates
        bottom = r[:,i[0],i[1],i[2]]-co # that fall within the cutoff radii of the voxel
        # Apply cuttoff 'box' on anions to reduce the number of distance calculations
        O2 = O[np.all(((O < top) == (O > bottom)),axis=1)] # Contains anions in cutoff box
        # Calculate the distance to each anion in the box
        Ri = np.array([ np.sqrt( (r[0,i[0],i[1],i[2]]-O2[k,0])**2 + # This way worked faster
                                (r[1,i[0],i[1],i[2]]-O2[k,1])**2 +  # than with linalg.norm()
                                (r[2,i[0],i[1],i[2]]-O2[k,2])**2)
                       for k in range(len(O2[:,0])) ])
        # Calculate the valence sum and apply the cutoff
        V[i[0],i[1],i[2]] = np.sum( np.exp((Ro-Ri[Ri<co])/b) )

    savefile = open('{0}.grd'.format(output_filename),"w") # outputfile
    savefile.write("Bond Valence Sum Difference\r") # Title
    from ase.geometry import cell_to_cellpar
    cellParams = cell_to_cellpar(a.cell) # get ABC alpha, beta, gamma
    savefile.write(" ".join([str(k) for k in cellParams])+"\r" )
    savefile.write(" ".join([str(k) for k in V.shape])+"\r" )
    for i in np.nditer(V.flatten()):
        savefile.write("%.6f  "%(i)) # Write each valence difference value
    savefile.close()

    return r,V

    #print "Total time taken = %.4f s" % (time.time()-start_time)


