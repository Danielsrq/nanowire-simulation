import kwant
import numpy as np
import tinyarray as ta

# eVm # physical_constants['Bohr magneton'][0]
bohr_magneton = 58E-6
# angstroms # 6.0583E-10 # 20E-3 # might need to change this.
lattice_constant_InAs = 200

s0 = np.identity(2)
sZ = np.array([[1.0, 0.0], [0.0, -1.0]])
sX = np.array([[0.0, 1.0], [1.0, 0.0]])
sY = np.array([[0.0, -1j], [1j, 0.0]])

tauZ = ta.array(np.kron(sZ, s0))
tauX = ta.array(np.kron(sX, s0))
tauY = ta.array(np.kron(sY, s0))
sigZ = ta.array(np.kron(s0, sZ))
sigX = ta.array(np.kron(s0, sX))
sigY = ta.array(np.kron(s0, sY))
tauZsigX = ta.array(np.kron(sZ, sX))
tauZsigY = ta.array(np.kron(sZ, sY))
tauYsigY = ta.array(np.kron(sY, sY))

lat = kwant.lattice.square(a=lattice_constant_InAs, norbs=4)


def hopX(site0, site1, t, alpha):
    # print('hopx', -t * tauZ + 1j * alpha * tauZsigY)
    return -t * tauZ + 1j * alpha * tauZsigY


def hopY(site0, site1, t, alpha):
    # print('hopy', -t * tauZ - 1j * alpha * tauZsigX)
    return -t * tauZ - 1j * alpha * tauZsigX


def sinuB(theta, stagger_ratio):
    # ssin, scos = rick_fourier(theta)
    # return sigY*scos + sigX*ssin
    return sigY * np.cos(theta) + sigX * np.sin(theta)
    # This is the onsite Hamiltonian, this is where the B-field can be varied.


def onsiteSc(site, muSc, t, B, Delta, M, addedSinu,
             barrier_length, stagger_ratio, user_B, period):
    gfactor = 10  # should be 10 in the real units
    if addedSinu == "sine":
        # Note that site.pos has real units i.e. 200A so all terms
        # must be in Angstroms
        counter = np.mod(site.pos[0] - (1 + barrier_length) *
                         lattice_constant_InAs, period)
        theta = (counter/period) * 2 * np.pi
        # print ("Counter: " + str(counter))
        # print ("theta: " + str(theta))
        # if -1 < counter < 4:
        #     theta = 0
        # elif 3 < counter < 8:
        #     theta = 0.2 * (counter - 3) * np.pi
        # elif 7 < counter < 12:
        #     theta = np.pi
        # else:
        #     theta = 0.2 * (counter - 6) * np.pi

        return (
            (4 * t - muSc) * tauZ
            + 0.5 * gfactor * bohr_magneton * B * sigX
            + Delta * tauX
            + 0.5 * gfactor * bohr_magneton * M * sinuB(theta, stagger_ratio)
        )

    elif addedSinu == "user1D":
        """ Assumes 2D array where 1st row is Bx and 2nd row By """
        current_xsite = site.pos[0]
        return (
            (4 * t - muSc)*tauZ + (gfactor*bohr_magneton*B*sigX) + Delta*tauX
            + user_B[0][current_xsite] * sigX
            + user_B[1][current_xsite] * sigY)

    elif addedSinu == "user2D":
        """ Assumes 2D array which maps onto the space of the field such that
        (i,j) is the Bvector as position (i,j); user_B[i][j][0] and
        user_B[i][j][1] are Bx and By"""
        current_xsite, current_ysite = int(site.pos[0]/lattice_constant_InAs), int(site.pos[1]/lattice_constant_InAs)
        # print (current_xsite, current_ysite)
        # print (user_B[current_xsite][current_ysite][0], user_B[current_xsite][current_ysite][1])
        return (
            (4 * t - muSc)*tauZ + (0.5*gfactor*bohr_magneton*B*sigX) + Delta*tauX
            + 0.5*gfactor*bohr_magneton*user_B[current_xsite][current_ysite][0] * sigX
            + 0.5*gfactor*bohr_magneton*user_B[current_xsite][current_ysite][1] * sigY)

    else:
        return (
            (4 * t - muSc) * tauZ
            + 0.5 * gfactor * bohr_magneton * B * sigX
            + Delta * tauX
        )


def onsiteNormal(site, mu, t):
    # print('onsitnormal', (4 * t - mu) * tauZ)
    return (4 * t - mu) * tauZ


def onsiteBarrier(site, mu, t, barrier):
    # print('onsitebarrier', (4 * t - mu + barrier) * tauZ)
    return (4 * t - mu + barrier) * tauZ


def make_lead(width, onsiteH=onsiteNormal, hopX=hopX, hopY=hopY):
    lead = kwant.Builder(
        kwant.TranslationalSymmetry((-lattice_constant_InAs, 0)),
        conservation_law=-tauZ,
        particle_hole=tauYsigY,
    )
    lat = kwant.lattice.square(a=lattice_constant_InAs, norbs=4)
    lead[(lat(0, j) for j in range(width))] = onsiteH
    lead[kwant.builder.HoppingKind((1, 0), lat, lat)] = hopX
    lead[kwant.builder.HoppingKind((0, 1), lat, lat)] = hopY
    return lead


def barrier_region(site, barrier_length, length, width):
    i = site // width
    j = site % width
    return(
        (
            (0 <= i < barrier_length)
            or
            (length - barrier_length <= i < length)
        )
        and
        (
            0 <= j < width
        )
      )


def make_wire(width, length, barrier_length,
              hamiltonian_wire=onsiteSc, hamiltonian_barrier=onsiteBarrier,
              hamiltonian_normal=onsiteNormal, hopX=hopX, hopY=hopY):

    syst = kwant.Builder()

    syst[
        (
            lat(i, j)
            for i in range(barrier_length, length - barrier_length)
            for j in range(width)
        )
    ] = onsiteSc

    syst[
        (
            lat(i, j)
            for i in range(0, barrier_length + 1)
            for j in range(width)
        )
    ] = onsiteBarrier

    syst[
        (
            lat(i, j)
            for i in range(length - barrier_length, length)
            for j in range(width)
        )
    ] = onsiteBarrier

    # Hopping:
    syst[kwant.builder.HoppingKind((1, 0), lat, lat)] = hopX
    syst[kwant.builder.HoppingKind((0, 1), lat, lat)] = hopY
    lead = make_lead(width, onsiteNormal, hopX, hopY)
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())

    return syst


def NISIN(width=7, length=56, barrier_length=1):
    # length = 8 * noMagnets - 2 + 2 * barrier_length  # "2*(noMagnets - 1)"
    length = length
    return make_wire(width, length, barrier_length).finalized()
