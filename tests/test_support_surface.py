"""A tolerated penetration into a vertical face is not a standable top surface."""
import numpy as np
from plan.geometry import Terrain


def test_vertical_tile_face_stays_wall_without_becoming_ground():
    words=np.zeros((12,12),np.uint32);words[:,5]=1
    ter=Terrain(words,0.,0.)
    # Native contact and the planner's tolerated 0.85-unit overlap into x=100.
    g,side=ter.support(np.array([93.99,94.85,125.15]),np.array([80.,80.,80.]))
    assert g.tolist()==[False,False,False]
    assert side.tolist()==[1,1,-1]


def test_floor_and_wall_corner_still_has_a_standable_foot():
    words=np.zeros((12,12),np.uint32);words[:,5]=1;words[0,:]=1
    ter=Terrain(words,0.,0.)
    g,side=ter.support(np.array([94.85,110.,70.]),np.array([26.,246.,26.]))
    assert g.tolist()==[True,True,True]
    assert side[0]==1


def test_platform_sides_and_interior_are_not_top_support():
    ter=Terrain(np.zeros((12,12),np.uint32),0.,0.)
    bar=[(100.,100.,30.,10.,False)]
    g,side=ter.support(np.array([64.85,100.,100.]),np.array([100.,100.,116.]),bar)
    assert g.tolist()==[False,False,True]
    assert side[0]==1
    # A separate real platform can support a body next to a tile wall.
    words=np.zeros((12,12),np.uint32);words[:,5]=1
    ter=Terrain(words,0.,0.)
    assert ter.support(np.array([94.85]),np.array([116.]),bar)[0][0]
